import torch
import torch.nn as nn
import torch.nn.functional as F
import torchex.nn as exnn
import torchvision.models as models

import pytorch_lightning as pl

import numpy as np
from scipy import linalg

from modules import *


class Base_CNN(pl.LightningModule):
    def __init__(self, cfg):
        super(Base_CNN, self).__init__()
        self.conv1 = nn.Conv1d(cfg.dataset.n_mels, 96, 3)
        self.conv2 = nn.Conv1d(96, 48, 3)
        self.conv3 = nn.Conv1d(48, 2, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(96)
        self.bn2 = nn.BatchNorm1d(48)
        self.bn3 = nn.BatchNorm1d(2)
        self.pool1 = nn.MaxPool1d(3)
        self.pool2 = nn.MaxPool1d(3)
        self.pool3 = nn.MaxPool1d(3)
        self.do3 = nn.Dropout(0.5)
        self.ap = exnn.GlobalAvgPool1d()

    def forward(self, x):
        # x: [batch_size, 5, n_mels, timestamps]
        x = self.pool1(F.leaky_relu(
            self.bn1(self.conv1(x)), negative_slope=0.2))
        x = self.pool2(F.leaky_relu(
            self.bn2(self.conv2(x)), negative_slope=0.2))
        x = self.do3(self.pool3(F.leaky_relu(
            self.bn3(self.conv3(x)), negative_slope=0.2)))
        x = self.ap(x)

        # x: [batch_size, 96, 1]
        x = x.squeeze(2)
        return x



class Musicnn(pl.LightningModule):
    """
    Taken from: https://github.com/ilaria-manco/music-audio-tagging-pytorch/blob/master/audio_tagging_pytorch/models/musicnn.py
    
    Base class for network modules in the musicnn model,
    a deep convolutional neural networks for music audio tagging.
    Model architecture and original Tensorflow implementation: Jordi Pons -
    https://github.com/jordipons/musicnn/
    Args:
    - y_input_dim (int): height of the input
    - timbral_k_height (float or list): timbral filter height as a factor of the input dim 
    - temporal_k_width (int or list): temporal filter width of the input dim 
    - filter_factor (float): factor that controls the number of filters (output channels)
    - pool_type: type of pooling in the backend (temporal or attention)
    """
    def __init__(self,
                 cfg,
                 filter_factor=0.1,
                 pool_type='attention',
                 timbral_k_height=0.7,
                 temporal_k_width=7):
        super(Musicnn, self).__init__()
        self.pool_type = pool_type
        y_input_dim = cfg.dataset.n_mels
        self.front_end = FrontEnd(y_input_dim, timbral_k_height, temporal_k_width,
                                  filter_factor)

        # front_end_channels = self.front_end.out_channels
        frontend_output_height = 15
        self.midend = MidEnd(input_channels=frontend_output_height)

        self.midend_channels = (self.midend.num_of_filters*3)+frontend_output_height
        self.back_end = BackEnd(input_height=self.midend_channels, output_units=50, pool_type=self.pool_type)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.front_end(x.unsqueeze(1))
        x = self.midend(x)
        x = self.back_end(x)
        x = self.sigmoid(x)

        return x


