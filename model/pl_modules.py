import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio, torchvision
from torchaudio.transforms import MelSpectrogram
import torchmetrics

import pytorch_lightning as pl 

import numpy as np
import matplotlib.pyplot as plt

from model import *
from utils import *


class Classifier(pl.LightningModule):
    """docstring for Trainer"""
    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self.model = eval(cfg.train.model_name)(cfg)
        self.valid_acc = torchmetrics.Accuracy()
        self.cfg = cfg
        

    def forward(self, x):
        self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch 
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        return {"loss": loss}


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)

        loss = F.cross_entropy(logits, y)
        self.valid_acc(logits, y)

        self.log('val_loss', loss)
        self.log('val_acc', self.valid_acc)
        return  {"loss": loss}


    def test_step(self, batch, batch_idx, plot=True):
        """
        batch: (batch_size, channels, wav_len)
        """
        wav, file_name = batch
        logits_clips = torch.Tensor()
        clip_samples = int(self.cfg.dataset.clip_dur * self.cfg.dataset.sample_rate)
        for i in range(0, wav.shape[-1], clip_samples):

            wav_clip = torch.mean(wav[:, :, i:i+clip_samples], axis=1) # downmix to mono
            melspec = MelSpectrogram(n_mels=self.cfg.dataset.n_mels)(wav_clip)
            if (melspec.shape[-2:] == (self.cfg.dataset.n_mels, int(clip_samples / self.cfg.dataset.hop_len) + 1)):
                logits = self.model(melspec)
                logits_clips = torch.cat((logits_clips, F.normalize(logits)))


        if postprocess:
            postprocess(logits_clips, file_name)

        if plot:
            plt.figure(figsize=(10, 2))
            plt.imshow(logits_clips.T, aspect='auto')
            plt.xlabel("time (seconds)")
            n_clips = int(wav.shape[-1] / clip_samples)
            plt.xticks(np.arange(0, n_clips, step=100), np.arange(0, n_clips, step=100) * self.cfg.dataset.clip_dur)
            plt.ylabel("classification")
            plt.yticks([0, 1], ["solo piano", "non-solo"])
            plt.title(file_name[0])
            plt.savefig(f"{file_name[0][:-4]}.png")

        return 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.lr)
        return optimizer



class SimCLR(pl.LightningModule):
    def __init__(self, cfg, hidden_dim=16, temperature=0.07):
        super().__init__()

        self.cfg = cfg
        # Base model f(.)
        self.save_hyperparameters()
        self.convnet = torchvision.models.resnet18(
            pretrained=False, num_classes=4 * hidden_dim
        )  # num_classes is the output size of the last linear layer

        # change the number of input channels to 1
        self.convnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)

        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.train.lr, weight_decay=1e-4)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=500, eta_min=self.cfg.train.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode="train"):
        imgs, _ = batch
        # imgs: (batch_size, n_views, n_mel, windows)

        imgs = torch.cat(tuple(imgs.transpose(1, 0)), dim=0).unsqueeze(1)
        # imgs: (batch_size * n_views, 1, n_mel, windows)

        # Encode all images
        feats = self.convnet(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)

        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)

        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")

