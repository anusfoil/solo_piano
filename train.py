import sys, os
sys.path.extend(['model'])

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torchmetrics

import hydra
from omegaconf import DictConfig

import numpy as np
import matplotlib.pyplot as plt

from data_loader import SPNS, SPNS_test
from model import * 
from pl_modules import * 



@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):

    """preprare datasets"""
    spns_train = SPNS(cfg, split="Train")
    spns_valid = SPNS(cfg, split="Valid")
    spns_train_dl = DataLoader(spns_train, batch_size=cfg.train.batch_size, shuffle=True)
    spns_valid_dl = DataLoader(spns_valid, batch_size=cfg.train.batch_size, shuffle=False)

    spns_test_dl = DataLoader(SPNS_test(), batch_size=1, shuffle=False)


    """logger and checkpointer"""
    wandb_logger = WandbLogger()

    """init and train"""
    pl.seed_everything(42)  # To be reproducable
    model = eval(cfg.train.method)(cfg)
    trainer = pl.Trainer(
        max_epochs=500, 
        logger=wandb_logger, 
        callbacks=[
            ModelCheckpoint(monitor=cfg.train.check_monitor)
        ],
        progress_bar_refresh_rate=1
        )
    trainer.fit(model, spns_train_dl, spns_valid_dl)


    """load and test"""
    # results = trainer.test(model=classifier, dataloaders=spns_test_dl, ckpt_path="solo_piano_detection/experiments/baseline_cnn/2022-04-10_18-28-59/uncategorized/3nwfn4n6/checkpoints/epoch=462-step=44911.ckpt")
    results = trainer.test(model=classifier, dataloaders=spns_test_dl, ckpt_path="solo_piano_detection/experiments/baseline_musicnn/2022-04-10_18-26-16/uncategorized/3q64bpuq/checkpoints/epoch=488-step=47433.ckpt")



if __name__ == '__main__':

    main()





