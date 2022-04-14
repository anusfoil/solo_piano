import os, sys
sys.path.extend(['../', './preprocess'])
import numpy as np
import random
import glob
from tqdm import tqdm

import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

import torch, torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram

from torch_audiomentations import Compose, Gain, PolarityInversion, AddColoredNoise, LowPassFilter, HighPassFilter, PitchShift

import hook


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x, sample_rate=16000) for i in range(self.n_views)]


contrast_transforms = Compose(
    transforms=[
        PolarityInversion(),
        AddColoredNoise(),
        Gain(min_gain_in_db=-6.0, max_gain_in_db=0.0),
        HighPassFilter(min_cutoff_freq=200, max_cutoff_freq=1200),
        LowPassFilter(min_cutoff_freq=2200, max_cutoff_freq=4000),
        PitchShift(sample_rate=16000, min_transpose_semitones=-5.0,max_transpose_semitones=5.0)
    ]
)

class SPNS(Dataset):
    def __init__(self, cfg, split='Train'):

        self.solo_piano_files = glob.glob(f"{get_original_cwd()}/SPNS/solo_piano/*.wav")
        self.non_solo_files = glob.glob(f"{get_original_cwd()}/SPNS/non_solo/*.wav")
        
        self.n_clips = int(cfg.dataset.segment_dur / cfg.dataset.clip_dur) - 1
        self.cfg = cfg

        # hook()

        M = int(cfg.dataset.train_ratio * len(self.solo_piano_files))
        N = int(cfg.dataset.train_ratio * len(self.non_solo_files))

        if split == "Train":
            self.solo_piano_files = self.solo_piano_files[:M]
            self.non_solo_files = self.non_solo_files[:N]
        elif split == "Valid":
            self.solo_piano_files = self.solo_piano_files[M:]
            self.non_solo_files = self.non_solo_files[N:]
        
        self.all_files  = self.solo_piano_files + self.non_solo_files


    def __len__(self):
        return len(self.all_files) * self.n_clips

    def __getitem__(self, idx):

        file_path = self.all_files[int(idx / self.n_clips)]
        clip_samples = int(self.cfg.dataset.clip_dur * self.cfg.dataset.sample_rate)
        wav, sr = torchaudio.load(file_path, frame_offset=(idx % self.n_clips)*clip_samples, num_frames=clip_samples, normalize=True)
        wav = torch.mean(wav, axis=0) # downmix to mono

        # if we are doing contrastive experiments
        if self.cfg.dataset.contrastive:
            wav = ContrastiveTransformations(contrast_transforms, n_views=2)(wav.view(1, 1, -1))

        if isinstance(wav, list):
            melspec = MelSpectrogram(n_mels=self.cfg.dataset.n_mels)(torch.stack(wav).squeeze(1).squeeze(1))

        if "non_solo" in file_path:
            label = 1
        else:
            label = 0

        assert(melspec.shape[-2:] == (self.cfg.dataset.n_mels, int(clip_samples / self.cfg.dataset.hop_len) + 1))
        return melspec, label


class SPNS_test(Dataset):
    """SPNS test dataset, consists of full music samples with noisy condition

    returns: full wav 
    """
    def __init__(self):
        self.all_files = glob.glob(f"{get_original_cwd()}/tests/*.wav")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        file_path = self.all_files[idx]
        wav, sr = torchaudio.load(file_path, normalize=True)
        file_name = file_path.split("/")[-1]

        return wav, file_name











