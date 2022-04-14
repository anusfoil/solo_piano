import torch 
import numpy as np


def postprocess(logits_clips, file_name):
    predictions = logits_clips.max(dim=1)[1]
    """a list of segment """
    label_groups = np.split(predictions, np.where(np.diff(predictions) != 0)[0]+1)

    N = len(predictions)

    """see if there are trailing segment of 1 (applauses)"""
    max_len, max_start, max_end = 5, 0, 0
    timestamp = 0
    for segment in label_groups:
        if segment[0] and len(segment) >= max_len:
            max_len = len(segment)
            max_start, max_end = timestamp, timestamp+len(segment)
        timestamp += len(segment)

    """applause at the beginning"""
    start, end = 0, N
    if max_end < N / 3:
        start = max_end
    """applause at the end"""
    if (max_start > (N * (2 / 3))):
        end = max_start

    """see if there are non-solo segements distributed across the whole duration"""
    predictions = predictions[start:end]
    if sum(predictions) / len(predictions) > 0.18:
        print(f"{file_name} might not be solo. ")
    else:
        wav = wav[0, :, start*clip_samples : end*clip_samples]
        torchaudio.save(f"{file_name[0]}", wav, self.cfg.dataset.sample_rate)
        print(f"Save cutted audio to {file_name[0]}")
