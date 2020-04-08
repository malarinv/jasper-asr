# import math

# import librosa
import torch
import pickle
# import torch.nn as nn
# from torch_stft import STFT

# from nemo import logging
from nemo.collections.asr.parts.perturb import AudioAugmentor
# from nemo.collections.asr.parts.segment import AudioSegment


class RpycWaveformFeaturizer(object):
    def __init__(
        self, sample_rate=16000, int_values=False, augmentor=None, rpyc_conn=None
    ):
        self.augmentor = augmentor if augmentor is not None else AudioAugmentor()
        self.sample_rate = sample_rate
        self.int_values = int_values
        self.remote_path_samples = rpyc_conn.get_path_samples

    def max_augmentation_length(self, length):
        return self.augmentor.max_augmentation_length(length)

    def process(self, file_path, offset=0, duration=0, trim=False):
        audio = self.remote_path_samples(
            file_path,
            target_sr=self.sample_rate,
            int_values=self.int_values,
            offset=offset,
            duration=duration,
            trim=trim,
        )
        return torch.tensor(pickle.loads(audio), dtype=torch.float)

    def process_segment(self, audio_segment):
        self.augmentor.perturb(audio_segment)
        return torch.tensor(audio_segment, dtype=torch.float)

    @classmethod
    def from_config(cls, input_config, perturbation_configs=None):
        if perturbation_configs is not None:
            aa = AudioAugmentor.from_config(perturbation_configs)
        else:
            aa = None

        sample_rate = input_config.get("sample_rate", 16000)
        int_values = input_config.get("int_values", False)

        return cls(sample_rate=sample_rate, int_values=int_values, augmentor=aa)
