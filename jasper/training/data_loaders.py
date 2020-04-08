from functools import partial
import tempfile

# from typing import Any, Dict, List, Optional

import torch
import nemo

# import nemo.collections.asr as nemo_asr
from nemo.backends.pytorch import DataLayerNM
from nemo.core import DeviceType

# from nemo.core.neural_types import *
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType, LabelsType
from nemo.utils.decorators import add_port_docs

from nemo.collections.asr.parts.dataset import (
    # AudioDataset,
    # AudioLabelDataset,
    # KaldiFeatureDataset,
    # TranscriptDataset,
    parsers,
    collections,
    seq_collate_fn,
)

# from functools import lru_cache
import rpyc
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from .featurizer import RpycWaveformFeaturizer

# from nemo.collections.asr.parts.features import WaveformFeaturizer

# from nemo.collections.asr.parts.perturb import AudioAugmentor, perturbation_types


logging = nemo.logging


class CachedAudioDataset(torch.utils.data.Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, transcripts, and durations (in seconds). Each new line is a
    different sample. Example below:

    {"audio_filepath": "/path/to/audio.wav", "text_filepath":
    "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the
    transcription", offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}

    Args:
        manifest_filepath: Path to manifest json as described above. Can
            be comma-separated paths.
        labels: String containing all the possible characters to map to
        featurizer: Initialized featurizer class that converts paths of
            audio to feature tensors
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include
            in dataset
        max_utts: Limit number of utterances
        blank_index: blank character index, default = -1
        unk_index: unk_character index, default = -1
        normalize: whether to normalize transcript text (default): True
        bos_id: Id of beginning of sequence symbol to append if not None
        eos_id: Id of end of sequence symbol to append if not None
        load_audio: Boolean flag indicate whether do or not load audio
    """

    def __init__(
        self,
        manifest_filepath,
        labels,
        featurizer,
        max_duration=None,
        min_duration=None,
        max_utts=0,
        blank_index=-1,
        unk_index=-1,
        normalize=True,
        trim=False,
        bos_id=None,
        eos_id=None,
        load_audio=True,
        parser="en",
    ):
        self.collection = collections.ASRAudioText(
            manifests_files=manifest_filepath.split(","),
            parser=parsers.make_parser(
                labels=labels,
                name=parser,
                unk_id=unk_index,
                blank_id=blank_index,
                do_normalize=normalize,
            ),
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
        )
        self.index_feature_map = {}

        self.featurizer = featurizer
        self.trim = trim
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.load_audio = load_audio
        print(f"initializing dataset {manifest_filepath}")

        def exec_func(i):
            return self[i]

        task_count = len(self.collection)
        with ThreadPoolExecutor() as exe:
            print("starting all loading tasks")
            list(
                tqdm(
                    exe.map(exec_func, range(task_count)),
                    position=0,
                    leave=True,
                    total=task_count,
                )
            )
        print(f"initializing complete")

    def __getitem__(self, index):
        sample = self.collection[index]
        if self.load_audio:
            cached_features = self.index_feature_map.get(index)
            if cached_features is not None:
                features = cached_features
            else:
                features = self.featurizer.process(
                    sample.audio_file,
                    offset=0,
                    duration=sample.duration,
                    trim=self.trim,
                )
                self.index_feature_map[index] = features
            f, fl = features, torch.tensor(features.shape[0]).long()
        else:
            f, fl = None, None

        t, tl = sample.text_tokens, len(sample.text_tokens)
        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        return f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

    def __len__(self):
        return len(self.collection)


class RpycAudioToTextDataLayer(DataLayerNM):
    """Data Layer for general ASR tasks.

    Module which reads ASR labeled data. It accepts comma-separated
    JSON manifest files describing the correspondence between wav audio files
    and their transcripts. JSON files should be of the following format::

        {"audio_filepath": path_to_wav_0, "duration": time_in_sec_0, "text": \
transcript_0}
        ...
        {"audio_filepath": path_to_wav_n, "duration": time_in_sec_n, "text": \
transcript_n}

    Args:
        manifest_filepath (str): Dataset parameter.
            Path to JSON containing data.
        labels (list): Dataset parameter.
            List of characters that can be output by the ASR model.
            For Jasper, this is the 28 character set {a-z '}. The CTC blank
            symbol is automatically added later for models using ctc.
        batch_size (int): batch size
        sample_rate (int): Target sampling rate for data. Audio files will be
            resampled to sample_rate if it is not already.
            Defaults to 16000.
        int_values (bool): Bool indicating whether the audio file is saved as
            int data or float data.
            Defaults to False.
        eos_id (id): Dataset parameter.
            End of string symbol id used for seq2seq models.
            Defaults to None.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        normalize_transcripts (bool): Dataset parameter.
            Whether to use automatic text cleaning.
            It is highly recommended to manually clean text for best results.
            Defaults to True.
        trim_silence (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        load_audio (bool): Dataset parameter.
            Controls whether the dataloader loads the audio signal and
            transcript or just the transcript.
            Defaults to True.
        drop_last (bool): See PyTorch DataLoader.
            Defaults to False.
        shuffle (bool): See PyTorch DataLoader.
            Defaults to True.
        num_workers (int): See PyTorch DataLoader.
            Defaults to 0.
        perturb_config (dict): Currently disabled.
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            # 'audio_signal': NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # 'a_sig_length': NeuralType({0: AxisType(BatchTag)}),
            # 'transcripts': NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # 'transcript_length': NeuralType({0: AxisType(BatchTag)}),
            "audio_signal": NeuralType(
                ("B", "T"),
                AudioSignal(freq=self._sample_rate)
                if self is not None and self._sample_rate is not None
                else AudioSignal(),
            ),
            "a_sig_length": NeuralType(tuple("B"), LengthsType()),
            "transcripts": NeuralType(("B", "T"), LabelsType()),
            "transcript_length": NeuralType(tuple("B"), LengthsType()),
        }

    def __init__(
        self,
        manifest_filepath,
        labels,
        batch_size,
        sample_rate=16000,
        int_values=False,
        bos_id=None,
        eos_id=None,
        pad_id=None,
        min_duration=0.1,
        max_duration=None,
        normalize_transcripts=True,
        trim_silence=False,
        load_audio=True,
        rpyc_host="",
        drop_last=False,
        shuffle=True,
        num_workers=0,
    ):
        super().__init__()
        self._sample_rate = sample_rate

        def rpyc_root_fn():
            return rpyc.connect(
                rpyc_host, 8064, config={"sync_request_timeout": 600}
            ).root

        rpyc_conn = rpyc_root_fn()

        self._featurizer = RpycWaveformFeaturizer(
            sample_rate=self._sample_rate,
            int_values=int_values,
            augmentor=None,
            rpyc_conn=rpyc_conn,
        )

        def read_remote_manifests():
            local_mp = []
            for mrp in manifest_filepath.split(","):
                md = rpyc_conn.read_path(mrp)
                mf = tempfile.NamedTemporaryFile(
                    dir="/tmp", prefix="jasper_manifest.", delete=False
                )
                mf.write(md)
                mf.close()
                local_mp.append(mf.name)
            return ",".join(local_mp)

        local_manifest_filepath = read_remote_manifests()
        dataset_params = {
            "manifest_filepath": local_manifest_filepath,
            "labels": labels,
            "featurizer": self._featurizer,
            "max_duration": max_duration,
            "min_duration": min_duration,
            "normalize": normalize_transcripts,
            "trim": trim_silence,
            "bos_id": bos_id,
            "eos_id": eos_id,
            "load_audio": load_audio,
        }

        self._dataset = CachedAudioDataset(**dataset_params)
        self._batch_size = batch_size

        # Set up data loader
        if self._placement == DeviceType.AllGpu:
            logging.info("Parallelizing Datalayer.")
            sampler = torch.utils.data.distributed.DistributedSampler(self._dataset)
        else:
            sampler = None

        if batch_size == -1:
            batch_size = len(self._dataset)

        pad_id = 0 if pad_id is None else pad_id
        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            collate_fn=partial(seq_collate_fn, token_pad_value=pad_id),
            drop_last=drop_last,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=1,
        )

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader
