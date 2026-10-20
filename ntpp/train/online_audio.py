"""Lazy waveform windows: codec extraction happens in DataLoader __getitem__."""

import math
import os

import soundfile as sf
import torch
from torch.utils.data import Dataset, get_worker_info

from ..utils.moss_audio import MossAudioTokenEncoder


class OnlineMossDataset(Dataset):
    """Index WAV headers once, read and encode each window on demand.

    No waveform/code cache is written. Each worker owns a lazily loaded frozen
    CPU codec. CUDA extraction requires num_workers=0 (one codec per DDP rank).
    """

    def __init__(self, rows, model_name_or_path, depth_token_ids, block_size,
                 stage="ntpp", device="cpu"):
        if stage not in ("pretrain", "ntpp"):
            raise ValueError("Unknown NTPP stage")
        self.stage = stage
        self.depth = len(depth_token_ids)
        self.channels = 2 if stage == "ntpp" else 1
        self.frame_size = self.channels * self.depth
        if block_size < 2 * self.frame_size or block_size % self.frame_size:
            raise ValueError("block_size must contain at least two complete frames")
        self.max_frames = block_size // self.frame_size
        self.model_name_or_path = model_name_or_path
        self.device = str(device)
        self.token_ids = torch.tensor(depth_token_ids, dtype=torch.long)
        metadata = MossAudioTokenEncoder(model_name_or_path, self.depth, device=device)
        self.sample_rate = metadata.sample_rate
        self.hop = metadata.downsample_rate
        self._encoder = None
        self._encoder_pid = None
        self.windows = []
        for row in rows:
            source = next((row[k] for k in ("audio_filepath", "audio_path", "wav_path", "path", "audio", "text") if k in row), None)
            if isinstance(source, dict):
                source = source.get("path")
            if not isinstance(source, str) or not source:
                raise ValueError("Online extraction requires a waveform file path in each manifest row")
            info = sf.info(source)  # Headers only; no waveform or codec extraction.
            if info.channels < self.channels:
                raise ValueError(f"{source}: stage {stage} requires {self.channels} audio channels")
            total_frames = info.frames * self.sample_rate // (info.samplerate * self.hop)
            for start in range(0, total_frames, self.max_frames):
                frames = min(self.max_frames, total_frames - start)
                if frames < 2:
                    continue
                begin = start * self.hop * info.samplerate // self.sample_rate
                end = math.ceil((start + frames) * self.hop * info.samplerate / self.sample_rate)
                self.windows.append((source, begin, min(end, info.frames), frames))
        if not self.windows:
            raise ValueError("Manifest contains no windows with at least two codec frames")

    def __len__(self):
        return len(self.windows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_encoder"] = None
        state["_encoder_pid"] = None
        return state

    def __getitem__(self, index):
        if self.device.startswith("cuda") and get_worker_info() is not None:
            raise ValueError("CUDA Moss extraction requires dataloader_num_workers=0")
        if self._encoder is None or self._encoder_pid != os.getpid():
            self._encoder = MossAudioTokenEncoder(self.model_name_or_path, self.depth, self.device)
            self._encoder_pid = os.getpid()
        path, start, end, frames = self.windows[index]
        samples, sample_rate = sf.read(path, start=start, stop=end, dtype="float32", always_2d=True)
        audio = {"array": samples.T, "sampling_rate": sample_rate}
        if self.stage == "ntpp":
            codes = self._encoder.encode_codes(audio)[..., :frames]  # [speaker, depth, time]
            if codes.shape[-1] < 2:
                raise ValueError(f"Codec produced fewer than two frames for {path}")
            depths = torch.arange(self.depth)[None, :, None]
            ids = self.token_ids[depths, codes].permute(2, 0, 1).reshape(-1)
        else:
            codes = self._encoder.encode_single_codes(audio)[..., :frames]
            if codes.shape[-1] < 2:
                raise ValueError(f"Codec produced fewer than two frames for {path}")
            depths = torch.arange(self.depth)[:, None]
            ids = self.token_ids[depths, codes].transpose(0, 1).reshape(-1)
        ids = ids.tolist()
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ids.copy()}
