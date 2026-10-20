import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from ntpp.train.online_audio import OnlineMossDataset


class FakeCodec:
    sample_rate = 16000
    downsample_rate = 1280
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def encode_codes(self, audio):
        FakeCodec.calls += 1
        frames = audio['array'].shape[-1] * self.sample_rate // audio['sampling_rate'] // self.downsample_rate
        return torch.stack([torch.arange(4)[:, None].expand(4, frames) + c for c in range(2)])

    def encode_single_codes(self, audio):
        return self.encode_codes(audio)[0]


class OnlineAudioTests(unittest.TestCase):
    def test_lazy_extraction_and_depth_channel_order(self):
        with tempfile.TemporaryDirectory() as directory, patch('ntpp.train.online_audio.MossAudioTokenEncoder', FakeCodec):
            wav = Path(directory)/'stereo.wav'
            sf.write(wav, np.zeros((8000, 2)), 8000)
            FakeCodec.calls = 0
            table = [[8*q+i for i in range(8)] for q in range(4)]
            for stage, block, expected in [('pretrain',16,[0,9,18,27]),('ntpp',32,[0,9,18,27,1,10,19,28])]:
                before = FakeCodec.calls
                ds = OnlineMossDataset([{'audio_filepath':str(wav)}], 'unused', table, block, stage)
                self.assertEqual(FakeCodec.calls,before)
                self.assertIsNone(ds._encoder)
                self.assertEqual(len(ds),3)
                first = next(iter(DataLoader(ds,batch_size=None,num_workers=0)))
                self.assertEqual(FakeCodec.calls,before+1)
                self.assertEqual(first['input_ids'],expected*4)
                self.assertEqual(first['labels'],first['input_ids'])
                again = ds[0]
                self.assertEqual(FakeCodec.calls,before+2)  # No pre-extracted token cache.
                self.assertEqual(first,again)
                self.assertIsNone(ds.__getstate__()['_encoder'])

    def test_mono_rejected_only_for_stage_two(self):
        with tempfile.TemporaryDirectory() as directory, patch('ntpp.train.online_audio.MossAudioTokenEncoder',FakeCodec):
            wav=Path(directory)/'mono.wav';sf.write(wav,np.zeros(8000),8000)
            args=([{'audio_filepath':str(wav)}],'unused',[[i for i in range(8)]]*4,32)
            self.assertGreater(len(OnlineMossDataset(*args,stage='pretrain')),0)
            with self.assertRaisesRegex(ValueError,'requires 2 audio channels'):
                OnlineMossDataset(*args,stage='ntpp')
