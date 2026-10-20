# NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction

[![arXiv](https://img.shields.io/badge/arXiv-2506.00975-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.00975)

NTPP is a generative speech language model for dual-channel spoken dialogue. It
uses discrete audio tokens and next-token-pair prediction to model two aligned
speaker streams.

## Repository contents

- `ntpp/`: NTPP models, training, online Moss audio token extraction, and
  offline chunked inference.
- `third_party/moss_nano_trainer_unofficial/`: MOSS Nano tokenizer source,
  included as a Git submodule from the [yoospeech MOSS Nano repository](https://github.com/yoospeech/moss_nano_trainer_unofficial).
- `docs/training.md`: environment, data format, training, evaluation, and
  inference guide.

The MOSS source code is provided by the linked custom repository under its
Apache-2.0 license. The pretrained codec weights are downloaded separately
from Hugging Face as `youspeech/moss_nano_trainer_unofficial`; they are not
stored in this repository.

## Setup

Clone this repository and its MOSS submodule:

```bash
git clone --recurse-submodules https://github.com/yoospeech/NTPP_unofficial.git
cd NTPP_unofficial
```

If the repository was cloned without submodules, initialize them with:

```bash
git submodule update --init --recursive
```

Install PyTorch and torchaudio builds compatible with your CUDA or CPU
environment, then install the remaining project dependencies:

```bash
python -m pip install -r requirement.txt
```

Transformers 4.46.3 is pinned because the model implementation uses its APIs.
The base Llama/Phi checkpoint and the MOSS codec checkpoint must be available
locally or downloadable from Hugging Face. Llama checkpoints may require
Hugging Face access approval.

## Training

Prepare a JSONL manifest with one audio path per line:

```json
{"audio_filepath": "/absolute/path/to/audio.wav"}
```

Stage 1 uses a single speaker track. Stage 2 expects two time-aligned speaker
tracks in stereo WAV files. Use separated tracks; ordinary stereo recordings
are not necessarily suitable. The online DataLoader extracts MOSS tokens on
the GPU by default and does not create a corpus-wide token cache.

Run both stages from the repository root:

```bash
BASE_MODEL=/path/to/local/Llama-or-Phi
TRAIN_MANIFEST=/path/to/train.jsonl

bash scripts/train_audio.sh pretrain \
  "$BASE_MODEL" "$TRAIN_MANIFEST" output/stage1 \
&& bash scripts/train_audio.sh sft \
  output/stage1 "$TRAIN_MANIFEST" output/stage2
```

For a smaller effective batch size, see `scripts/train_audio_small_batch.sh`.
The training script defaults to an automatic 95/5 train/validation split and
caps validation at 10 examples. TensorBoard logs are written under each output
directory:

```bash
tensorboard --logdir output --port 6006
```

Open <http://localhost:6006> and select **Scalars**. See the [training guide](docs/training.md)
for all arguments, data requirements, and evaluation details.

## Offline inference

The inference runner accepts a two-channel aligned WAV and generates B-channel
audio conditioned on A:

```bash
python -m ntpp.inference.streaming \
  --model output/stage2 \
  --input-wav /path/to/aligned_stereo.wav \
  --output-wav output/generated_b.wav \
  --device cuda \
  --prompt-frames 5 \
  --chunk-frames 5
```

Put the observed question or prompt in channel 1 (A) and silence in channel 2
(B). The first `--prompt-frames` of both channels are used as context. After
that prefix, each remaining A frame conditions generation of a B frame. The
WAV must contain more codec frames than `--prompt-frames`.

This command encodes the complete WAV before generating in chunks. It is an
offline chunked-generation test, not a live microphone and playback service.
See the [training guide](docs/training.md) for details.

## Live streaming inference TODO

- Buffer microphone input into codec-compatible windows and incrementally feed
  the resulting A token frames to the generator.
- Buffer generated audio chunks for playback while maintaining codec state and
  continuity across chunk boundaries.
- Connect capture, encoding, generation, and playback with bounded queues.
- Test chunk continuity, startup latency, and behavior under processing delays.

## License

This repository is distributed under the Apache-2.0 license. Third-party
components retain their own licenses; see their included license and notice
files, including the MOSS submodule's Apache-2.0 license.
