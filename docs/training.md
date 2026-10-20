# NTPP Training and Evaluation Guide

## Environment

Run commands from the repository root. The supported language-model backbones
are Llama and Phi; unsupported architectures fail explicitly.

Clone with the MOSS source submodule, or initialize it after cloning:

```bash
git submodule update --init --recursive
```

Install PyTorch and torchaudio builds that match your CPU/CUDA environment,
then install the remaining dependencies:

```bash
python -m pip install -r requirement.txt
```

Transformers is pinned to 4.46.3 for compatibility with the model code. WAV
input/output uses SoundFile, so TorchCodec is not required. Moss tokenizer
source comes from the custom [yoospeech/moss_nano_trainer_unofficial GitHub
repository](https://github.com/yoospeech/moss_nano_trainer_unofficial), included
as a submodule. The pretrained codec checkpoint is loaded separately from
Hugging Face as `youspeech/moss_nano_trainer_unofficial`; the weights are not
bundled in this repository. The MOSS source code carries its own Apache-2.0
license.

The default training path does not use DeepSpeed. Existing scripts under
`scripts/continue_pretrain/` and configs under `ntpp/config/ds_*.json` are
legacy examples and are not used by the commands below.

## Data format

Each JSONL row must contain an `audio_filepath` pointing to a WAV file:

```json
{"audio_filepath": "/absolute/path/to/audio.wav"}
```

- Stage 1 uses a single speaker track. For stereo input it reads channel 1.
- Stage 2 uses two separate, time-aligned speaker tracks in a stereo WAV. A
  generic stereo recording is not necessarily a valid two-speaker example.
- If `--validation_file` is omitted, the first 5% of manifest rows are used
  for validation and the remaining 95% for training. Provide a separate
  validation manifest with `--validation_file` if desired.
- Both stages must use the same MOSS checkpoint and RVQ depth.
- BOS/EOS tokens are not inserted. Long files are indexed by WAV header and
  codec frame rate, then read and encoded as windows in the DataLoader. Windows
  from different conversations are never concatenated. A final remainder
  shorter than two codec frames is dropped.
- `block_size` must be a multiple of D in stage 1 or 2D in stage 2, and must
  contain at least two frames. With D=4, 1024 is valid for both stages.

## Online MOSS extraction

With `--use_moss_audio True`, both stages follow this path:

```text
manifest -> WAV header/window index -> DataLoader.__getitem__ -> read window
-> resample -> frozen Moss.encode -> token IDs -> padding -> language model
```

Initialization reads only codec settings and WAV headers. Codec weights load
lazily in each process on first sample access and are frozen. A window is
re-encoded when it is read again in a later epoch; token IDs are not cached.
Long recordings are split into windows no larger than `block_size`, avoiding a
full-file encode on each access. The codec context restarts at each window, so
token values near window boundaries can differ from encoding the full file
before splitting it.

GPU extraction is the default: `--moss_device cuda --dataloader_num_workers 0`.
CUDA extraction must run in the main DataLoader process. For CPU extraction
with workers, use `--moss_device cpu --dataloader_num_workers 2`; workers use
`spawn` and each loads its own codec weights. On DDP, `cuda` resolves to the
rank's current GPU. The codec and language model share device memory.

## Stage 1: audio pretraining

The wrapper below supplies the recommended stage settings and supports extra
Trainer/MOSS arguments after `OUT_DIR`:

```bash
bash scripts/train_audio.sh pretrain /path/to/Llama-or-Phi \
  /path/to/train.jsonl output/stage1
```

Equivalent direct invocation:

```bash
python -m ntpp.train.pretrain \
  --model_name_or_path /path/to/Llama-or-Phi \
  --train_file /path/to/train.jsonl \
  --output_dir output/stage1 \
  --use_moss_audio True \
  --moss_model_name_or_path youspeech/moss_nano_trainer_unofficial \
  --moss_device cuda --dataloader_num_workers 0 \
  --moss_num_codebooks 4 \
  --block_size 1024 \
  --do_train --do_eval \
  --bf16 True \
  --gradient_checkpointing True \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 4e-4 \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs '{"min_lr": 4e-6}' \
  --max_steps 40000 \
  --eval_strategy steps --eval_steps 1000 \
  --save_steps 1000 --save_total_limit 3 \
  --validation_split_percentage 5 \
  --max_eval_samples 10 \
  --logging_steps 10 --report_to tensorboard
```

These defaults target a single GPU. Effective batch size is the number of GPUs
multiplied by per-device batch size and gradient accumulation. Adjust batch
size and accumulation to available memory and dataset size. The default
learning-rate range and 40,000 stage-1 steps follow Appendix C of the paper;
the wrapper settings are practical defaults, not a guarantee of reproducing
the published experiment.

## Stage 2: dual-speaker supervised fine-tuning

Stage 2 starts from the stage-1 checkpoint and requires stereo files with
separated, aligned speaker tracks:

```bash
bash scripts/train_audio.sh sft output/stage1 \
  /path/to/dual_speaker_train.jsonl output/stage2
```

Use `scripts/train_audio_small_batch.sh` instead to run with a smaller
per-device batch size and accumulation settings. The script definitions are
the source of truth for the exact values. The default SFT wrapper uses a
learning rate of `5e-5` and one epoch.

## Train and validation loss in TensorBoard

The training scripts write training loss every 10 optimizer steps and
validation loss every 1,000 optimizer steps, using at most 10 validation
examples. A final evaluation is also recorded when training ends. If no
`--validation_file` is set, 5% of manifest rows are assigned to validation;
both stages use the same split. If fewer than 10 validation examples exist,
all available examples are evaluated.

```bash
tensorboard --logdir output --port 6006
```

Open <http://localhost:6006> and select the Scalars tab. The main tags are
`train/loss` and `eval/loss`. Event files are written to
`output/stage1/tensorboard` and `output/stage2/tensorboard`. Add an argument
such as `--eval_steps 100` to a training command to evaluate more often.
TensorFlow is not required; the PyTorch Trainer writes TensorBoard event files.

## Offline inference

Run the stage-2 conditional generator with a time-aligned stereo WAV:

```bash
python -m ntpp.inference.streaming \
  --model output/stage2 \
  --moss-model youspeech/moss_nano_trainer_unofficial \
  --input-wav /path/to/aligned_stereo.wav \
  --output-wav output/generated_b.wav \
  --rvq-depth 4 --prompt-frames 5 --chunk-frames 5 \
  --device cuda
```

Channel 1 (A) is the observed input speaker; channel 2 (B) is the response to
generate. To test a question, put the question in channel 1 and silence in
channel 2. The first `--prompt-frames` from both channels form the initial
context. After that prefix, each remaining A frame is read and a B frame is
generated. The input must contain more codec frames than `--prompt-frames`.
At the default 12.5 codec frames per second, five frames are about 0.4 seconds;
they are not five seconds.

The runner encodes the complete WAV before generation and then processes the
remaining frames in chunks using a single KV cache. It is an offline
chunked-generation simulation, not a live microphone or playback service. It
does not wait for a question to finish before starting the response: generated
B frames align with the A frames after the prompt prefix. Implementing
turn-end detection and delayed response generation requires additional logic.
The checkpoint must be stage 2 and must use the same MOSS checkpoint and RVQ
depth as training.

`model.generate()` accepts complete-frame prompts and generates both channels.
`max_new_tokens` counts flattened tokens, not frames; for stage 2 it must be a
multiple of `2D`. Greedy decoding and temperature/top-k/top-p sampling are
supported. Beam search and EOS-based early stopping are not supported. When
MOSS token metadata is saved in the checkpoint, generation restricts tokens
to the valid vocabulary for each RVQ depth.

## Verification and limitations

The test suite uses small random Llama/Phi models to check mask causality,
checkpointing output and gradients, cache/full-prefix equivalence, VQ/RVQ
generation, stage-checkpoint compatibility, depth-vocabulary restrictions,
and chunk-size invariance. CLI integration tests exercise preprocessing and
evaluation for both model families and stages with test tokenizers. They do not
run optimizer steps.

Separate smoke tests loaded the cached MOSS weights, encoded and decoded
synthetic audio, and exercised online extraction and evaluation for both
stages. A GPU smoke test also ran one optimizer step, evaluation, and checkpoint
save for each stage with a real 1B model. These short tests do not establish
long-run training stability, memory use at the default `block_size=1024`, audio
quality, or paper-level results. The current MOSS implementation also does not
reproduce the paper's codec and training corpus exactly.
