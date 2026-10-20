# stage1: Audio modality-adaptation pretraining
import random
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
import math
import tqdm
import glob
import transformers
from transformers import Trainer, AutoTokenizer, HfArgumentParser, TrainingArguments, DataCollatorForSeq2Seq
from ..model import LlamaForCausalLM, PhiForCausalLM
from transformers import AutoConfig
from transformers.trainer_utils import get_last_checkpoint
import os
from .data import split_aligned_examples
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the tokenized data"},
    )

    #tune embedding
    train_embeddings: bool = field(
        default=False,
        metadata={
                "help": (
                    "only train embeddings while training"
                    )
        },
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default="", 
        metadata={"help": "Path to the training data."})
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The input training data file (a text file)."}
        )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=10,
        metadata={
            "help": (
                "Maximum number of validation examples (default: 10)."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    use_text: bool = field(
        default=False, metadata={"help": "Use text data for pretraining"}
    )
    use_moss_audio: bool = field(
        default=False,
        metadata={"help": "Encode manifest waveforms with Moss Audio before tokenization"},
    )
    moss_model_name_or_path: Optional[str] = field(
        default="youspeech/moss_nano_trainer_unofficial",
        metadata={"help": "Moss Audio tokenizer checkpoint"},
    )
    moss_device: str = field(
        default="cuda", metadata={"help": "Online codec device: cpu or cuda (CUDA requires zero DataLoader workers)"}
    )
    moss_num_codebooks: int = field(
        default=4,
        metadata={"help": "Moss RVQ depth per speaker"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    optim: str = field(default="adamw_torch")



def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if data_args.use_moss_audio and training_args.dataloader_num_workers > 0:
        # Forking after LM/CUDA/thread-pool initialization can deadlock codec workers.
        torch.multiprocessing.set_start_method("spawn", force=True)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


    backbone_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    if backbone_config.model_type not in ("llama", "phi"):
        raise ValueError("Supported backbones: llama, phi")
    model_class = PhiForCausalLM if backbone_config.model_type == "phi" else LlamaForCausalLM
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation="sdpa",
    ).to(torch.device(training_args.device))
    model.config.use_cache = False
    model.config.num_channel = 2
    model.config.ntpp_stage = "ntpp"

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "right"  # Preserve frame/depth indices when padding.
    #Extend vocab for speech units
    if '<sosp>' not in tokenizer.get_vocab():
        units_size=1000
        logger.info(f"Add special unit tokens <0>-<{units_size-1} to tokenizer.vocab")
        new_tokens = [f"<{x}>" for x in range(units_size)] + ['<sosp>', '<eosp>']
        tokenizer.add_tokens(new_tokens)

    if data_args.use_moss_audio:
        if int(getattr(model.config, "num_channel", 2)) != 2:
            raise ValueError(
                "Moss dual-speaker input requires an NTPP checkpoint configured with num_channel=2"
            )
        from ..utils.moss_audio import MossAudioTokenEncoder

        if model.config.rvq_depth != data_args.moss_num_codebooks:
            raise ValueError("SFT RVQ depth must match the stage-1 checkpoint")
        saved_codec = getattr(model.config, "moss_model_name_or_path", None)
        if saved_codec is not None and saved_codec != data_args.moss_model_name_or_path:
            raise ValueError("SFT must use the stage-1 Moss checkpoint")
        moss_encoder = MossAudioTokenEncoder(data_args.moss_model_name_or_path, data_args.moss_num_codebooks)
        existing_vocab = tokenizer.get_vocab()
        if any(token not in existing_vocab for token in moss_encoder.token_strings):
            raise ValueError("Stage-1 tokenizer is missing Moss tokens; run Moss pretraining first")
        model.config.rvq_depth = data_args.moss_num_codebooks
        tokenizer.add_special_tokens({"additional_special_tokens": moss_encoder.token_strings})
        model.config.audio_depth_token_ids = [
            [tokenizer.convert_tokens_to_ids(f"<moss_q{q}_{code}>")
             for code in range(moss_encoder.codebook_size)]
            for q in range(moss_encoder.num_codebooks)
        ]
        model.config.moss_model_name_or_path = data_args.moss_model_name_or_path

    #resize embedding
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model_args.train_embeddings:
        logger.info("only update embedding parameters")
        for name, param in model.named_parameters():
            if "embed" not in name:
                param.requires_grad = False


    #data
    data_files = {}
    dataset_args = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    extension = (
        data_args.train_file.split(".")[-1]
        if data_args.train_file is not None
        else data_args.validation_file.split(".")[-1]
    )
    if extension == "txt":
        extension = "text"
    elif extension == "jsonl":
        extension = "json"
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        **dataset_args,
    )
    # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    if "validation" not in raw_datasets.keys():
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{data_args.validation_split_percentage}%:]",
            **dataset_args,
        )
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{data_args.validation_split_percentage}%]",
            **dataset_args,
        )

    
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    frame_size = 2 * model.config.rvq_depth
    if block_size < 2 * frame_size or block_size % frame_size:
        raise ValueError(f"block_size must be a multiple of {frame_size} and at least {2 * frame_size}")

    if data_args.use_moss_audio:
        from .online_audio import OnlineMossDataset
        codec_device = data_args.moss_device
        if codec_device == "cuda":
            codec_device = str(training_args.device)
        if codec_device.startswith("cuda") and training_args.dataloader_num_workers != 0:
            raise ValueError("CUDA online extraction requires --dataloader_num_workers 0")
        lm_datasets = {
            split: OnlineMossDataset(
                rows, data_args.moss_model_name_or_path,
                model.config.audio_depth_token_ids, block_size,
                stage=model.config.ntpp_stage, device=codec_device,
            )
            for split, rows in raw_datasets.items()
        }
    else:
        def tokenize_function(examples):
            output = tokenizer(examples[text_column_name], add_special_tokens=False)
            return output



        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=list(raw_datasets["train" if training_args.do_train else "validation"].features),
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",

                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=list(raw_datasets["train" if training_args.do_train else "validation"].features),
                )

        def group_texts(examples):
            return split_aligned_examples(examples, block_size, frame_size)

        with training_args.main_process_first(desc="grouping texts together"):
            if not data_args.streaming:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",

                )
            else:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                )

    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = torch.utils.data.Subset(train_dataset, range(max_train_samples))


    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = torch.utils.data.Subset(eval_dataset, range(max_eval_samples))


    data_collator = DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=math.lcm(8, frame_size), return_tensors="pt", padding=True
        )

    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=train_dataset if training_args.do_train else None, 
        eval_dataset=eval_dataset if training_args.do_eval else None, 
        data_collator=data_collator
        )
    

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

        

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)



if __name__ == "__main__":
    train()
