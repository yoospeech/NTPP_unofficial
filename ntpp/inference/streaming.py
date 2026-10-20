"""Chunk-wise conditional streaming inference for RVQ NTPP models."""

import argparse
from pathlib import Path
from typing import Iterator, Optional

import torch
import soundfile as sf
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache

from ..model import LlamaForCausalLM, PhiForCausalLM
from ..utils.moss_audio import MossAudioTokenEncoder


class NTPPStreamingGenerator:
    """Generate speaker B while receiving RVQ frames from speaker A.

    A completed frame has the paper's order
    [A_q0, ..., A_qD, B_q0, ..., B_qD]. Within each speaker, q(d) predicts
    q(d+1), and q(D-1) predicts q0 of the next frame. Incoming A codes are
    observed; only B is sampled.
    """

    def __init__(
        self,
        model,
        tokenizer,
        moss_encoder: MossAudioTokenEncoder,
        chunk_frames: int = 5,
        temperature: float = 0.8,
        top_k: int = 50,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.eval().to(self.device)
        self.tokenizer = tokenizer
        self.moss = moss_encoder
        if model.config.ntpp_stage != "ntpp":
            raise ValueError("Streaming requires a stage-2 NTPP checkpoint")
        self.depth = moss_encoder.num_codebooks
        self.stride = 2 * self.depth
        self.chunk_frames = int(chunk_frames)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        if self.chunk_frames < 1:
            raise ValueError("chunk_frames must be positive")
        if getattr(model.config, "num_channel", 2) != 2:
            raise ValueError("Streaming NTPP requires num_channel=2")
        if getattr(model.config, "rvq_depth", self.depth) != self.depth:
            raise ValueError("Model rvq_depth and Moss codebook count must match")

        token_ids = []
        for depth in range(self.depth):
            ids = [
                tokenizer.convert_tokens_to_ids(f"<moss_q{depth}_{code}>")
                for code in range(moss_encoder.codebook_size)
            ]
            if any(token_id == tokenizer.unk_token_id for token_id in ids):
                raise ValueError(f"Tokenizer is missing Moss depth-{depth} tokens")
            token_ids.append(ids)
        self.code_token_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)

    def codes_to_ids(self, codes: torch.Tensor) -> torch.Tensor:
        """Map [depth, frames] codec values to LM token ids."""
        codes = torch.as_tensor(codes, dtype=torch.long, device=self.device)
        if codes.ndim != 2 or codes.shape[0] != self.depth:
            raise ValueError(f"Expected [{self.depth}, frames] codes, got {tuple(codes.shape)}")
        if codes.numel() and (codes.min() < 0 or codes.max() >= self.moss.codebook_size):
            raise ValueError("Moss code is outside the configured codebook")
        depth_ids = torch.arange(self.depth, device=self.device)[:, None]
        return self.code_token_ids[depth_ids, codes]

    def ids_to_codes(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map depth-specific LM ids [depth, frames] back to codec values."""
        token_ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device)
        matches = token_ids[:, :, None] == self.code_token_ids[:, None, :]
        if not torch.all(matches.any(dim=-1)):
            raise ValueError("Generated token does not belong to its Moss depth")
        return matches.to(torch.long).argmax(dim=-1).cpu()

    def _sample_depth(self, logits: torch.Tensor, depth: int) -> torch.Tensor:
        allowed = self.code_token_ids[depth]
        logits = logits.index_select(0, allowed)
        if self.temperature <= 0:
            code = logits.argmax()
        else:
            logits = logits / self.temperature
            if 0 < self.top_k < logits.numel():
                values, indices = torch.topk(logits, self.top_k)
                selected = torch.multinomial(torch.softmax(values, dim=-1), 1)
                code = indices[selected].squeeze(0)
            else:
                code = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(0)
        return allowed[code]

    @torch.inference_mode()
    def stream(
        self,
        user_codes: torch.Tensor,
        prompt_codes: torch.Tensor,
    ) -> Iterator[torch.Tensor]:
        """Yield generated B chunks as [depth, <=chunk_frames] codec values.

        prompt_codes must contain at least one aligned two-speaker frame and
        have shape [2, depth, prompt_frames]. user_codes contains subsequent
        speaker-A frames with shape [depth, frames].
        """
        prompt_codes = torch.as_tensor(prompt_codes, dtype=torch.long)
        user_codes = torch.as_tensor(user_codes, dtype=torch.long)
        if prompt_codes.ndim != 3 or tuple(prompt_codes.shape[:2]) != (2, self.depth):
            raise ValueError(f"prompt_codes must have shape [2, {self.depth}, frames]")
        if prompt_codes.shape[-1] < 1:
            raise ValueError("At least one complete prompt frame is required")
        if user_codes.ndim != 2 or user_codes.shape[0] != self.depth:
            raise ValueError(f"user_codes must have shape [{self.depth}, frames]")

        prompt_a = self.codes_to_ids(prompt_codes[0])
        prompt_b = self.codes_to_ids(prompt_codes[1])
        prompt_ids = torch.cat((prompt_a, prompt_b), dim=0).transpose(0, 1).reshape(1, -1)
        output = self.model(
            input_ids=prompt_ids,
            past_key_values=DynamicCache(),
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        # The final B_q(D-1) state predicts B_q0 of the next frame.
        next_b_logits = output.logits[0, -1]

        for chunk_start in range(0, user_codes.shape[-1], self.chunk_frames):
            chunk = user_codes[:, chunk_start : chunk_start + self.chunk_frames]
            generated_ids = []
            for frame_idx in range(chunk.shape[-1]):
                a_ids = self.codes_to_ids(chunk[:, frame_idx : frame_idx + 1]).squeeze(1)
                # Cache all observed A depths first. Pair masking prevents the
                # current B depths from conditioning on the aligned A frame.
                output = self.model(
                    input_ids=a_ids.unsqueeze(0),
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = output.past_key_values

                b_ids = []
                for depth in range(self.depth):
                    token_id = self._sample_depth(next_b_logits, depth)
                    b_ids.append(token_id)
                    output = self.model(
                        input_ids=token_id.reshape(1, 1),
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
                    past = output.past_key_values
                    next_b_logits = output.logits[0, -1]
                b_ids = torch.stack(b_ids)
                generated_ids.append(b_ids)
            if generated_ids:
                yield self.ids_to_codes(torch.stack(generated_ids, dim=1))


def load_streaming_generator(args):
    moss = MossAudioTokenEncoder(args.moss_model, num_codebooks=args.rvq_depth, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    if config.model_type not in ("llama", "phi"):
        raise ValueError("Supported backbones: llama, phi")
    if getattr(config, "moss_model_name_or_path", args.moss_model) != args.moss_model:
        raise ValueError("Use the same Moss checkpoint as training")
    model_class = PhiForCausalLM if config.model_type == "phi" else LlamaForCausalLM
    model = model_class.from_pretrained(args.model, attn_implementation="sdpa")
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        raise ValueError("Tokenizer vocabulary does not match the trained model")
    return NTPPStreamingGenerator(
        model,
        tokenizer,
        moss,
        chunk_frames=args.chunk_frames,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trained NTPP checkpoint")
    parser.add_argument("--input-wav", required=True, help="Aligned stereo evaluation WAV")
    parser.add_argument("--output-wav", required=True)
    parser.add_argument("--moss-model", default="youspeech/moss_nano_trainer_unofficial")
    parser.add_argument("--rvq-depth", type=int, default=4)
    parser.add_argument("--prompt-frames", type=int, default=5)
    parser.add_argument("--chunk-frames", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    generator = load_streaming_generator(args)
    codes = generator.moss.encode_codes(args.input_wav)
    if codes.shape[-1] <= args.prompt_frames:
        raise ValueError("Input must contain more frames than --prompt-frames")
    prompt = codes[:, :, : args.prompt_frames]
    user = codes[0, :, args.prompt_frames :]
    generated = torch.cat(list(generator.stream(user, prompt)), dim=1)
    audio = generator.moss.decode_codes(generated)
    output = Path(args.output_wav)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), audio.transpose(0, 1).numpy(), generator.moss.sample_rate)
    print(f"saved {generated.shape[-1]} generated frames to {output}")


if __name__ == "__main__":
    main()
