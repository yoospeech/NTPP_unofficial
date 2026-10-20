"""Cached decoding matching the stage-1 NTP and stage-2 RVQ losses."""

import torch
from transformers.cache_utils import DynamicCache


def generate_frames(
    model, input_ids, *, max_new_tokens=None, max_length=None,
    do_sample=False, temperature=1.0, top_k=0, top_p=1.0,
    attention_mask=None, use_cache=True, num_beams=1,
    num_return_sequences=1, depth_token_ids=None, **kwargs,
):
    """Return prompt plus complete frames; lengths count flattened tokens.

    A full, unpadded prompt is required. Stage 2 samples both channels; use
    NTPPStreamingGenerator for observed-A/conditional-B generation. Optional
    depth_token_ids restricts each depth to its codec vocabulary. EOS is not
    used: audio duration is specified by max_new_tokens or max_length.
    """
    if kwargs:
        raise ValueError(f"Unsupported NTPP generation options: {sorted(kwargs)}")
    if not use_cache or num_beams != 1 or num_return_sequences != 1:
        raise ValueError("NTPP generation supports one cached greedy/sampled sequence per prompt")
    if input_ids is None or input_ids.ndim != 2:
        raise ValueError("Supply input_ids with shape [batch, complete-frame tokens]")
    depth = model.config.rvq_depth
    channels = model.config.num_channel if model.config.ntpp_stage == "ntpp" else 1
    stride = channels * depth
    if input_ids.shape[1] < stride or input_ids.shape[1] % stride:
        raise ValueError(f"Prompt must contain complete {stride}-token frames")
    if attention_mask is not None and not torch.all(attention_mask == 1):
        raise ValueError("Use unpadded complete-frame prompts")
    if max_new_tokens is None:
        max_new_tokens = stride if max_length is None else max_length - input_ids.shape[1]
    if max_new_tokens < 0 or max_new_tokens % stride:
        raise ValueError(f"max_new_tokens must be a nonnegative multiple of {stride}")
    if temperature <= 0 or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("Require temperature > 0, top_k >= 0, and 0 < top_p <= 1")
    if not max_new_tokens:
        return input_ids
    if depth_token_ids is None:
        depth_token_ids = getattr(model.config, "audio_depth_token_ids", None)
    if depth_token_ids is not None and len(depth_token_ids) != depth:
        raise ValueError("depth_token_ids must provide one vocabulary per RVQ depth")

    def sample(logits, d):
        allowed = None
        if depth_token_ids is not None:
            allowed = torch.as_tensor(depth_token_ids[d], device=logits.device)
            logits = logits.index_select(-1, allowed)
        if do_sample:
            logits = logits / temperature
            if top_k:
                threshold = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, -torch.inf)
            if top_p < 1:
                ordered, indices = logits.sort(descending=True, dim=-1)
                remove = ordered.softmax(-1).cumsum(-1) > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                logits = logits.scatter(-1, indices, ordered.masked_fill(remove, -torch.inf))
            token = torch.multinomial(logits.softmax(-1), 1)
        else:
            token = logits.argmax(-1, keepdim=True)
        return allowed[token] if allowed is not None else token

    output = model(input_ids=input_ids, past_key_values=DynamicCache(), use_cache=True, return_dict=True)
    past = output.past_key_values
    # Each previous channel's final depth predicts that channel's next q0.
    next_logits = [output.logits[:, -stride + (c + 1) * depth - 1] for c in range(channels)]
    pieces = [input_ids]
    for _ in range(max_new_tokens // stride):
        for c in range(channels):
            logits = next_logits[c]
            for d in range(depth):
                token = sample(logits, d)
                pieces.append(token)
                output = model(input_ids=token, past_key_values=past, use_cache=True, return_dict=True)
                past = output.past_key_values
                logits = output.logits[:, -1]
            next_logits[c] = logits
    return torch.cat(pieces, dim=1)
