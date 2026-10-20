"""Frame-aligned audio examples, without attention across conversations."""


def split_aligned_examples(examples, block_size, frame_size):
    if frame_size < 1 or block_size < 2 * frame_size or block_size % frame_size:
        raise ValueError("block_size must contain at least two complete frames")
    result = {key: [] for key in examples}
    result["labels"] = []
    for row, ids in enumerate(examples["input_ids"]):
        if len(ids) % frame_size:
            raise ValueError(
                f"Audio row {row} has {len(ids)} tokens, not a multiple of {frame_size}; "
                "check BOS/EOS, whitespace, and RVQ/channel ordering"
            )
        for start in range(0, len(ids), block_size):
            stop = min(start + block_size, len(ids))
            if stop - start < 2 * frame_size:
                continue  # No next-frame target; do not join unrelated conversations.
            for key, rows in examples.items():
                result[key].append(rows[row][start:stop])
            result["labels"].append(ids[start:stop].copy())
    return result
