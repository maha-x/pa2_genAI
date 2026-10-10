import torch
import torch.nn.functional as F

IGNORE_INDEX = -100  # PyTorch's "skip this position" label


def _encode(tokenizer, text):
    """Text -> list of token ids. No special tokens: we add EOS ourselves."""
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        out = tokenizer(text, add_special_tokens=False)
        ids = getattr(out, "input_ids", None)
        if ids is None:
            ids = out["input_ids"] if hasattr(out, "keys") else out
    return [int(t) for t in ids]


def build_sft_batch(tokenizer, prompts, responses, max_length):
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0

    rows = []
    for prompt, response in zip(prompts, responses):
        p_ids = _encode(tokenizer, prompt)
        r_ids = _encode(tokenizer, response)
        if eos_id is not None:
            r_ids = r_ids + [eos_id]            # add EOS first ...
        ids = (p_ids + r_ids)[:max_length]      # ... then truncate
        labels = ([IGNORE_INDEX] * len(p_ids) + r_ids)[:max_length]
        rows.append((ids, labels))

    n = len(rows)
    longest = max(len(ids) for ids, _ in rows)
    input_ids = torch.full((n, longest), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n, longest), dtype=torch.long)
    labels_t = torch.full((n, longest), IGNORE_INDEX, dtype=torch.long)
    for i, (ids, labels) in enumerate(rows):  # row order is preserved
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :L] = 1
        labels_t[i, :L] = torch.tensor(labels, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels_t}


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def response_logprobs_from_batch(model, batch):
    """Returns (sum of log-probs over response tokens [B], number of response tokens [B])."""
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits if hasattr(out, "logits") else out
    logits = logits[:, :-1, :]            # position t predicts token t+1
    targets = batch["labels"][:, 1:]
    token_logps = -F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,        # prompt + padding contribute exactly 0
        reduction="none",
    ).view(targets.shape)
    n_tokens = (targets != IGNORE_INDEX).sum(dim=-1)
    return token_logps.sum(dim=-1), n_tokens


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    batch = _to_device(batch, _model_device(model))
    seq_logps, _ = response_logprobs_from_batch(model, batch)
    return seq_logps  # shape [B]


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Scalar loss = mean over ALL unmasked response tokens in the batch."""
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    batch = _to_device(batch, _model_device(model))
    seq_logps, n_tokens = response_logprobs_from_batch(model, batch)
    return -seq_logps.sum() / n_tokens.sum().clamp(min=1)
