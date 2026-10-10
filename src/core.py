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


def build_sft_batch(tokenizer, prompts, responses, max_length, max_prompt_length=None):
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0

    rows = []
    for prompt, response in zip(prompts, responses):
        p_ids = _encode(tokenizer, prompt)
        if max_prompt_length is not None and len(p_ids) > max_prompt_length:
            p_ids = p_ids[-max_prompt_length:]   # keep the END of the prompt
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


# ----------------------------- DPO -----------------------------

def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses,
                    max_length, max_prompt_length=None):
    """Two aligned LM batches: same prompts, chosen answers vs rejected answers."""
    return {
        "chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length, max_prompt_length),
        "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length, max_prompt_length),
    }


def compute_dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps,
                                reference_chosen_logps, reference_rejected_logps, beta):
    """-log sigmoid(beta * ((pol_c - pol_r) - (ref_c - ref_r))), averaged over the batch."""
    dev, dt = policy_chosen_logps.device, policy_chosen_logps.dtype
    ref_c = reference_chosen_logps.detach().to(device=dev, dtype=dt)    # reference gets no gradient
    ref_r = reference_rejected_logps.detach().to(device=dev, dtype=dt)
    margin = (policy_chosen_logps - policy_rejected_logps) - (ref_c - ref_r)
    return -F.logsigmoid(beta * margin).mean()


def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts,
                     chosen_responses, rejected_responses, beta, max_length):
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)

    pdev = _model_device(policy_model)
    pol_c, _ = response_logprobs_from_batch(policy_model, _to_device(batch["chosen"], pdev))
    pol_r, _ = response_logprobs_from_batch(policy_model, _to_device(batch["rejected"], pdev))

    rdev = _model_device(reference_model)
    with torch.no_grad():                      # frozen reference: no graph, no gradients
        ref_c, _ = response_logprobs_from_batch(reference_model, _to_device(batch["chosen"], rdev))
        ref_r, _ = response_logprobs_from_batch(reference_model, _to_device(batch["rejected"], rdev))

    return compute_dpo_loss_from_logps(pol_c, pol_r, ref_c, ref_r, beta)
