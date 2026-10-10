from datasets import load_dataset


def _parse_hh(text):
    """'\\n\\nHuman: Q\\n\\nAssistant: A' -> (Q, A) if single-turn, else None."""
    if text.count("\n\nHuman:") != 1 or text.count("\n\nAssistant:") != 1:
        return None
    left, answer = text.split("\n\nAssistant:")
    question = left.split("\n\nHuman:")[1]
    return question.strip(), answer.strip()


def load_hh(split, n, seed):
    ds = load_dataset("Anthropic/hh-rlhf", split=split).shuffle(seed=seed)
    out = []
    for ex in ds:
        c, r = _parse_hh(ex["chosen"]), _parse_hh(ex["rejected"])
        if c is None or r is None or c[0] != r[0]:
            continue
        if not c[1] or not r[1] or c[1] == r[1]:
            continue
        out.append({"prompt": c[0], "chosen": c[1], "rejected": r[1]})
        if len(out) == n:
            break
    return out


def load_uf(split, n, seed):
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split).shuffle(seed=seed)
    out = []
    for ex in ds:
        ch, rj = ex["chosen"], ex["rejected"]
        if len(ch) != 2 or len(rj) != 2:
            continue
        if ch[0]["role"] != "user" or ch[1]["role"] != "assistant" or rj[1]["role"] != "assistant":
            continue
        prompt = ex["prompt"].strip()
        c, r = ch[1]["content"].strip(), rj[1]["content"].strip()
        if not prompt or not c or not r or c == r:
            continue
        out.append({"prompt": prompt, "chosen": c, "rejected": r})
        if len(out) == n:
            break
    return out


# name -> (loader, train split, validation split)
DATASETS = {
    "hh": (load_hh, "train", "test"),
    "uf": (load_uf, "train_prefs", "test_prefs"),
}
