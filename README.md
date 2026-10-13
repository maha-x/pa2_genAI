# Instruction Tuning and DPO on a Small Language Model

Course project: post-train `Qwen/Qwen2.5-0.5B` with supervised fine-tuning (SFT) and two **from-scratch** Direct Preference
Optimization (DPO) runs (HH-RLHF and UltraFeedback), and run a separate **TRL + LoRA** persona-tuning experiment on
`HuggingFaceTB/SmolLM2-360M-Instruct`. Models are compared with a pairwise LLM judge on fixed prompt sets.
Everything ran on Kaggle (2 x T4). The assignment text is in `docs/assignment_manual.pdf`; the final report is in `report/`.

> The from-scratch SFT and DPO code does **not** use `Trainer`, TRL's `DPOTrainer`, or any prebuilt DPO loop.
> Only the LoRA task uses TRL `DPOTrainer` + PEFT, as the assignment allows.

## Headline results

| Comparison (judge: `google/gemini-3.6-flash`) | n | Win / Tie / Loss | Win % excl. ties [95% CI] | Ties as half |
|---|---|---|---|---|
| DPO-HH vs SFT (HH helpful/harmless set, safety prompt) | 300 | 149 / 45 / 106 | 58.4 [52.4, 64.5] | 57.2 |
| DPO-UF vs SFT (AlpacaEval subset, helpfulness prompt) | 300 | 114 / 40 / 146 | 43.8 [37.8, 49.9] | 44.7 |

LoRA persona (submitted setting: lr 1e-4, 3 epochs, beta 0.3) against the plain SmolLM2 model, 30 prompts, ties as half:
style 66.7%, usefulness 40.0%, safety 45.0%. Across five tried settings, no setting gained style without losing usefulness
or safety. All numbers come from a single seed; see the report for intervals, the judge's position bias, and caveats.

## Repository layout

```text
submission_adapter.py        graded interface; thin wrappers around src/core.py
src/
  core.py                    batching, response-only masking, log-probs, SFT loss, DPO loss
  data.py                    shared prompt template
  sft.py                     SFT training loop (AMP fp16, grad accumulation, clipping, best/final checkpoints)
  dpo.py                     from-scratch DPO (policy on cuda:0, frozen reference on cuda:1)
  pref_data.py               HH-RLHF and UltraFeedback loaders (single-turn filtering)
  lora_dpo.py                TRL DPOTrainer + LoRA persona training
  generate.py                generation for Qwen models (SFT / DPO / base)
  generate_lora.py           generation for SmolLM2 with or without a LoRA adapter
  build_pairs.py             randomised A/B pair files with a hidden model mapping
  make_eval_subset.py        AlpacaEval 300-prompt subset (seed 42)
  plot_metrics.py, plot_dpo.py   training plots
scripts/
  judge_pairwise_kaggle.py   resumable pairwise judge (modes: helpful, safety, style, useful, safe)
  make_persona_eval.py       30-prompt persona style evaluation set
  check_pref_data.py         quick check of the preference-data loaders
  make_report_figures.py     builds the report's training-curve figures from the metrics files
tests/                       public adapter smoke tests (toy tokenizer and model, no GPU needed)
data/persona_lora/           provided persona preference data (220 pairs)
eval_sets/                   fixed evaluation prompts (AlpacaEval subset, HH set, persona set)
outputs/                     configs, metrics, plots, generations, pair files, judged results
report/                      final report (LaTeX source, figures, PDF)
docs/assignment_manual.pdf   assignment text
```

Model checkpoints are **not** committed (`outputs/checkpoints/` is git-ignored).

## Environments

Two library setups were used on purpose; the report lists the exact versions.

**SFT, DPO, generation (Transformers < 5).** Kaggle notebook, `GPU T4 x2`. Keep Kaggle's preinstalled PyTorch.
```bash
python -m pip uninstall -y -q torchao || true
python -m pip install -q -r requirements.txt
python -m pip install -q -e . --no-deps
```
Versions used: PyTorch 2.11, Transformers 4.57.6, Datasets 4.8.5.

**LoRA task (Transformers 5.x, TRL).** `requirements.txt` does not include TRL. Use Kaggle's preinstalled Transformers 5.x and PEFT,
add TRL, and remove the stale `torchao` build that conflicts with PEFT. Do not run `pip install -r requirements.txt` here.
```bash
python -m pip install -q trl
python -m pip uninstall -y -q torchao || true
```
Versions used: Transformers 5.16.1, TRL 1.15.0, PEFT 0.20.0. TRL's `DPOTrainer` refuses multi-GPU `DataParallel`, so run it with
`CUDA_VISIBLE_DEVICES=0`.

Run every command from the repository root with `export PYTHONPATH=.` (the editable install does not package `src/`).

## Reproducing the pipeline

Approximate run times on Kaggle T4s: SFT 108 min (one GPU); DPO-HH 99 min and DPO-UF 132 min (two GPUs); LoRA run about 2 min;
generation 5-10 min per model and prompt set; judging about 10 min per 300 pairs.

```bash
export PYTHONPATH=.

# 0. Public tests (expect: 7 passed)
python -m pytest -q tests/test_public_adapter_smoke.py

# 1. Evaluation sets
python -m src.make_eval_subset                 # eval_sets/alpaca_eval_300_seed42.json (300 of 805, random.Random(42))
python scripts/make_persona_eval.py            # eval_sets/persona_style_eval_30.json
python scripts/check_pref_data.py hh 15000 1000   # optional loader check; also try: uf 15000 1000

# 2. Baseline generations (plain greedy, base model)
python -m src.generate --eval_path eval_sets/alpaca_eval_300_seed42.json \
  --out_path outputs/generations/base_alpaca_eval.json --generator_name qwen2.5-0.5b-base

# 3. SFT: 30,000 UltraChat examples, effective batch 32 (1 x 32), lr 2e-5, cosine, 1 epoch
python -m src.sft --micro_bs 1 --accum 32
python -m src.plot_metrics --metrics outputs/sft/metrics.jsonl --out outputs/sft/sft_curves.png

# 4. From-scratch DPO (needs 2 GPUs), beta 0.1, lr 1e-6 fixed, RMSprop, effective batch 32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m src.dpo --dataset hh --sft_path outputs/checkpoints/sft/best
python -m src.dpo --dataset uf --sft_path outputs/checkpoints/sft/best
python -m src.plot_dpo --metrics outputs/dpo_hh/metrics.jsonl --out outputs/dpo_hh/dpo_hh_curves.png --title "DPO (hh)"
python -m src.plot_dpo --metrics outputs/dpo_uf/metrics.jsonl --out outputs/dpo_uf/dpo_uf_curves.png --title "DPO (uf)"

# 5. Generations for the comparisons: identical decoding for every model
#    (greedy, repetition penalty 1.1, 256 new tokens)
EXTRA="--repetition_penalty 1.1"
python -m src.generate --model_path outputs/checkpoints/sft/best    --eval_path eval_sets/alpaca_eval_300_seed42.json   --out_path outputs/generations/sft_on_alpaca.json    --generator_name sft    $EXTRA
python -m src.generate --model_path outputs/checkpoints/dpo_uf/best --eval_path eval_sets/alpaca_eval_300_seed42.json   --out_path outputs/generations/dpo_uf_on_alpaca.json --generator_name dpo_uf $EXTRA
python -m src.generate --model_path outputs/checkpoints/sft/best    --eval_path eval_sets/hh_rlhf_eval_300_seed42.json  --out_path outputs/generations/sft_on_hh.json         --generator_name sft    --dataset_name hh_rlhf_style_eval $EXTRA
python -m src.generate --model_path outputs/checkpoints/dpo_hh/best --eval_path eval_sets/hh_rlhf_eval_300_seed42.json  --out_path outputs/generations/dpo_hh_on_hh.json      --generator_name dpo_hh --dataset_name hh_rlhf_style_eval $EXTRA

# 6. Pair files (random A/B order, seed 42; the model mapping is stored as hidden metadata)
mkdir -p outputs/eval
python -m src.build_pairs --sft_path outputs/generations/sft_on_alpaca.json --dpo_path outputs/generations/dpo_uf_on_alpaca.json \
  --out_path outputs/eval/dpo_uf_vs_sft_pairs.jsonl --sft_name sft --dpo_name dpo_uf
python -m src.build_pairs --sft_path outputs/generations/sft_on_hh.json --dpo_path outputs/generations/dpo_hh_on_hh.json \
  --out_path outputs/eval/dpo_hh_vs_sft_pairs.jsonl --sft_name sft --dpo_name dpo_hh
```

### Judging (Kaggle benchmark notebook)

The judge calls need Kaggle's model proxy, which is configured in a notebook created at
<https://www.kaggle.com/benchmarks/tasks/new> (the `kaggle_benchmarks` library is preinstalled there). In an ordinary notebook the
library can be installed (`pip install kaggle-benchmarks "protobuf==5.29.6"`) but model access is not set up.
Use `--dry_run` to test the pipeline without any API call. Judging is resumable (`--reuse_judgments`), sends 5 pairs per request,
waits at least 5 s between requests, and retries a batch up to 3 times.

```bash
python scripts/judge_pairwise_kaggle.py --pairs_path outputs/eval/dpo_uf_vs_sft_pairs.jsonl \
  --output_path outputs/eval/dpo_uf_vs_sft_judged.jsonl --judge_model google/gemini-3.6-flash \
  --mode helpful --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments --baseline_name sft

python scripts/judge_pairwise_kaggle.py --pairs_path outputs/eval/dpo_hh_vs_sft_pairs.jsonl \
  --output_path outputs/eval/dpo_hh_vs_sft_judged.jsonl --judge_model google/gemini-3.6-flash \
  --mode safety --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments --baseline_name sft \
  --meta_path eval_sets/hh_rlhf_eval_300_seed42.json
```
Each run writes `*_judged.jsonl`, `*_judged_summary.json` (wins, ties, losses, win rates, failures, per-category results) and
`*_judged_judge_info.json` (the judge model actually used). The `--baseline_name` must match a name in the pair file;
the script stops with an error otherwise.

### LoRA persona task (TRL + PEFT)

Submitted setting (rank 16, alpha 32, all seven linear projections, lr 1e-4, 3 epochs, beta 0.3, batch 4 x 4):
```bash
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python -m src.lora_dpo --lr 1e-4 --epochs 3 --beta 0.3 \
  --out_dir outputs/lora_persona_beta03 --adapter_dir outputs/checkpoints/lora_persona_beta03

python -m src.generate_lora --repetition_penalty 1.1 \
  --out_path outputs/generations/persona_base_rp11.json --generator_name smollm2-360m-instruct-rp11
python -m src.generate_lora --repetition_penalty 1.1 --adapter_path outputs/checkpoints/lora_persona_beta03 \
  --out_path outputs/generations/persona_lora_beta03_rp11.json --generator_name persona_lora_beta03_rp11
python -m src.build_pairs --sft_path outputs/generations/persona_base_rp11.json \
  --dpo_path outputs/generations/persona_lora_beta03_rp11.json \
  --out_path outputs/eval/lora_beta03_rp11_vs_base_pairs.jsonl --sft_name base --dpo_name lora_beta03_rp11

# three separate judge questions; the baseline is named "base" here
for mode in style useful safe; do
  python scripts/judge_pairwise_kaggle.py --pairs_path outputs/eval/lora_beta03_rp11_vs_base_pairs.jsonl \
    --output_path outputs/eval/lora_beta03_rp11_${mode}_judged.jsonl --judge_model google/gemini-3.6-flash \
    --mode $mode --reuse_judgments --baseline_name base
done
```
Five settings were trained and judged; the submitted one was chosen by a rule fixed before judging (highest style score among
settings whose usefulness and safety are both at least 40%, ties as half).

| Setting | Output folder | lr | epochs | beta |
|---|---|---|---|---|
| pilot 1 | `outputs/lora_persona/` | 5e-5 | 3 | 0.1 |
| mid_a | `outputs/lora_persona_mid_a/` | 7e-5 | 3 | 0.1 |
| mid_b | `outputs/lora_persona_mid_b/` | 1e-4 | 3 | 0.1 |
| pilot 2 | `outputs/lora_persona_pilot2/` | 1e-4 | 5 | 0.1 |
| **beta 0.3 (submitted)** | `outputs/lora_persona_beta03/` | 1e-4 | 3 | 0.3 |

## Where to find results

| What | Path |
|---|---|
| Run configuration, metrics, plots | `outputs/sft/`, `outputs/dpo_hh/`, `outputs/dpo_uf/`, `outputs/lora_persona*/` (`config.json`, `metrics.jsonl`, `summary.json`, `*_curves.png`) |
| Generations | `outputs/generations/` (`sft_on_*`, `dpo_*`, `persona_*`, `base_alpaca_eval.json`, `decoding_settings.json`) |
| Pair files and judged outputs | `outputs/eval/*_pairs.jsonl`, `*_judged.jsonl`, `*_judged_summary.json`, `*_judge_info.json` |
| Decoding used for SFT/DPO comparisons | `outputs/generations/decoding_settings.json` |
| Implementation | `src/core.py`, `src/sft.py`, `src/dpo.py`, `src/lora_dpo.py`, `submission_adapter.py` |

## Submission adapter and public tests

`submission_adapter.py` keeps the six required signatures and calls into `src/core.py`:
```bash
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py     # 7 passed
```
Design choices that the manual leaves open:
- **Padding:** batches are padded to the longest row in the batch (never beyond `max_length`); longer rows are truncated to `max_length`.
- **EOS:** appended to every response *before* truncation, so a truncated response loses its EOS.
- **Masking:** labels are `-100` on prompt and padding tokens; log-probabilities are summed over response tokens only.
- **SFT loss reduction:** mean over all unmasked response tokens in the batch.
- **DPO:** the reference model is computed under `torch.no_grad()`; `build_dpo_batch` returns `{"chosen": ..., "rejected": ...}`, each with
  `input_ids`, `attention_mask` and `labels`. The training code additionally cuts DPO prompts to their last 256 tokens (an optional
  argument of the core function that the adapter wrapper does not pass).

## Fixed files

- `eval_sets/hh_rlhf_eval_300_seed42.json`: the provided synthetic helpful/harmless evaluation set (300 prompts, 10 categories of 30).
- `eval_sets/alpaca_eval_300_seed42.json`: 300 of the 805 AlpacaEval prompts, drawn with `random.Random(42)` by `src/make_eval_subset.py`.
- `eval_sets/persona_style_eval_30.json`: 30 persona prompts we wrote (8 cooking, 6 coding, 8 general help, 8 safety-sensitive).
- `data/persona_lora/ramsay_style_preferences.json`: the provided persona preference data (200 train / 20 held-out pairs after a seeded shuffle).

## Checkpoints

Checkpoints are too large for git. The SFT best checkpoint was kept in a private Kaggle notebook output and the DPO checkpoints in
private Hugging Face repositories, so they are not available to others; every checkpoint can be regenerated with the commands above.
Only the pilot-1 LoRA adapter was archived; the judged generations and the configuration of every LoRA setting are in `outputs/`.

## Known limitations

- Single seed for every run; run-to-run variance is unknown.
- One judge model and one answer order per pair. The judge favours the first-listed answer (see the report's position table).
- Small evaluation sets: 30 prompts per HH category and 30 persona prompts.
- Results hold for greedy decoding with repetition penalty 1.1; other decoding settings were not tested for the DPO comparisons.
- SFT answers longer than 512 tokens lose their EOS token (allowed by the manual), which probably explains answers that never stop.
- The first 108-minute SFT run lost its checkpoints when an interactive Kaggle session ended before a notebook version was saved; it was rerun
  with identical code and settings, and the rerun is the SFT model used throughout.

## Report

`report/report.pdf` (source: `report/report.tex`, figures: `report/fig_*.pdf`). It follows the structure required by the manual:
repository and data, configuration, deviations, training behaviour, DPO-versus-SFT results, interpretation, limitations.

<!-- Optional disclosure: uncomment and edit only if your course policy requires it.
## Acknowledgements
An AI assistant helped with debugging, planning experiments and drafting the report; all runs and numbers come from our own experiments.
-->

