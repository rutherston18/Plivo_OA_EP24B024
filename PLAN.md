# 2,000-Step LLM Speedrun — Plan

Goal: minimize **bits-per-byte (bpb)** on a hidden test file (same style as the
data) under hard caps: ≤2000 optimizer steps, ≤2,000,000 params, only
`train_corpus.txt` for data/tokenizer, pure PyTorch/numpy/stdlib, CPU only.
`python evaluate.py --checkpoint ckpt.pt --text_file <file>` must keep working
and the tokenizer must be lossless (`decode(encode(x)) == x`).

## Baseline (done)
Reproduced in `baseline_results/` → **dev bpb 2.3718** (1.34M params, 2000 steps).
This is the number every experiment must beat. See `baseline_results/BASELINE.md`.

## Method
- Change **one thing at a time**, keep seed fixed, re-score on `dev_eval.txt`.
- Each run ~3 min → budget ~6-10 runs. Watch the loss curve, log every run.
- Track: hypothesis → change → dev bpb before/after → conclusion (in RUNLOG.md).

## Experiment queue (highest expected payoff first)
1. **LR schedule + warmup**: cosine decay, ~100-step warmup, raise peak LR (e.g. 1e-3).
   Biggest cheap win — baseline is under-trained with constant 3e-4.
2. **AdamW + weight decay (~0.1) + grad clip (1.0)**: stability, better generalization.
3. **Weight tying** (`tie_weights=True`): frees ~40K params, often lowers bpb.
4. **Tokenizer → byte-level BPE trained only on the corpus** (target vocab ~1–4K):
   shortens Hindi/Devanagari sequences dramatically, so the same block covers more
   text. Must stay lossless with byte fallback. Re-tune n_embd to respect param cap.
5. **Reshape the model within param cap**: trade depth/width, try block_size 256,
   principled init (scaled residual init), maybe QK-norm / RMSNorm.
6. **Batch size / grad accumulation**: larger effective batch for steadier steps.

## Validation & guardrails
- After every change, assert: params ≤ 2,000,000, steps ≤ 2000, tokenizer round-trip
  lossless, and `evaluate.py` runs unmodified in the submission folder.
- Keep the best checkpoint; only promote a change if dev bpb improves.

## Deliverables to assemble at the end
- `ckpt.pt` (final), modified `model.py` / `train.py` / `tokenizer.py` + working `evaluate.py`
- `RUNLOG.md` (one entry per run), `NOTES.md` (≤10 sentences), `SUMMARY.html`
