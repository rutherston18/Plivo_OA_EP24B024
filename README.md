# 2,000-Step LLM Speedrun

> **📦 Deliverables are in [`./deliverables/`](./deliverables/)** — start with
> [`deliverables/SUMMARY.html`](./deliverables/SUMMARY.html) for the full report, and
> [`deliverables/README.md`](./deliverables/README.md) for how to reproduce the results.
> The submission checklist (SUMMARY.html, RUNLOG.md, NOTES.md, all code, and the `ckpt.pt`
> track output) all live in that folder.

A from-scratch language model for mixed **English + Hindi** text, trained under hard limits:
**≤ 2000 optimizer steps, ≤ 2,000,000 parameters, CPU-only, pure PyTorch/numpy/stdlib, one
training corpus, a lossless tokenizer.** Scored by **bits-per-byte (bpb)** on held-out text.

## Result

| | Baseline | Best (E6) |
|---|---|---|
| **dev bpb** | 2.3718 | **1.8225** (**−23.2%**) |
| params | 1,339,840 | 1,884,160 |
| steps | 2000 | 2000 |

Reproduce in ~10 seconds:

```bash
cd deliverables/code
python evaluate.py --checkpoint ../ckpt.pt --text_file ../data/dev_eval.txt
# -> {"bpb": 1.8225, ...}
```

## How I got there

I treated the caps as three sub-games — **sample efficiency** (only 2000 steps),
**bytes-per-token** (bpb is per byte; Devanagari costs 3–4 bytes/char), and **parameter
efficiency** — and ran a ladder of controlled experiments, each scored by the official
metric and snapshotted for reproducibility:

| Exp | Change | dev bpb | vs baseline |
|-----|--------|--------:|------------:|
| Baseline | starter GPT, byte tokenizer, Adam const-LR | 2.3718 | — |
| **E1** | trapezoidal LR + AdamW + grad clip | 2.1653 | −8.7% |
| **E2** | RoPE + RMSNorm + QK-norm + ReLU² + zero-init + softcap | 2.0421 | −13.9% |
| **E3** | Muon optimizer (Newton–Schulz) | 1.9516 | −17.7% |
| **E4** | byte-level BPE-1024 + weight tying | 1.8583 | −21.6% |
| **E5** | SwiGLU + deeper stack (reinvest freed params) | 1.8385 | −22.5% |
| **E6** | hashed n-gram (Engram) embeddings — **best** | **1.8225** | **−23.2%** |

*(E1–E3 numbers as scored in `RUNLOG.md`.)* Two findings stood out: at a 2000-step budget the
**optimizer and tokenizer** do most of the work, and **capacity is not the bottleneck** —
E6's richer input representation beat E5's deeper, higher-parameter model.

**Future work (E7–E8):** a selective SSM (Mamba-style) backbone for linear-time, cache-friendly
CPU inference (implemented + param-verified), and hierarchical byte patching (BLT/SpaceByte).
See the Future Work and References sections of the report.

## Repo layout

| Path | What |
|------|------|
| [`deliverables/`](./deliverables/) | **The submission** — best checkpoint, all code, reproduction guide, and the three docs below. |
| `deliverables/SUMMARY.html` | Full write-up: plots, every improvement (what & why), wandb logging, future work, references, appendix. |
| `deliverables/RUNLOG.md` | Per-experiment lab log (hypothesis → change → result → conclusion). |
| `deliverables/NOTES.md` | Design decisions, ideas, and what was learned from prior art. |
| `submission/` | Working directory where experiments were developed (source of truth for the code). |
| `experiments/` | Per-run archives: exact code + config + checkpoint + `run_meta.json` for each experiment. |
| `baseline_results/` | Reproduced starter baseline. |
| `llm_handout/` | Provided assignment starter code and data. |
| `nanochat/`, `MeowLLM/`, `SiliconLLM/` | Reference repos studied for implementation techniques (Muon, RoPE, SSM, tying, SwiGLU). |

## Constraints honored

Pure PyTorch/numpy/stdlib (byte-BPE tokenizer, Muon, and SSM all reimplemented from scratch —
no tiktoken/rustbpe/flash-attn/custom kernels); lossless tokenizer; ≤2M params; ≤2000 steps; CPU.

## Environment

conda env `plivo_env` (`torch 2.13.0+cpu`, `numpy 2.0.2`). Any CPU PyTorch install works.
