# Porting nanochat ideas into the 2,000-step speedrun

Source: Karpathy's `nanochat` (in `./nanochat`). Target: beat baseline **dev bpb
2.3718** under the hard caps (CPU-only, ≤2000 steps, ≤2M params, only
`train_corpus.txt`, pure PyTorch/numpy/stdlib, no tiktoken/rustbpe/flash-attn/
custom kernels, `evaluate.py` interface + lossless tokenizer must hold).

## What nanochat does, and whether it's portable under the PS

| nanochat feature | portable? | why / constraint | expected value |
|---|---|---|---|
| Trapezoidal LR schedule (warmup→const→linear warmdown) | ✅ pure Python | baseline uses constant LR | **high** |
| AdamW (decoupled WD) + grad clip | ✅ | baseline is plain Adam, no clip/WD | high |
| Per-param-group LRs (embd high, unembd low, matrix, scalar) | ✅ | just optimizer groups | high |
| **Muon optimizer** (Newton–Schulz / Polar-Express orthogonalization) | ✅ pure PyTorch (matmuls only) | drop distributed/`torch.compile`/fp8 wrappers; keep AdamW for embds/scalars | **very high** for fixed step budget |
| RoPE (rotary pos), drop learned pos_emb | ✅ | pure; frees `block_size*n_embd` params, better length gen | high |
| QK norm | ✅ | `F.rms_norm` on q,k; stabilizes attention | medium-high |
| RMSNorm, no learnable params | ✅ `F.rms_norm` | replaces LayerNorm | medium |
| No bias in linear layers | ✅ | fewer params, common | medium |
| ReLU² MLP activation | ✅ | swap GELU | low-medium |
| Norm after token embedding | ✅ | cheap | low-medium |
| Zero-init attn/mlp output projections; lm_head std 0.001 | ✅ | better-behaved residual at init | medium-high |
| Logit softcap (`15*tanh(logit/15)`) | ✅ | stabilizes early training | low-medium |
| Per-layer resid_lambdas / x0_lambdas (modded-nanogpt) | ✅ | few params, learnable residual mixing | medium |
| Smear (mix prev-token embd), backout | ✅ | cheap bigram-ish info, tiny params | low-medium |
| **BPE tokenizer trained on corpus** | ⚠️ must reimplement in pure Python (no rustbpe/tiktoken; `re` has no `\p{}`) | PS explicitly flags byte tokenizer = 3 tokens/Devanagari char; bpb is per-byte so this is high-leverage | **very high** but most work/risk |
| Weight tying (embd↔lm_head) | ✅ | matters more with large BPE vocab | situational |
| Value embeddings (ResFormer) | ✅ but adds embedding params per layer | watch 2M param cap | medium (later) |
| GQA | ✅ | tiny model, minor benefit | low |
| Sliding-window attention | ❌ | needs FA3; SDPA has no SWA and block is small | skip |
| FA3 / fp8 / torch.compile fused kernels / bf16 | ❌/skip | FA3+fp8 disallowed; bf16 slow on CPU; compile marginal & risky | skip |

## Experiment order (one lever at a time, each logged in RUNLOG.md)
1. **E1 – Training recipe**: trapezoidal LR (warmup 100 / linear warmdown ~40%), Adam→AdamW+WD, grad clip 1.0, raise peak LR. Architecture untouched → isolates the schedule/optimizer effect.
2. **E2 – Modern architecture**: RoPE (drop learned pos), RMSNorm (no params), QK-norm, ReLU², no-bias, zero-init projections, logit softcap. Reallocate freed params into width/depth under the 2M cap.
3. **E3 – Muon**: pure-PyTorch single-device Muon for 2-D matrices + AdamW for embeddings/scalars, with nanochat's per-group LRs and momentum warmup. Expected biggest single win at fixed steps.
4. **E4 – BPE tokenizer**: pure-Python lossless byte-level BPE trained only on the corpus (byte fallback), tuned vocab (~1–4k). Directly lowers per-byte bpb, especially on Hindi.
5. **E5 – Extras**: resid/x0 lambdas, smear, value embeddings — only if they fit the param cap and help dev bpb.

## Guardrails checked every run
params ≤ 2,000,000 · steps ≤ 2000 · tokenizer round-trip lossless · `python evaluate.py --checkpoint ckpt.pt --text_file <file>` runs unmodified.
