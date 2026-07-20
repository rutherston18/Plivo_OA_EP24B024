# Idea families from SiliconLLM / MeowLLM (+ the three PS directions)

Constraints unchanged: CPU-only, ≤2000 steps, ≤2M params, only `train_corpus.txt`,
pure PyTorch/numpy/stdlib, no tiktoken/rustbpe/flash-attn/custom-kernels, lossless
tokenizer + working `evaluate.py`. Current best: **E3 dev bpb 1.9516** (byte vocab).
E4 (BPE-1024) is implemented + verified lossless but not yet scored (run pending).

## What each repo gives us
- **MeowLLM** (`meow/model.py`): compact modern decoder — **tied input/output embeddings**,
  **SwiGLU** FFN, RoPE, RMSNorm, depth-scaled init. Confirms the PS advice ("weight
  tying non-negotiable", "SwiGLU", clean dims). All pure PyTorch, trivially portable.
- **SiliconLLM** (`benchmarks/phase55/phase55_ssm.py`): pure-PyTorch **Mamba-1 selective
  diagonal SSM** backbone (`ArchA`): SSM layers + one sliding-window-attention layer,
  ~1.6M params, BPE-1024, CE-only, val-bpb loop — a ready blueprint for family 3.
  Also has n-gram assets (offline C-engine drafter — not an in-model trainable module),
  dReLU sparse MLP, ternary BitLinear, MoE, and an associative "recall slot".

## The three PS families → concrete, in-budget experiments
### Family 1 — Hierarchical / dynamic patching (BLT / SpaceByte)
- Full BLT (entropy-gated dynamic patches + local encoder/decoder + latent transformer)
  is high-effort. Our BPE-1024 already delivers most of the benefit it targets
  (compressing multi-byte Devanagari into single tokens), losslessly.
- **Verdict:** stretch goal. If attempted, a simplified *static* patching (fixed byte
  groups / whitespace-delimited patches with a light encoder→core→decoder) — but expected
  ROI is below families 2/3 given BPE is in place. Deprioritized.

### Family 2 — Hash-enriched n-gram / Engram hybrid  ⭐ (cheap, tiny-model win)
- Inject cheap **hashed bigram/trigram embeddings** into the token representation (or
  attention values): hash last-2 / last-3 token ids → fixed bucket table (e.g. 2^14–2^16
  rows × small dim) → add to the stream. Gives instant local spelling/morphology memory,
  freeing attention/depth for higher-level structure — exactly what <2M-param models need.
- Pure PyTorch, few params (one/two hash tables, sized to the cap). Build fresh (repos
  only have an offline drafter). **High ROI, low-med effort.**

### Family 3 — CPU-native recurrent / linear attention (SSM / RWKV) ⭐
- Port SiliconLLM's `ArchA` **selective SSM** (Mamba-1) as a pure-PyTorch backbone sized
  to <2M params (tied embeddings), O(N) in sequence length — a hybrid of SSM layers + 1
  windowed-attention layer. On CPU the sequential scan over block=128 is acceptable.
- **Med-high ROI, med-high effort** (scan is slower per step; seq is short so OK).

## Quick wins to fold in regardless (from MeowLLM + PS)
- **Weight tying** (tok_emb ↔ head): frees 163,840 params at vocab 1024 → reinvest in depth/width.
- **SwiGLU** FFN instead of ReLU²/GELU.
- Narrower-deeper vs 4× FFN; keep dims multiples of 8/16 for CPU SIMD.

## Roadmap (each = one logged run; ASK before every full run)
- **E4** BPE-1024 + weight tying (+ reinvest freed params). [tokenizer ready]
- **E5** SwiGLU + depth/width tuning within the 2M cap.
- **E6** Family 2: hash n-gram (Engram) embeddings.
- **E7** Family 3: SSM/hybrid backbone (ArchA port).
- **E8** (stretch) Family 1: hierarchical patching.

## Infra added this session
- `train.py` now supports `--wandb` (project/mode/run_name), `--eval_every` + `--eval_bytes`
  (periodic dev-bpb charts; cheap subset periodically, full eval at end), and
  `--tag` which snapshots **all .py + bpe.json + ckpt + run_meta.json** into
  `experiments/<tag>/` so any checkpoint is reproducible / resumable.
