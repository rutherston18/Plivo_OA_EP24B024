# RUNLOG

Reference: baseline dev bpb **2.3718** (starter code, 1,339,840 params, 2000 steps).
Scorer: `python evaluate.py --checkpoint ckpt.pt --text_file ../llm_handout/data/dev_eval.txt`.
All runs: seed 1337, batch 8, block 128, 2000 steps, CPU (conda `plivo_env`, torch 2.13.0+cpu).

---

## E1 — Training-recipe port (nanochat) on baseline architecture
**Hypothesis:** the baseline is under-trained because it uses a constant LR with no
warmup/decay and plain Adam. nanochat's trapezoidal LR + AdamW + grad clip should
extract more from the same 2000 steps without touching the architecture.

**Changed (only train.py):**
- Adam → AdamW, decoupled weight_decay 0.1 on 2-D weights only, betas (0.9, 0.95)
- constant LR 3e-4 → trapezoidal: 100-step linear warmup, constant, linear warmdown
  over last 40% to 0; peak LR raised 3e-4 → 1e-3
- gradient clipping at norm 1.0

**Result:** dev bpb **2.3718 → 2.1864** (−0.185, −7.8%). Final train loss 1.7315 → 1.5388.
Params unchanged (1,339,840). Steps 2000.

**Conclusion:** big, cheap win purely from optimization. The warmdown clearly helps —
loss drops fastest once LR decays (steps 1300→2000). Peak LR 1e-3 was stable with
clipping. Keep this recipe as the new base for architecture changes (E2).

---

## E2 — Architecture modernization (nanochat) on the E1 recipe
**Hypothesis:** the starter block (learned pos_emb, LayerNorm, GELU, biases, plain
init) is dated. nanochat's stack (RoPE, parameter-free RMSNorm, QK-norm, ReLU²,
no-bias, zero-init output projections, logit softcap) should train faster and
generalize better in the same 2000 steps — and RoPE + no-bias even frees params.

**Changed (only model.py):** RoPE (dropped learned pos table), RMSNorm (no params),
QK-norm, ReLU² MLP, bias=False everywhere, zero-init attn/mlp output projections,
logit softcap 15, init std 0.02. Training recipe identical to E1.

**Result:** dev bpb **2.1864 → 2.0421** (−0.144 vs E1; −0.330 / −13.9% vs baseline).
Params 1,339,840 → **1,310,720** (fewer, from dropping pos_emb + biases). Steps 2000.
Init loss ≈ ln(256), confirming zero-init projections give a clean starting point.

**Conclusion:** modernization helps and is "free" on params. Freed budget + headroom
under the 2M cap can later be reinvested in width/depth. Next: E3 (Muon optimizer),
the expected biggest single lever for a fixed 2000-step budget.

---

## E3 — Muon optimizer (nanochat / modded-nanogpt) for matrix params
**Hypothesis:** with only 2000 steps, per-step efficiency matters most. Muon
orthogonalizes each 2-D update via a Newton-Schulz iteration, which converges
faster per step than Adam. Put the block matrices on Muon; keep embeddings + head
on AdamW (Muon should not touch embedding/final layers).

**Changed (muon.py + train.py):** custom single-device pure-PyTorch Muon (NS-5,
momentum 0.95, lr 0.02, nesterov). Split: 1,228,800 matrix params → Muon,
81,920 (tok_emb + head) → AdamW (lr 1e-3). Architecture identical to E2.

**Result:** dev bpb **2.0421 → 1.9516** (−0.091 vs E2; −0.420 / −17.7% vs baseline).
Faster convergence visible early (step 1000 train loss 1.49 vs E2's 1.59). Params
unchanged. Cost: ~260 ms/step (vs 173) from the NS iterations — still ~8.6 min/run.

**Conclusion:** Muon is a real, cheap win at fixed step budget, as predicted. muon_lr
0.02 was stable first try; worth a small sweep later. Next: E4 (BPE tokenizer) — the
biggest remaining lever, since bpb is per-byte and the byte tokenizer wastes 3
tokens per Devanagari char.

---

## E4 — Byte-level BPE-1024 tokenizer + weight tying
**Hypothesis:** bpb is per-*byte*, but the byte tokenizer emits one token per byte, so
Devanagari costs 3–4 tokens/char and a 128-token context covers only 128 bytes. A
lossless BPE that merges common byte sequences means each predicted token covers more
bytes → directly lower bpb and much longer effective context. Tying input/output
embeddings frees the params a 1024-vocab head would otherwise cost.

**Changed (tokenizer.py + train.py + model.py):** pure-Python lossless byte-BPE, vocab
1024 (dev 2.33 bytes/token). Weight tying on (tok_emb ≡ head). Muon on block matrices
(1,228,800) + AdamW on tied embedding (163,840). Trapezoidal LR, 2000 steps. Encoded
corpus now cached to `.corpus_ids_v1024.pt` to skip the ~22s re-encode per run.

**Result:** dev bpb **1.9516 → 1.8583** (−0.093 vs E3; −0.514 / **−21.6%** vs baseline).
1,392,640 params. ~266 ms/step, 532s total. 7.32 MB corpus → 3.31 M tokens (vocab 1024).

**Conclusion:** largest single win so far and cheap. Tying + smaller token count leaves
plenty of headroom under the 2M cap to reinvest in depth/width (E5) or an SSM backbone
(E7). Infra note: wandb switched to **offline** mode + fail-safe init (a wandb timeout
can no longer abort a multi-minute run); corpus-id caching added.

---

## E5 — SwiGLU MLP + reinvest freed params (deeper stack)
**Hypothesis:** E4's weight tying freed ~600k params under the 2M cap. Spend them on
(a) more depth and (b) a SwiGLU MLP, which gives more capacity per parameter than a
plain 4× ReLU² MLP.

**Changed (model.py already had SwiGLU; via CLI):** `--mlp_act swiglu --mlp_mult 3
--n_layer 5 --n_embd 160 --tie_weights`, Muon. Params 1,392,640 → **1,827,840**
(deeper: 4→5 blocks; SwiGLU adds a 3rd matrix per MLP). 2000 steps, ~388 ms/step.

**Result:** dev bpb **1.8583 → 1.8385** (−0.020 vs E4; −0.533 / **−22.5%** vs baseline).

**Conclusion:** a small further win. Doubling params (1.39M→1.83M) buys only ~1% — the
returns from width/depth are clearly diminishing at a fixed 2000-step budget; the
binding constraint is data/steps, not capacity. Better to spend effort on the token
representation (E6 n-gram) and a more sample-efficient backbone (E7 SSM) than on more
params. Per-step cost rose (266→388 ms) for little gain, so E7 SSM stays the priority.

---

## E6 — Hashed n-gram (Engram) embeddings
**Hypothesis:** E5 showed extra params/depth barely move bpb, so capacity isn't the
bottleneck — the *input representation* is. Cheap hashed bigram/trigram lookups give the
model instant local (spelling/morphology) memory, freeing attention+depth for
higher-level structure. This is the param-efficient "family 2" idea.

**Changed (model.py NgramEmbed + train.py CLI):** causal hashed n-gram embeddings,
orders {2,3}, 1536 buckets/order, zero-init (starts as a no-op, learns to use them).
Added to the token embedding. Base is the **same 4-layer d160 ReLU² stack as E4** (so
this isolates the n-gram effect), Muon. Params 1,392,640 → **1,884,160** (+491k in two
hash tables → AdamW). Verified causal: n-gram at position t ignores tokens > t.

**Result:** dev bpb **1.8583 → 1.8225** (−0.036 vs E4; −0.549 / **−23.2%** vs baseline).
~291 ms/step, 583s. Beats E5 (1.8385) despite E5 being *deeper* (5 layers) at similar
param count.

**Conclusion:** best result so far, and the key finding: at equal params, **n-gram
memory beats depth**. The token representation is a better place to spend budget than
width/depth. n-gram tables + Muon + BPE + tying compose cleanly. Next: E7 — a selective
SSM backbone (the explicitly-requested direction), more sample-efficient per step and
cache-friendly on CPU; can layer n-gram on top later.
