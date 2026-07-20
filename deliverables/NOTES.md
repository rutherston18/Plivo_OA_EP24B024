# NOTES — design decisions, ideas, and what I took from prior art

This is the running "lab notebook" behind the project: the ideas I chose to pursue, *why*
I expected them to help under this problem's specific constraints, and what I learned by
studying a few open-source repos for their technical implementations. The strategy and the
choice of what to build (and, just as importantly, what *not* to build) are mine; where I
reused an implementation pattern from a repo I say so explicitly.

---

## 1. How I framed the problem

The scoring metric is **bits-per-byte (bpb)** on held-out text, and everything is capped:
**≤ 2000 optimizer steps, ≤ 2,000,000 parameters, CPU-only, pure PyTorch/numpy/stdlib,
only `train_corpus.txt` for data, a lossless tokenizer.**

Reading those constraints, I decided the game is really three sub-games, in priority order:

1. **Sample efficiency** — with only 2000 steps, *how fast the model learns per step*
   dominates. This points at the optimizer and the LR schedule first, architecture second.
2. **Bytes-per-token** — because the score is per *byte*, a tokenizer that packs more bytes
   into each token lowers bpb almost mechanically and simultaneously lengthens the effective
   context. The corpus is mixed English + Hindi, and Devanagari costs 3–4 UTF-8 bytes/char,
   so a byte tokenizer is very wasteful here.
3. **Parameter efficiency** — 2M is small, so every parameter should buy as much modeling
   power as possible; spend the budget on representation, not just raw width.

That ranking is what produced the experiment order below, and it turned out to be right:
the biggest wins came from the optimizer (E3) and the tokenizer (E4), exactly as predicted.

---

## 2. The ideas, and why I expected each to help

**E1 — Train harder (trapezoidal LR + AdamW + grad clip).** The starter used a constant
LR and plain Adam. With a fixed, tiny step budget you want to (a) warm up so you can use a
high peak LR safely, (b) hold it, then (c) decay to ~0 at the end, where loss drops
fastest. Decoupled weight decay on the 2-D weights and gradient clipping keep that
aggressive schedule stable. This is pure optimization — no new parameters — so it was the
obvious first move.

**E2 — Modernize the block (RoPE, RMSNorm, QK-norm, ReLU², no biases, zero-init
projections, logit softcap).** The starter block was a dated GPT block. Each of these is a
small, well-established upgrade that is either free on parameters or *saves* parameters
(dropping the learned position table and all biases). Zero-initializing the residual
projections makes every block an identity map at init, so training starts from a clean
`≈ ln(V)` loss instead of fighting noise.

**E3 — Muon optimizer.** This is the one I expected to matter most at a 2000-step budget.
Muon orthogonalizes each 2-D gradient update (via a short Newton–Schulz iteration), which
empirically converges faster *per step* than Adam. Embeddings and the LM head stay on
AdamW — Muon is only for the hidden matmul weights. It was the first thing to push me under
2.0 bpb.

**E4 — Byte-level BPE-1024 + weight tying.** The single most "mechanical" win. A lossless
byte-BPE that merges frequent byte sequences takes the dev set from 1 byte/token to ~2.33
bytes/token (≈4 for Hindi). Since bpb is per-byte, each token now covers more bytes → lower
score, and a 128-token window suddenly spans ~300 bytes of context. Tying the input and
output embeddings pays for the larger 1024-vocab head for free.

**E5 — SwiGLU + reinvest the freed params into depth.** Tying freed ~600k params under the
cap; I spent them on a deeper stack and a SwiGLU MLP (better capacity-per-parameter than a
plain 4× MLP). This is where I *learned something surprising*: it barely helped (−1%).
Doubling the parameter count bought almost nothing — clear evidence that at 2000 steps the
binding constraint is **data/steps, not capacity**.

**E6 — Hashed n-gram ("Engram") embeddings.** Given E5's lesson, I stopped adding capacity
and instead enriched the *input representation*. Cheap causal hashed bigram/trigram lookups
give the model instant local spelling/morphology memory, freeing attention and depth for
higher-level structure. It became my **best result (1.8225)** and, tellingly, beat the
deeper SwiGLU model (E5) at the *same* parameter count. That confirmed the thesis:
representation beats raw capacity here.

**E7 / E8 — future work.** A selective SSM (Mamba-style) backbone for linear-time,
cache-friendly sequence mixing on CPU (implemented, param-verified, held back only because
the pure-PyTorch scan is slow per step), and hierarchical byte patching (BLT/SpaceByte),
which I deprioritized because BPE already captures most of that benefit far more cheaply.

---

## 3. What I took from each repo (technical implementation reference)

I studied three repos primarily for *how* to implement things correctly in pure PyTorch,
then reimplemented each idea from scratch to fit the constraints (no external kernels,
single-device CPU, ≤2M params).

- **nanochat / nanoGPT (Karpathy).** My reference for the modern-transformer + optimization
  recipe. I adapted the shapes of the Muon Newton–Schulz iteration, the RoPE application,
  QK-norm, ReLU² MLP, zero-init projections, logit softcap, and the trapezoidal LR. I
  deliberately dropped everything the PS forbids or that doesn't help on CPU: distributed
  training, `torch.compile`, fp8, FlashAttention.

- **MeowLLM.** A compact modern decoder that reinforced two decisions: **weight tying is
  non-negotiable** at a real vocab size, and **SwiGLU** is the MLP to reach for. Useful as a
  clean, minimal cross-check on my block design.

- **SiliconLLM.** A deep CPU-native research repo. Its pure-PyTorch **selective SSM** block
  is the blueprint for my E7 backbone (linear-time recurrence, depthwise causal conv,
  input-dependent A/B/C). Its n-gram asset is an *offline decode-time drafter*, which
  doesn't fit a trainable model — but it's what inspired me to build the **trainable** hashed
  n-gram embeddings of E6 fresh rather than porting anything.

---

## 4. Key findings (the parts worth remembering)

- **Optimizer + tokenizer did most of the work.** Muon (E3) and BPE+tying (E4) are the two
  biggest levers; together they account for the bulk of the −23% vs baseline.
- **At 2000 steps, capacity is not the bottleneck.** E5 doubled params for ~1%; E6 kept the
  small 4-layer base and won by improving the *representation*. Spend budget on how the model
  *sees* the data, not on making it wider.
- **Everything composes.** BPE + tying + Muon + trapezoidal LR + n-gram embeddings stack
  cleanly; no single change had to be undone.
- **Reproducibility discipline pays off.** Every run snapshots its exact code + config +
  checkpoint, logs to wandb (offline), and is written up hypothesis→result→conclusion in
  `RUNLOG.md`. That made it easy to compare fairly and to trust the numbers.

Baseline dev bpb **2.3718 → 1.8225** (−23.2%) at 1.88M params, within all constraints.

---

## 5. References

The prior-art papers behind each technique are listed in the "Related work & references"
section of `SUMMARY.html` (Muon, Mamba, N-Grammer, RoPE, RMSNorm, SwiGLU, QK-norm, ReLU²/
Primer, weight tying, byte-level BPE, BLT/SpaceByte, WSD/trapezoidal LR).
