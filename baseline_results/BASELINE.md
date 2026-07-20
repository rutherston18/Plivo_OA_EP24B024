# Baseline reproduction

Unmodified starter code run exactly as documented, to establish the reference
score everything else will be measured against.

## Environment
- conda env `plivo_env`: Python 3.12.13, torch 2.13.0+cpu, numpy 2.0.2 (CPU only)

## Commands
Run from inside this folder:

```bash
python train.py --data ../llm_handout/data/train_corpus.txt --steps 2000 --out ckpt.pt
python evaluate.py --checkpoint ckpt.pt --text_file ../llm_handout/data/dev_eval.txt
```

## Configuration (starter defaults)
- Tokenizer: byte-level, vocab 256 (Devanagari = 3 tokens/char)
- Model: 4 layers, 4 heads, n_embd 160, block_size 128, no weight tying, dropout 0.0
- Params: 1,339,840 (cap 2,000,000)
- Optimizer: Adam, constant lr 3e-4, no warmup / schedule / weight decay / grad clip
- Batch 8, seed 1337, 2000 steps

## Result (the number to beat)
| metric | value |
|---|---|
| **dev bpb** | **2.3718** |
| final train loss (last 100-step avg) | 1.7315 |
| n_params | 1,339,840 |
| steps | 2000 |
| tokens_in_eval / scored | 159,225 / 159,224 |
| wall time | ~270 s train, ~26 s eval |

## Artifacts
- `ckpt.pt` — trained checkpoint (records step count)
- `train_log.txt` — full training loss curve
- `dev_bpb.json` — official scorer output
- `model.py`, `train.py`, `tokenizer.py`, `evaluate.py` — exact starter snapshot used

## What's questionable in the baseline (targets for improvement)
1. Byte tokenizer triples Hindi sequence length → wastes context & compute on Devanagari.
2. Constant LR, no warmup, no cosine decay — under-trained in 2000 steps.
3. Adam without weight decay; no gradient clipping.
4. No weight tying — embedding + head duplicate ~40K params that could buy capacity.
5. block_size 128 may be short; init std 0.05 fixed for all layers is not principled.
