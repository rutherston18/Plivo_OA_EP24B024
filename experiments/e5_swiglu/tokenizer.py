"""Byte-level BPE tokenizer — pure Python / stdlib (no tiktoken, no rustbpe).

Design (GPT-2-lite, guaranteed lossless):
  * Base vocab = the 256 raw bytes, so ANY UTF-8 text encodes (byte fallback).
  * Pre-tokenize with a trivial, fully-reversible split (whitespace runs vs
    non-whitespace runs) so merges never cross word boundaries but no bytes are
    ever dropped -> decode(encode(text)) == text exactly.
  * BPE merges are learned ONLY from train_corpus.txt and saved to bpe.json,
    which load() resolves relative to __file__ (works with cwd = submission dir).

Interface kept for train.py / evaluate.py:
  load() -> tokenizer with .encode(str)->list[int], .decode(list[int])->str, .vocab_size
"""
import json
import os
import re
from collections import Counter

# Whitespace-run OR non-whitespace-run. Every character matches exactly one,
# and the pieces concatenate back to the original string (lossless split).
_SPLIT = re.compile(r"\s+|\S+")

_BPE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe.json")


class BPETokenizer:
    def __init__(self, merges):
        # merges: list of [a, b] token-id pairs, in learned order (rank = index)
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        # vocab: id -> bytes
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self.vocab_size = 256 + len(self.merges)
        self._cache = {}

    def _encode_chunk(self, piece_bytes):
        ids = list(piece_bytes)
        if len(ids) < 2:
            return ids
        while True:
            # find the adjacent pair with the lowest merge rank
            best_rank, best_i = None, None
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            new_id = 256 + best_rank
            ids[best_i:best_i + 2] = [new_id]
        return ids

    def encode(self, text):
        out = []
        for piece in _SPLIT.findall(text):
            cached = self._cache.get(piece)
            if cached is None:
                cached = self._encode_chunk(piece.encode("utf-8"))
                self._cache[piece] = cached
            out.extend(cached)
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path=_BPE_FILE):
        with open(path, "w") as f:
            json.dump({"merges": [list(m) for m in self.merges]}, f)


# -----------------------------------------------------------------------------
# Training (run as: python tokenizer.py --data ../llm_handout/data/train_corpus.txt --vocab_size 1024)

def train_bpe(text, vocab_size):
    """Efficient incremental-count byte-level BPE over unique pre-tokenized words."""
    assert vocab_size >= 256
    num_merges = vocab_size - 256

    word_freq = Counter(_SPLIT.findall(text))
    # words[i] = [list_of_ids, count]; only keep words with >=2 symbols for merging
    words = [[list(w.encode("utf-8")), c] for w, c in word_freq.items()]

    pair_counts = Counter()
    pair_to_words = {}
    for wi, (ids, c) in enumerate(words):
        for a, b in zip(ids, ids[1:]):
            pair_counts[(a, b)] += c
            pair_to_words.setdefault((a, b), set()).add(wi)

    merges = []
    for m in range(num_merges):
        if not pair_counts:
            break
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        if pair_counts[best] <= 0:
            break
        new_id = 256 + m
        merges.append(list(best))
        a, b = best
        affected = list(pair_to_words.get(best, ()))
        for wi in affected:
            ids, c = words[wi]
            # remove this word's old adjacent-pair contributions
            for x, y in zip(ids, ids[1:]):
                pair_counts[(x, y)] -= c
                s = pair_to_words.get((x, y))
                if s is not None:
                    s.discard(wi)
            # rebuild ids merging every occurrence of (a, b)
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            words[wi][0] = new_ids
            # add the new adjacent-pair contributions
            for x, y in zip(new_ids, new_ids[1:]):
                pair_counts[(x, y)] += c
                pair_to_words.setdefault((x, y), set()).add(wi)
        # clean up the merged pair
        pair_counts.pop(best, None)
        pair_to_words.pop(best, None)
        if (m + 1) % 100 == 0:
            print(f"  merge {m + 1}/{num_merges}  (last pair count {pair_counts.get(best, 0)})")

    return merges


def load(path=None):
    """Return the tokenizer used by train.py / evaluate.py. Falls back to a raw
    byte tokenizer (vocab 256) if no trained bpe.json is found, so it never fails."""
    path = path or _BPE_FILE
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return BPETokenizer(data["merges"])
    return BPETokenizer([])  # byte-level fallback, still lossless


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab_size", type=int, default=1024)
    ap.add_argument("--out", default=_BPE_FILE)
    args = ap.parse_args()
    text = open(args.data, encoding="utf-8").read()
    print(f"training BPE: {len(text.encode('utf-8')):,} bytes -> vocab {args.vocab_size}")
    merges = train_bpe(text, args.vocab_size)
    BPETokenizer(merges).save(args.out)
    print(f"saved {len(merges)} merges to {args.out}")
