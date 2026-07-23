# NOTES

Best configuration: byte-level BPE tokenizer at vocab 3072 trained only on
`train_corpus.txt`, feeding a 6-layer, 6-head, 144-dim decoder-only
transformer with RoPE, RMSNorm, SwiGLU (n_ff 384) and tied input/output
embeddings — 1,937,232 parameters, trained for exactly 2000 AdamW steps at
batch 16 x block 512.

The tokenizer is the dominant win because the score is bits per *byte*: BPE
raises compression from 1.0 to 3.152 bytes/token on dev, so the same per-token
cross-entropy divides by 3.15x more bytes, and the Hindi text stops costing
three tokens per character.

Vocab 3072 is deliberately not larger — compression gains flatten after ~4k
while the embedding table grows linearly, and under a 2M cap a bigger vocab
starves the layers that do the actual computation.

Weight tying frees 442,368 parameters (22% of the budget) that the baseline
spent storing token identity twice, and those parameters were reinvested in
depth; the baseline also left a third of the parameter cap simply unused
(1,339,840 of 2,000,000).

The init rewrite (std 0.02 embeddings, 1/sqrt(fan_in) linears, residual
projections scaled by 1/sqrt(2*n_layer)) matters far more than usual here
because a 2000-step budget cannot afford the few hundred steps the baseline
wastes undoing `std=0.05` everywhere.

With steps hard-capped, the schedule is the algorithm: 150-step warmup plus
cosine decay to 5% allowed a 10x higher peak LR (3e-4 to 3e-3) than the
baseline's constant rate, with gradient clipping at 1.0 as insurance against a
single bad batch that there would be no budget to recover from.

Weight decay is applied to matrices only and never to norms or the tied
embedding, and dropout is zero on purpose, since 2000 steps over a 7 MB corpus
is roughly one epoch and the model is underfitting rather than overfitting.

RoPE replaces the learned positional table because it is parameter-free,
encodes relative position, and stays correct on the short windows that the
sliding-window scorer produces.

**Measured result: dev bpb 1.6692, against the baseline's 2.3718 — a 29.6%
reduction**, with the tokenizer change accounting for most of the gap; the
scorer's own output corroborates it, with `tokens_in_eval` falling from 159,225
to 50,511 on the identical file.

One caveat I want on the record: `train.py`'s in-training estimate read ~0.99
bpb, but the official scorer returned 1.6692 — the estimate ignores the
scorer's 50% sliding-window context carry-over and is optimistic by ~1.7x, so
1.6692 is the number I report.