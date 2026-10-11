# Plan: improving the Tolstoy GPT

## Where we are (2026-10-10)

Baseline `gpt-tolstoy-all.model.pt`: 33.7M parameters (8 layers, 8 heads, 512 wide), 8K BPE vocabulary,
trained 10,000 steps on `data/tolstoy/tolstoy_all.txt` (9.8M tokens, ~18 passes over the text).
Best validation 1.473 bits/char at step 9000 (W&B run `paveld/tolstoy-gpt/mlzt0alk`).

- Validation was still improving at the end, but slowly: 1.482 -> 1.473 over the last 1,500 steps.
- Train/validation gap is widening (2.93 vs 3.66): the model is limited by how much text it has,
  more than by its size or how long it trains.

## Improvements, by expected impact

| # | Change | Why | Cost |
|---|---|---|---|
| 1 | Pre-train on a large Russian prose corpus (Turgenev, Dostoevsky, Chekhov, Goncharov, Leskov... from az.lib.ru, plus `about_tolstoy.txt`), then fine-tune on Tolstoy | How real LLMs are trained: general language from lots of text, style from a short fine-tune. Biggest gain by far | Download + hours of training |
| 2 | Training recipe: LR warmup + cosine decay, weight decay 0.1, gradient clipping, AdamW betas (0.9, 0.95) | Constant LR leaves training noisy at the end | Small change in `gpt.py` |
| 3 | Architecture: tie input/output embeddings, GELU, fused attention (`F.scaled_dot_product_attention`), scaled init | Tying removes 4.2M parameters that mostly memorize; fused attention ~2x faster | Model rewrite; `infer.py` must load both old and new checkpoints |
| 4 | Context 512 tokens instead of 256 (~900 characters now) | Keeps the model on topic over a paragraph | Needs #3 for speed |
| 5 | Data fixes: drop non-Tolstoy items from `tolstoy_all.txt` (peasants' memoirs, museum bulletins); validation from blocks spread over all works instead of the last 10% of the file (mostly letters) | Cleaner data; validation that reflects all genres | Small change |
| 6 | Sampling: top-p, repetition penalty in `infer.py` | Better-reading output without retraining | A few lines |

Not worth it now: a bigger model on the same text (memorizes faster), or training longer on the same text.

## Phases

- [x] **Phase 1** — #5 + #2 + #3 + #6, retrain on the cleaned Tolstoy text, compare with the baseline.
      Done 2026-10-10: `gpt-tolstoy-v2.model.pt` (W&B run `8wrefs8g`), 29.5M parameters, 10,000 steps in 50 minutes
      (2x faster, half the GPU memory). Best validation 1.349 bits/char at the last step, still improving.
      On text neither model trained on (16 blocks of letters and late works): **1.290 vs 1.417 bits/char**
      for the baseline, about 9% better. Samples read similarly; both still make grammar slips (gender agreement).
- [ ] **Phase 2** — #1 + #4: build a pre-training corpus of 19th-century Russian prose, pre-train with
      512-token context, fine-tune on Tolstoy.
