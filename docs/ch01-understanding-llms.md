# Chapter 1: Understanding Large Language Models

LLMs are neural networks trained on massive text corpora. The core training task is **next-word prediction** — simple, yet sufficient for models to develop emergent capabilities like translation, summarization, and code generation without being explicitly trained for them.

## Architecture

Built on the **Transformer** (Vaswani et al., 2017), which uses **self-attention** to let the model attend to the entire input sequence when producing each output token.

- **Encoder-only** (BERT): masked word prediction → strong at classification tasks.
- **Decoder-only** (GPT): next-word prediction → strong at generation. This is what the book implements.

GPT is **autoregressive**: each new token is conditioned on all previous tokens.

## Two-stage Training

1. **Pretraining** — self-supervised on a large raw text corpus → produces a base/foundation model.
2. **Fine-tuning** — on a smaller labeled dataset → instruction following or classification.

## Book Roadmap

| Stage | Content |
|-------|---------|
| 1 | Data preprocessing + attention mechanism |
| 2 | Pretrain a GPT model from scratch |
| 3 | Fine-tune: instruction following + classification |
