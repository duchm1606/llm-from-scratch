# llm-from-scratch

Re-implementing every component of an LLM in C, following the book *Build a Large Language Model (From Scratch)* by Sebastian Raschka.

## Structure

```
.
├── src/          # C source code
├── include/      # Header files
├── cli/          # Command-line programs built on the library (chat)
├── tools/        # Development tools (weight exporter, Unicode table generator)
├── docs/         # Notes, summaries, and deep-dives per chapter
└── tests/        # Unit tests
```

## Chapters

| # | Topic |
|---|-------|
| 1 | Understanding Large Language Models |
| 2 | Working with Text Data |
| 3 | Coding Attention Mechanisms |
| 4 | Implementing a GPT Model |
| 5 | Pretraining on Unlabeled Data |
| 6 | Fine-tuning for Classification |
| 7 | Fine-tuning to Follow Instructions |

## Build

```sh
make          # build/libllm.a
make test     # unit tests; downloads the tokenizer files and the book's text into data/ first
```

## Chat with GPT-2

`build/chat` runs GPT-2 124M on the C model in this repo. The weights are those of
[vicgalle/gpt2-alpaca](https://huggingface.co/vicgalle/gpt2-alpaca), GPT-2 fine-tuned on the
Alpaca instruction dataset, so it answers messages instead of only continuing text.

The weights are not in the repository: you download and export them yourself. Everything
downloaded or exported lands in `data/`, which is gitignored, as are `*.safetensors` and
`*.bin` files anywhere else.

### 1. Download and export the weights

```sh
make data       # GPT-2 tokenizer: data/gpt2/encoder.json and data/gpt2/vocab.bpe
make weights    # data/gpt2-alpaca/model.safetensors (~510 MB) -> data/gpt2_alpaca.bin (~622 MB)
```

`make weights` does two steps, which you can also run by hand:

```sh
# Download the checkpoint from Hugging Face.
mkdir -p data/gpt2-alpaca
curl -fL -o data/gpt2-alpaca/model.safetensors \
    https://huggingface.co/vicgalle/gpt2-alpaca/resolve/main/model.safetensors

# Export it to the layout of the C model.
make build/export_weights
build/export_weights data/gpt2-alpaca/model.safetensors data/gpt2_alpaca.bin
```

The exporter ([tools/export_weights.c](tools/export_weights.c)) needs no Python. It reads the
safetensors file and writes the parameters as flat float32s in the order `gpt_bind` expects
([include/gpt.h](include/gpt.h)), with the same mapping as the book's `load_weights_into_gpt`
(section 5.5):

- GPT-2's `Conv1D` layers store weights as `[in x out]`, so they are transposed to
  `nn.Linear`'s `[out x in]`.
- `c_attn` is split into the query, key and value projections, biases included (`qkv_bias`).
- The output head is tied to the token embeddings, so `wte` is written again as `out_head`.

Once `data/gpt2_alpaca.bin` exists, you can delete `model.safetensors`; `make weights` will
not download it again.

### 2. Run the chat

```sh
make chat
./build/chat
```

Run it from the repository root, where it finds `data/`. A session with greedy decoding
(`./build/chat -t 0`, so these replies are reproducible) looks like this:

```
GPT-2 124M fine-tuned on Alpaca instructions, seed 1791477624.
Type a message. /reset starts a new conversation, /quit or Ctrl-D exits.

> What is the capital of France?
The capital of France is Paris.

> And what about Germany?
The capital of Germany is Berlin.
```

| Option | Default | Meaning |
|--------|---------|---------|
| `-m PATH` | `data/gpt2_alpaca.bin` | exported weights |
| `-t TEMP` | `0.7` | temperature; `0` is greedy decoding |
| `-k K` | `40` | sample only from the `K` most likely tokens; `0` for all of them |
| `-n N` | `256` | most tokens per reply |
| `-s SEED` | the time | random seed, for reproducible replies |

How it works:

- Each message is wrapped in the Alpaca prompt format the model was trained on, with its three
  added tokens (`### Instruction:`, `### Response:\n`, `### End`).
- The reply is sampled with temperature scaling and top-k (section 5.3). It ends when the model
  emits `### End`.
- Earlier turns stay in the context. A KV cache ([include/kv_cache.h](include/kv_cache.h))
  keeps their keys and values, so each new token costs one pass of a single row instead of
  the whole context. On one core of an Apple M4 that gives about 19 tokens/s.
- When the 1024-token context cannot fit another reply, the chat starts a new conversation.

It is a 124M-parameter model: expect fluent but often wrong answers. With the default sampling,
the same two questions can come back with Munich as the capital of Germany.

## Reference

- Book: *Build a Large Language Model (From Scratch)* — Sebastian Raschka
