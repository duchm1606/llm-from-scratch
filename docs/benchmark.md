# Benchmark: generation with and without the KV cache

How fast does the C model generate text, and what does the KV cache
(`src/kv_cache.c`) buy over the book's `generate_text_simple` (section 4.7)?
This page describes the experiment, how it measures, and the results on one
machine. `tools/bench.c` is the benchmark; [benchmark-m4.csv](benchmark-m4.csv)
holds every timed run behind the tables below, and
[benchmark-llamacpp-m4.csv](benchmark-llamacpp-m4.csv) those of the
comparison with llama.cpp.

## Questions

1. How long does one generated token take at context length n, with the
   cache and without it, and how does the gap grow with n?
2. How long until the first token of an n-token prompt (prefill)?
3. How much memory does generation need?
4. How far is this from an inference engine in production use, llama.cpp,
   on the same weights and the same machine?

## Setup

| | |
|---|---|
| Machine | MacBook Air M4 (Mac16,13), 4 performance + 6 efficiency cores, 32 GB, fanless |
| OS | macOS 26.5.2 (Darwin 25.5.0, arm64) |
| Power | AC adapter, Low Power Mode off |
| Compiler | Apple clang 17.0.0 (clang-1700.4.4.1) |
| Flags | `-std=c11 -O2 -Wall -Wextra -Wpedantic`, the Makefile's: no `-march=native`, no `-ffast-math` |
| Threads | 1: the library has no threading |
| Model | GPT-2 124M fine-tuned on Alpaca ([vicgalle/gpt2-alpaca](https://huggingface.co/vicgalle/gpt2-alpaca)), float32, 50,260-token vocabulary, as `make weights` exports it |
| Input | the first n tokens of `data/the-verdict.txt`, the book's short story |

## Method

### What each phase times

Every phase calls the library exactly as a program generating text would.
"Context n" means the model produces the token that follows n tokens.

| Phase | One timed run | Stands for |
|---|---|---|
| `load` | `fread` of the whole weights file into memory | starting the chat |
| `prefill` | `gpt_forward_cached` on n tokens into an empty cache | time to the first token of an n-token prompt |
| `decode` | `gpt_forward_cached` on token n, with tokens 1..n-1 cached | one generated token with the cache, at context n |
| `nocache` | `generate_text_simple` for one token after n tokens | one generated token without the cache, at context n |
| `generate_cached` | prefill of a 64-token prompt, then 32 greedy tokens with the cache | a short reply, end to end |
| `generate_nocache` | `generate_text_simple` for the same 32 tokens | the same reply without the cache |
| `drift` | `decode` at n = 32 again, after everything else | throttling check |

n sweeps 32, 64, 128, 256, 512 and 1024, the model's whole context.

Without the cache, the benchmark times one step at each n instead of
generating all the way to 1024 tokens. Each step costs more than the last, so
a full run would take hours, while the step at n is all the curve needs. A
decode run puts back the cache's length to n - 1 before each call, so every
run times the step at the same position.

### Repetitions and statistics

- Each measurement starts with untimed warm-up runs, which bring the weights
  into the caches and the core up to speed: one for `load`, `prefill` and
  `nocache`, three for `decode`.
- Then it times 5 runs (20 for `decode`, whose runs are short) and reports the
  **median**, with the minimum and maximum as the spread. The two end-to-end
  runs are timed once.
- The clock is `clock_gettime(CLOCK_MONOTONIC)` around the single library call,
  so tokenizing and printing stay out of every number.

### Checks that the numbers come from a correct model

The benchmark fails rather than report timings of a wrong result:

- At every n, the greedy token after the context must be the same with the
  cache (`decode`) and without it (`nocache`).
- The 32 greedy tokens of `generate_cached` and `generate_nocache` must match.
- The `prefill` and `decode` logits at every n must be identical, bit for bit:
  they compute the same last row through two code paths.

`tests/test_kv_cache.c` additionally checks the cached logits against
`gpt_forward`, which `tests/test_gpt.c` checks against PyTorch.

### Throttling

The MacBook Air has no fan, so a long run at full load could slow down. The
`drift` phase repeats `decode` at n = 32 after all the other phases, at the
end of the run, and the two medians are compared. Every timed run is also in the
CSV, so a slowdown within a measurement shows there too.

### Memory

The benchmark reads the process's peak resident memory (`getrusage`) after
loading, after the cached phases (the chat's footprint at a full context: the
weights, a full KV cache and its scratch), and at the end, when
`generate_text_simple`'s scratch is added. `/usr/bin/time -l` gives the
whole process's peak as a cross-check. The expected sizes come from the
library's own size functions.

### Not measured

- **Cold load.** The weights file sits in the OS page cache after the first
  read, so `load` times a warm read. Timing a read from disk takes `sudo purge`
  before each run.
- **Several threads, other flags.** The library is single-threaded, and
  `-O3 -march=native` or `-ffast-math` would change the numbers; these results
  hold for the Makefile's flags only.
- **Sampling.** Greedy decoding is timed. Temperature and top-k add one pass
  over the 50,260 logits per token, small next to the model.

## Results

One run on 2026-10-10, 1,064 s in all, of the library at commit 7909416 (the
CSV says `7909416-dirty`: only the benchmark itself was not committed yet).
Every correctness check passed. Apart from `load`, every timed run lies within
3.2% of its median, except one prefill run at n = 256 (+5.1%).

### Time per token

| Context n | Prefill: first token after n | Decode: next token, with cache | Next token, without cache | Speedup of the cache | Speedup predicted by the work |
|--:|--:|--:|--:|--:|--:|
| 32 | 1.20 s | 53.6 ms (18.7 tok/s) | 1.23 s | 23× | 22× |
| 64 | 2.45 s | 54.8 ms (18.3 tok/s) | 2.46 s | 45× | 44× |
| 128 | 4.96 s | 55.0 ms (18.2 tok/s) | 4.91 s | 89× | 88× |
| 256 | 9.86 s | 56.0 ms (17.9 tok/s) | 9.85 s | 176× | 175× |
| 512 | 20.0 s | 58.4 ms (17.1 tok/s) | 19.9 s | 341× | 346× |
| 1024 | 41.5 s | 64.3 ms (15.6 tok/s) | 40.2 s | 625× | 679× |

Medians. "Predicted by the work" is the ratio of multiply-adds, derived under
[Analysis](#analysis).

- **With the cache, a token costs about the same at any context:** 53.6 ms at
  n = 32, 64.3 ms at n = 1024 (+20%).
- **Without it, a token costs the whole context:** about 39 ms per context
  token, 40 s per token at n = 1024. Each step costs the same as a prefill of
  the whole context (the two columns agree within 3.2%), since that is what
  `generate_text_simple` does.
- **The speedup grows linearly with n**, from 23× to 625×.

### End to end

32 greedy tokens after a 64-token prompt take **4.16 s with the cache** (a
2.45 s prefill and 31 decode steps) and **96.5 s without** (23×). The sweep
predicts both: 2.45 s + 31 × 54.8 ms = 4.15 s, and the sum of the uncached
steps from n = 64 to 95 is about 98 s.

### Memory

| Point in the run | Expected (library sizes) | Peak resident memory |
|---|--:|--:|
| Weights loaded | 622.0 MiB | 628.4 MiB |
| + full KV cache (72.0) and its scratch (15.0): the chat at a full context | 709.0 MiB | 715.5 MiB |
| + `generate_text_simple`'s scratch (28.2) | 737.2 MiB | 743.7 MiB |

`/usr/bin/time -l` agrees: 779,796,480 bytes = 743.7 MiB. The 6.5 MiB above
the library's buffers is the rest of the process: the program, the token ids
and the tokenizer while it was loaded.

### Load and throttling

- **Load:** 49 ms (48–65 ms) for the 622 MiB file, 13 GB/s: a copy out of the
  page cache, not a read from the SSD.
- **Throttling:** none. Decode at n = 32 took 53.60 ms at the start and
  53.65 ms after 17 minutes of full load on one core (+0.1%).

## Analysis

### Why the speedup grows linearly

Counting multiply-adds (MACs) for GPT-2 124M:

- **Blocks**, per token: 84.9M, each of the 12 blocks having four 768×768
  attention projections and the 768×3072 and 3072×768 feed forward layers.
- **Output head**, per generated token: 38.6M (768 × 50,260), on the last row
  only.
- **Attention** at context s: 18,432 · s (scores and weighted sum, 768 · s
  each, in 12 layers).

| | Work for the token after n |
|---|---|
| With the cache | 84.9M + 38.6M + 18,432 · n: about 124M, nearly flat in n |
| Without it | 84.9M · n + 38.6M + 18,432 · n(n+1)/2: the whole context through the blocks again |

Both paths run at the same rate, 4.4–4.8 GFLOP/s, so time is work over that
rate and the speedup is the ratio of the work: 22× at n = 32 up to 679× at
n = 1024. The measured speedups match within 3.1% up to n = 512. At n = 1024,
the measured 625× is 8% lower: decode slows to 4.4 GFLOP/s, its attention
reading 72 MiB of cached keys and values, while the uncached step speeds up
to 4.8.

Decode slows by 20% from n = 32 to 1024 for the same reason: attention over
the cache grows to 18.9M MACs (+15% work), plus the reads of the cache.

### How far from the hardware

- **Compute:** 4.4–4.8 GFLOP/s, on a core averaging 4.12 GHz over the run
  (4.38 × 10¹² cycles in 1,061 s, from `/usr/bin/time -l`): about 1.1
  floating-point operations per cycle.
- **Memory:** decode reads 494 MB of weights per token, 7.7–9.2 GB/s, 7% of
  the M4's 120 GB/s. Even that bandwidth alone would allow about 240 float32
  tokens/s.

Neither limits it. The time goes into `matmul_bt`'s dot product,
`dot += a_i[p] * b_j[p]`. Without `-ffast-math` the compiler must add in
source order: the assembly vectorizes the multiplies (`fmul.4s`) but sums the
products with 16 scalar `fadd`s into one register per iteration, each waiting
for the one before. The loop runs at the latency of a float add.

This is also why a prefill token (37–41 ms) is barely cheaper than a decode
step (54 ms), in the ratio of their work (84.9M against 124M MACs).
`matmul_bt` walks the whole weight matrix again for each row, so a batch of
rows gains nothing. An optimized matrix multiply reuses each weight across
rows, and there prefill is far faster per token than decode.

These are the directions for speed, all outside the book: several
accumulators in the dot product (changing the rounding, so the tests'
tolerances against PyTorch must still hold), blocking the matrix multiply for
reuse across rows, and threads.

## Comparison with llama.cpp

llama.cpp runs GPT-2 among many other models, on the CPU with ggml's own
kernels. Running it on the same weights answers two questions: does it
compute the same model, and how much faster does an optimized engine run it?

### Setup

| | |
|---|---|
| Version | release b11539, commit 10a60cf |
| Build | `cmake -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DGGML_BLAS=OFF`: the CPU backend alone, without the GPU or Accelerate, so the thread count bounds all the compute. Defaults otherwise: `-O3 -mcpu=native+dotprod+i8mm+nosve+sme`, ggml's thread pool (Apple clang has no OpenMP) |
| Weights | the same safetensors, converted by llama.cpp's `convert_hf_to_gguf.py --outtype f32`: 499.5 MB, the output head tied to the token embeddings instead of copied. Q8_0 (136.7 MB) from `llama-quantize` |
| Configurations | **default**: f16 KV cache, flash attention auto. **exact**: `-ctk f32 -ctv f32 -fa off`, float32 like ours. **Q8_0**: 8-bit weights, defaults otherwise |

Two one-line fixes to llama.cpp, in the checkout used here only:

- The converter of this release fails on the checkpoint's `attn.bias`
  tensors, the causal-mask buffers older versions of transformers saved; its
  own comment says to skip them, and now it does.
- `llama-bench` does not take `f32` for `-ctk` and `-ctv`, although the
  library does; its parser now maps the name.

### Same results

A C program against `llama.h` feeds llama.cpp the same 64 token ids as
`generate_cached`, decodes 32 greedy tokens and compares each step's logits
with ours, which reach 152 in magnitude:

| llama.cpp | Greedy tokens equal to ours | Largest logit difference |
|---|--:|--:|
| F32, default | 32 of 32 | 0.087 |
| F32, exact | 32 of 32 | 0.054 |
| F32, exact, rebuilt without `GGML_GELU_FP16` | 32 of 32 | 0.00041 |
| Q8_0, default | 7 of 32 | 1.7, over the first 7 |

By default, three of ggml's shortcuts work in half precision: the GELU
reads a table indexed by its input rounded to float16, the KV cache holds
float16, and flash attention computes in float16. Without all three, the
largest difference falls to 4 × 10⁻⁴, 3 × 10⁻⁶ of the logits' size: float32
rounding in a different order of summation. The two programs compute the
same model. 8-bit weights change it enough to pick another token at the
eighth step.

### Method

- `llama-bench` times prefill (`-p n`, n tokens into an empty context) and
  decode (`-n 16 -d d`: 16 tokens one at a time after d tokens, so positions
  d + 1 to d + 16). Decode at depth d stands next to our `decode` at n = d,
  and depth 1008 next to n = 1024, as d + 16 must fit in the context.
- Each test runs a warm-up and 5 repetitions and reports their mean. The
  whole sweep ran three times, interleaved, and the tables give the **median
  of the three rounds**.
- A llama.cpp token takes a few milliseconds, so a burst of background work
  shows: two earlier sweeps, run just after the build, were erratic and up to
  twice as slow, and are not used. At one thread the three rounds agree
  within 3.2%, except the default configuration's prefill of 32 and 64 tokens
  (11% and 6%). At four threads, sharing the four performance cores with the
  desktop, they spread by up to 24%.

### Speed, one thread

Tokens per second; the factor is over ours.

| Context | Ours: decode | llama.cpp F32 exact | llama.cpp F32 default | llama.cpp Q8_0 |
|--:|--:|--:|--:|--:|
| 32 | 18.7 | 122 (6.5×) | 132 (7.1×) | 470 (25×) |
| 64 | 18.3 | 122 (6.7×) | 131 (7.2×) | 453 (25×) |
| 128 | 18.2 | 121 (6.7×) | 128 (7.0×) | 424 (23×) |
| 256 | 17.9 | 114 (6.4×) | 123 (6.9×) | 372 (21×) |
| 512 | 17.1 | 107 (6.3×) | 114 (6.7×) | 301 (18×) |
| 1024 (llama.cpp 1008) | 15.6 | 102 (6.6×) | 98 (6.3×) | 214 (14×) |

| Prompt | Ours: prefill | llama.cpp F32 exact | llama.cpp F32 default | llama.cpp Q8_0 |
|--:|--:|--:|--:|--:|
| 32 | 26.7 | 223 (8.4×) | 222 (8.3×) | 1,565 (59×) |
| 64 | 26.1 | 226 (8.7×) | 236 (9.0×) | 1,617 (62×) |
| 128 | 25.8 | 229 (8.9×) | 242 (9.4×) | 1,596 (62×) |
| 256 | 26.0 | 230 (8.9×) | 242 (9.3×) | 1,534 (59×) |
| 512 | 25.5 | 216 (8.5×) | 238 (9.3×) | 1,407 (55×) |
| 1024 | 24.7 | 203 (8.2×) | 232 (9.4×) | 1,208 (49×) |

On the same float32 weights and one core, llama.cpp decodes 6–7× faster and
prefills 8–9× faster.

### Speed, four threads

| Tokens per second | 1 thread | 4 threads |
|---|--:|--:|
| Decode, F32 default, context 32 | 132 | 142 |
| Decode, F32 default, context 1008 | 98 | 109 |
| Decode, Q8_0, context 32 | 470 | 458 |
| Prefill, F32 default, 128 tokens | 242 | 744 |
| Prefill, Q8_0, 128 tokens | 1,596 | 3,784 |

### What the gap is made of

- **llama.cpp's decode is bound by memory.** At 132 tokens/s it streams the
  497.8 MB of float32 weights at 66 GB/s, about what one core draws. The
  Q8_0 weights are 3.7× smaller (134.9 MB) and decode 3.6× faster, at the
  same 63 GB/s. Ours streams at 9 GB/s, held back by the chain of float adds
  in `matmul_bt` (see [How far from the hardware](#how-far-from-the-hardware)):
  that is the 6–7×.
- **Its prefill reuses the weights.** At 242 tokens/s the blocks run at
  about 41 GFLOP/s, 9× our 4.6, and a prefill token costs half a decode step
  (4.1 ms against 7.6 ms), where ours costs 0.7 of one.
- **Threads help prefill, hardly decode.** A decode step is 2–8 ms of work in
  12 layers of small operations, with the threads meeting after each, while
  the memory bandwidth one core already uses grows little. Prefill has the
  work to share: 3.1× faster on four threads.
- **Context costs it relatively more.** Its decode slows by 26% (F32) and 55%
  (Q8_0) from context 32 to 1008, against our 17%: attention over the cache
  shrank less than the matrix multiplies did.

So ours stays within an order of magnitude of llama.cpp on the same float32
weights and one core, and the difference sits in one loop. Beyond that,
llama.cpp's real speed on a CPU comes from smaller weights: 8-bit ones
decode 14–25× faster than ours.

## Reproduce

```sh
make data weights bench
./build/bench > results.csv     # about 20 minutes; summary on stderr
```

The CSV starts with comment lines naming the date, the commit, the compiler,
the flags and the machine, then one row per timed run: `phase,n,rep,seconds`.

The comparison with llama.cpp, from a checkout of release b11539 with the two
fixes above:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DGGML_BLAS=OFF
cmake --build build -j --target llama-bench llama-quantize
# a directory with the model's config.json, tokenizer files and model.safetensors
python convert_hf_to_gguf.py gpt2-alpaca --outtype f32 --outfile gpt2-alpaca-f32.gguf
build/bin/llama-quantize gpt2-alpaca-f32.gguf gpt2-alpaca-q8_0.gguf Q8_0
build/bin/llama-bench -m gpt2-alpaca-f32.gguf -t 1 -p 32,64,128,256,512,1024 -n 0 -o csv
build/bin/llama-bench -m gpt2-alpaca-f32.gguf -t 1 -p 0 -n 16 -d 32,64,128,256,512,1008 -o csv
```

Add `-ctk f32 -ctv f32 -fa off` for the exact configuration and `-t 4` for
four threads, and run the sweep three times.
