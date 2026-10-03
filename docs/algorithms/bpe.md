# Byte Pair Encoding (BPE)

> **Origin:** "Neural Machine Translation of Rare Words with Subword Units" — Sennrich et al., 2016.
> Used by GPT-2, GPT-3, ChatGPT. Vocab size: **50,257 tokens**.

---

## Why BPE exists

Word-level tokenizers have a fundamental flaw: any word outside the training vocabulary causes a failure or gets collapsed to `<|unk|>`, losing all information.

BPE solves this with one insight:

> **No word is truly unknown — every string can be decomposed into smaller known pieces.**

Instead of tokenizing at word boundaries, BPE operates at the **subword** level. Common words stay whole; rare or unknown words get split into pieces, down to individual characters if necessary.

```
"running"       → ["running"]          (common, stays whole)
"unhappiness"   → ["un", "happiness"]  (less common, split)
"Akwirw"        → ["Ak", "w", "ir", "w"]  (unknown, split to near-chars)
```

![BPE subword splitting](../../public/ch02/bpe-subword.png)

---

## How the vocabulary is built

BPE builds its vocabulary **bottom-up** from the training corpus.

### Step 0 — Start with characters

```
corpus = "low low low lowest newest"

initial tokens: ["l","o","w"," ","l","o","w"," ","l","o","w"," ","l","o","w","e","s","t"," ","n","e","w","e","s","t"]
```

In practice, words are pre-split by whitespace first, and a special end-of-word marker is added (e.g. `</w>`). Then the vocabulary starts as the set of all individual characters.

### Step 1 — Count all adjacent pairs

```
pair frequencies:
  (l, o)  → 4
  (o, w)  → 4
  (w, e)  → 2
  (e, s)  → 2
  ...
```

### Step 2 — Merge the most frequent pair

```
most frequent: (l, o)  → merge into "lo"

new token list: ["lo","w"," ","lo","w"," ","lo","w"," ","lo","w","e","s","t"," ","n","e","w","e","s","t"]
```

The merge rule `(l, o) → lo` is saved. Its **rank** = the order it was created (earlier = higher priority).

### Step 3 — Repeat

```
most frequent: (lo, w) → merge into "low"
most frequent: (e, s)  → merge into "es"
most frequent: (es, t) → merge into "est"
...
```

Keep merging until vocabulary reaches the target size. GPT-2 performs **50,000 merges** on top of 256 base byte tokens + 1 special token = **50,257 total**.

---

## Two files that encode a trained BPE

When you download GPT-2 weights, you get two files:

### `encoder.json`
Maps every token string → integer ID.
```json
{
  "!": 0,
  "\"": 1,
  ...
  "Ġthe": 262,
  ...
  "<|endoftext|>": 50256
}
```

Note: GPT-2 uses a `bytes_to_unicode` mapping so all byte values (0–255) become printable characters. The space character `0x20` becomes `Ġ`. This is why `" the"` appears as `"Ġthe"` in the file.

### `vocab.bpe`
Lists all merge rules, one per line, in the order they were created.
Line number = rank (priority). Lower rank = applied first.
```
#version: 0.2
Ġ t
Ġ a
h e
i n
r e
...
```

---

## How encoding works at inference time

Given the string `"lowest"`:

### Step 1 — Pre-tokenization
Split by whitespace and punctuation using GPT-2's regex pattern. Each piece is processed independently.

```
"lowest" → one piece: "lowest"
```

### Step 2 — Convert to bytes → unicode representation
```
"lowest" → "lowest"   (all ASCII, no change)
" lowest" → "Ġlowest"  (leading space → Ġ)
```

### Step 3 — Start as individual characters
```
["l", "o", "w", "e", "s", "t"]
```

### Step 4 — Apply merge rules by rank

Look at all adjacent pairs, find the one with the **lowest rank** (earliest merge), apply it:

```
round 1: (l,o) has rank 3 → merge → ["lo", "w", "e", "s", "t"]
round 2: (lo,w) has rank 7 → merge → ["low", "e", "s", "t"]
round 3: (e,s) has rank 12 → merge → ["low", "es", "t"]
round 4: (es,t) has rank 15 → merge → ["low", "est"]
round 5: (low,est) has rank 21 → merge → ["lowest"]
no more pairs in merge table → stop
```

### Step 5 — Look up IDs in `encoder.json`
```
"lowest" → 9CBA  (some integer ID)
```

---

## Special token: `<|endoftext|>`

This token is **not** processed through BPE. It's detected before everything else and directly assigned ID `50256`.

```
"Hello <|endoftext|> world"
  → ["Hello", "<|endoftext|>", "world"]   (split first)
  → BPE on "Hello", BPE on "world"
  → ID 50256 for <|endoftext|>
```

---

## Decoding

Reverse lookup: ID → token string → concatenate → convert Ġ back to spaces → raw text.

```
[15496, 11, 995] → ["Hello", ",", "Ġworld"] → "Hello, world"
```

Decoding is straightforward because each ID maps to exactly one byte sequence.

---

## GPT-2 pre-tokenization regex

Before BPE merges, GPT-2 splits input text using this regex:

```
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

This handles contractions (`don't` → `don` + `'t`), words, numbers, punctuation, and whitespace as separate units. Each unit is then BPE-encoded independently — merges **never cross word boundaries**.

---

## Test vectors (from the book)

| Input | Expected token IDs |
|-------|-------------------|
| `"Akwirw ier"` | `[33901, 86, 343, 86, 220, 959]` |
| `the-verdict.txt` (full) | 5,145 tokens total |
| `"Hello, do you like tea?"` | `[15496, 11, 466, 345, 588, 8887, 30]` |
| `"<\|endoftext\|>"` | `[50256]` |

---

## Summary

| Property | Value |
|----------|-------|
| Vocab size | 50,257 |
| Base tokens | 256 bytes |
| Merge rules | 50,000 |
| Special tokens | 1 (`<\|endoftext\|>` = 50256) |
| Unknown word handling | Always splits to bytes — no `<\|unk\|>` needed |
| Merge direction | Greedy, lowest rank first |
| Word boundaries | Pre-tokenizer ensures merges never cross them |
