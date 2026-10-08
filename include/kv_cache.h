#ifndef LLM_KV_CACHE_H
#define LLM_KV_CACHE_H

#include <stddef.h>

#include "gpt.h"

/*
 * KV cache for generation (the book's bonus material on KV caching, after
 * chapter 4). generate_text_simple runs the whole context through the model
 * for every new token, although only the new token's row is new: under the
 * causal mask, the keys and values of earlier tokens never change. The cache
 * keeps them, one [capacity x emb_dim] matrix of keys and one of values per
 * block, so each step runs only the new tokens, which attend to the cached
 * rows and their own.
 *
 * Positions are absolute, so tokens can only be appended; past capacity the
 * context must be rebuilt from scratch. Set len to 0 to start over. The
 * struct points at caller-owned memory.
 */
typedef struct {
    size_t n_layers, emb_dim;
    size_t capacity; /* most tokens it holds, at most context_length */
    size_t len;      /* tokens held: the position of the next one */
    float *keys;     /* [n_layers x capacity x emb_dim] */
    float *values;   /* [n_layers x capacity x emb_dim] */
} KVCache;

/* Floats of memory a cache of capacity tokens needs for cfg. */
size_t kv_cache_size(const GPTConfig *cfg, size_t capacity);

/* Sets up an empty cache of capacity tokens on kv_cache_size(cfg, capacity)
 * floats at mem. */
void kv_cache_bind(KVCache *cache, const GPTConfig *cfg, size_t capacity, float *mem);

/* Floats of scratch gpt_forward_cached needs for n new tokens. */
size_t gpt_cached_scratch_size(const GPTModel *model, size_t n);

/*
 * The model in evaluation mode on n >= 1 new token ids at positions
 * cache->len onwards, with cache->len + n <= cache->capacity. Appends their
 * keys and values to the cache and gives logits [vocab_size] for the token
 * after the last one: the last row of gpt_forward on the cached tokens
 * followed by ids.
 */
void gpt_forward_cached(float *logits, const int *ids, size_t n, const GPTModel *model,
                        KVCache *cache, float *scratch);

#endif
