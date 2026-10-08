#include <stdio.h>
#include <stdlib.h>

#include "gpt.h"
#include "kv_cache.h"
#include "rng.h"
#include "test_util.h"

/* The cached model is checked against gpt_forward, itself checked against
 * PyTorch in test_gpt. Both do the same float operations in the same
 * order, so the tolerance only covers the compiler fusing them differently. */
#define TOL 1e-6f

enum { GUARD = 16 };

#define SENTINEL -1234.5f

/* malloc'd room for n floats followed by GUARD sentinels, or NULL. */
static float *guarded(size_t n)
{
    float *p = malloc(sizeof(float) * (n + GUARD));
    if (!p) {
        fprintf(stderr, "FAIL out of memory\n");
        failures++;
        return NULL;
    }
    for (size_t i = 0; i < GUARD; i++)
        p[n + i] = SENTINEL;
    return p;
}

static bool intact(const float *p, size_t n)
{
    bool ok = true;
    for (size_t i = 0; i < GUARD; i++)
        ok &= p[n + i] == SENTINEL;
    return ok;
}

/* The last row of gpt_forward's logits on ids [1 x T] into out [V]. */
static void last_logits(float *out, const int *ids, size_t T, const GPTModel *model)
{
    size_t V = model->cfg.vocab_size;
    float *logits = malloc(sizeof(float) * T * V);
    float *scratch = malloc(sizeof(float) * gpt_scratch_size(model, 1, T));
    if (!logits || !scratch) {
        fprintf(stderr, "FAIL out of memory\n");
        failures++;
    } else {
        gpt_forward(logits, ids, 1, T, model, scratch, NULL);
        for (size_t v = 0; v < V; v++)
            out[v] = logits[(T - 1) * V + v];
    }
    free(logits);
    free(scratch);
}

/*
 * A random model on cfg, fed ids in chunks: a prompt, single tokens as in
 * generation, and a longer chunk, filling the whole context. After each
 * chunk the cached logits must match the last row of gpt_forward on
 * everything so far. Then the cache is emptied and the first chunk run
 * again, which must give its logits again.
 */
static void check_config(const GPTConfig *cfg, const int *ids, const size_t *chunks,
                         size_t n_chunks)
{
    size_t T = cfg->context_length, V = cfg->vocab_size, longest = 0;
    for (size_t c = 0; c < n_chunks; c++)
        longest = chunks[c] > longest ? chunks[c] : longest;

    float *params = malloc(sizeof(float) * gpt_num_params(cfg));
    if (!params) {
        fprintf(stderr, "FAIL out of memory\n");
        failures++;
        return;
    }
    GPTModel model;
    gpt_bind(&model, cfg, params);
    size_t cache_size = kv_cache_size(cfg, T);
    size_t scratch_size = gpt_cached_scratch_size(&model, longest);
    float *mem = guarded(cache_size);
    float *scratch = guarded(scratch_size);
    float *logits = guarded(V);
    float *want = malloc(sizeof(float) * V);
    float *first = malloc(sizeof(float) * V);
    if (!mem || !scratch || !logits || !want || !first) {
        fprintf(stderr, "FAIL out of memory\n");
        failures++;
        goto done;
    }

    Rng rng;
    rng_seed(&rng, 123);
    gpt_init(&model, &rng);

    KVCache cache;
    kv_cache_bind(&cache, cfg, T, mem);
    CHECK(cache.len == 0);
    size_t seen = 0;
    for (size_t c = 0; c < n_chunks; c++) {
        gpt_forward_cached(logits, ids + seen, chunks[c], &model, &cache, scratch);
        seen += chunks[c];
        CHECK(cache.len == seen);
        last_logits(want, ids, seen, &model);
        CHECK(floats_close(logits, want, V, TOL));
        if (c == 0)
            for (size_t v = 0; v < V; v++)
                first[v] = logits[v];
    }
    CHECK(seen == T);

    cache.len = 0;
    gpt_forward_cached(logits, ids, chunks[0], &model, &cache, scratch);
    CHECK(cache.len == chunks[0]);
    CHECK(floats_close(logits, first, V, 0));

    CHECK(intact(mem, cache_size));
    CHECK(intact(scratch, scratch_size));
    CHECK(intact(logits, V));
done:
    free(params);
    free(mem);
    free(scratch);
    free(logits);
    free(want);
    free(first);
}

/* test_gpt's small model, with qkv_bias=True as GPT-2's checkpoints have. */
static void check_small(void)
{
    const GPTConfig cfg = {
        .vocab_size = 11, .context_length = 5, .emb_dim = 8, .n_heads = 2, .n_layers = 2,
        .drop_rate = 0.25, .qkv_bias = true,
    };
    static const int ids[] = {1, 4, 10, 0, 7};
    static const size_t chunks[] = {2, 1, 1, 1};
    check_config(&cfg, ids, chunks, sizeof chunks / sizeof chunks[0]);
}

/* More heads and blocks, no qkv bias, and a chunk after single tokens. */
static void check_wider(void)
{
    const GPTConfig cfg = {
        .vocab_size = 37, .context_length = 12, .emb_dim = 24, .n_heads = 4, .n_layers = 3,
        .drop_rate = 0.1, .qkv_bias = false,
    };
    static const int ids[] = {5, 36, 0, 12, 12, 7, 30, 2, 19, 1, 33, 8};
    static const size_t chunks[] = {4, 1, 1, 5, 1};
    check_config(&cfg, ids, chunks, sizeof chunks / sizeof chunks[0]);
}

/* A single token from an empty cache, as when a context starts with it. */
static void check_one_token(void)
{
    const GPTConfig cfg = {
        .vocab_size = 11, .context_length = 1, .emb_dim = 8, .n_heads = 2, .n_layers = 2,
        .drop_rate = 0.1, .qkv_bias = true,
    };
    static const int ids[] = {3};
    static const size_t chunks[] = {1};
    check_config(&cfg, ids, chunks, 1);
}

int main(void)
{
    check_small();
    check_wider();
    check_one_token();
    return test_report("test_kv_cache");
}
