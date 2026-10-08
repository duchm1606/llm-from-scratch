#include "kv_cache.h"

#include <math.h>

#include "embedding.h"
#include "linear.h"
#include "ops.h"

size_t kv_cache_size(const GPTConfig *cfg, size_t capacity)
{
    return 2 * cfg->n_layers * capacity * cfg->emb_dim;
}

void kv_cache_bind(KVCache *cache, const GPTConfig *cfg, size_t capacity, float *mem)
{
    size_t layer_size = cfg->n_layers * capacity * cfg->emb_dim;
    *cache = (KVCache){
        .n_layers = cfg->n_layers,
        .emb_dim = cfg->emb_dim,
        .capacity = capacity,
        .len = 0,
        .keys = mem,
        .values = mem + layer_size,
    };
}

size_t gpt_cached_scratch_size(const GPTModel *model, size_t n)
{
    const GPTConfig *cfg = &model->cfg;
    /* x, normed, q, context and branch, the attention weights of one query
     * and the feed forward network's hidden row. */
    return 5 * n * cfg->emb_dim + cfg->context_length + 4 * cfg->emb_dim;
}

/*
 * Causal multi-head attention of n queries q [n x d] at positions start
 * onwards over the keys and values of positions [0, start + n), giving
 * context [n x d]. The arithmetic is attention_forward's, query by query, so
 * the result matches the uncached model.
 */
static void cached_attention(float *context, float *weights, const float *q, const float *keys,
                             const float *values, size_t n, size_t start, size_t d,
                             size_t n_heads)
{
    size_t head_dim = d / n_heads;
    float scale = (float)sqrt((double)head_dim);
    for (size_t h = 0; h < n_heads; h++) {
        size_t col = h * head_dim;
        for (size_t i = 0; i < n; i++) {
            size_t seen = start + i + 1;
            const float *q_i = q + i * d + col;
            for (size_t j = 0; j < seen; j++) {
                const float *k_j = keys + j * d + col;
                float dot = 0;
                for (size_t c = 0; c < head_dim; c++)
                    dot += q_i[c] * k_j[c];
                weights[j] = dot / scale;
            }
            softmax_rows(weights, 1, seen);
            float *o = context + i * d + col;
            for (size_t c = 0; c < head_dim; c++)
                o[c] = 0;
            for (size_t j = 0; j < seen; j++) {
                const float *v_j = values + j * d + col;
                for (size_t c = 0; c < head_dim; c++)
                    o[c] += weights[j] * v_j[c];
            }
        }
    }
}

void gpt_forward_cached(float *logits, const int *ids, size_t n, const GPTModel *model,
                        KVCache *cache, float *scratch)
{
    const GPTConfig *cfg = &model->cfg;
    size_t d = cfg->emb_dim, size = n * d, start = cache->len;
    float *x = scratch;
    float *normed = x + size;
    float *q = normed + size;
    float *context = q + size;
    float *branch = context + size;
    float *weights = branch + size;
    float *hidden = weights + cfg->context_length;

    input_embedding_forward(x, ids, model->tok_emb, model->pos_emb + start * d, 1, n, d);
    for (size_t l = 0; l < cfg->n_layers; l++) {
        TransformerBlock tb = gpt_block(model, l);
        const MultiHeadAttention *att = &tb.att;
        float *keys = cache->keys + l * cache->capacity * d;
        float *values = cache->values + l * cache->capacity * d;

        /* transformer_block_forward on the new rows, whose keys and values
         * go straight into the cache after the earlier ones. */
        layernorm_forward(normed, x, n, &tb.norm1);
        linear_forward(q, normed, n, att->w_query, att->b_query, d, d);
        linear_forward(keys + start * d, normed, n, att->w_key, att->b_key, d, d);
        linear_forward(values + start * d, normed, n, att->w_value, att->b_value, d, d);
        cached_attention(context, weights, q, keys, values, n, start, d, att->num_heads);
        linear_forward(branch, context, n, att->w_out, att->b_out, d, d);
        residual_add(x, branch, size);

        layernorm_forward(normed, x, n, &tb.norm2);
        feed_forward_forward(branch, normed, n, &tb.ff, hidden);
        residual_add(x, branch, size);
    }
    cache->len += n;

    float *last = x + (n - 1) * d;
    layernorm_forward(last, last, 1, &model->final_norm);
    linear_forward(logits, last, 1, model->out_head, NULL, d, cfg->vocab_size);
}
