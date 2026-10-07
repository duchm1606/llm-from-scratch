#include "transformer.h"

#include "dropout.h"
#include "ops.h"

size_t transformer_block_num_params(size_t emb_dim, bool qkv_bias)
{
    size_t d = emb_dim, h = 4 * emb_dim;
    size_t att = 4 * d * d + (qkv_bias ? 3 * d : 0) + d;
    size_t ff = h * d + h + d * h + d;
    return att + ff + 4 * d;
}

/* Hands out the next n floats of *p. */
static float *take(float **p, size_t n)
{
    float *start = *p;
    *p += n;
    return start;
}

void transformer_block_bind(TransformerBlock *tb, float *params, size_t emb_dim, size_t n_heads,
                            bool qkv_bias, double dropout)
{
    size_t d = emb_dim, h = 4 * emb_dim;
    float *p = params;
    MultiHeadAttention *att = &tb->att;
    *att = (MultiHeadAttention){.d_in = d, .d_out = d, .num_heads = n_heads, .dropout = dropout};
    att->w_query = take(&p, d * d);
    att->b_query = qkv_bias ? take(&p, d) : NULL;
    att->w_key = take(&p, d * d);
    att->b_key = qkv_bias ? take(&p, d) : NULL;
    att->w_value = take(&p, d * d);
    att->b_value = qkv_bias ? take(&p, d) : NULL;
    att->w_out = take(&p, d * d);
    att->b_out = take(&p, d);
    FeedForward *ff = &tb->ff;
    *ff = (FeedForward){.emb_dim = d, .hidden_dim = h};
    ff->w_in = take(&p, h * d);
    ff->b_in = take(&p, h);
    ff->w_out = take(&p, d * h);
    ff->b_out = take(&p, d);
    tb->norm1.dim = d;
    tb->norm1.scale = take(&p, d);
    tb->norm1.shift = take(&p, d);
    tb->norm2.dim = d;
    tb->norm2.scale = take(&p, d);
    tb->norm2.shift = take(&p, d);
    tb->drop_rate = dropout;
}

void transformer_block_init(const TransformerBlock *tb, Rng *rng)
{
    multi_head_attention_init(&tb->att, rng);
    feed_forward_init(&tb->ff, rng);
    layernorm_init(&tb->norm1);
    layernorm_init(&tb->norm2);
}

size_t transformer_block_scratch_size(const TransformerBlock *tb, size_t B, size_t T)
{
    size_t att = multi_head_attention_scratch_size(&tb->att, T);
    size_t ff = feed_forward_scratch_size(&tb->ff);
    return 2 * B * T * tb->ff.emb_dim + (att > ff ? att : ff);
}

void transformer_block_forward(float *out, const float *x, size_t B, size_t T,
                               const TransformerBlock *tb, float *scratch, Rng *rng)
{
    size_t n = B * T, size = n * tb->ff.emb_dim;
    float *normed = scratch;
    float *branch = normed + size;
    float *rest = branch + size;

    /* Attention branch, written straight into out. */
    layernorm_forward(normed, x, n, &tb->norm1);
    multi_head_attention_forward(out, normed, B, T, &tb->att, rest, rng);
    if (rng)
        dropout_forward(out, size, tb->drop_rate, rng);
    residual_add(out, x, size);

    /* Feed forward branch on the result. */
    layernorm_forward(normed, out, n, &tb->norm2);
    feed_forward_forward(branch, normed, n, &tb->ff, rest);
    if (rng)
        dropout_forward(branch, size, tb->drop_rate, rng);
    residual_add(out, branch, size);
}
