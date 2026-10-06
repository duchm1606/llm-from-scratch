#include "attention.h"

#include <math.h>

#include "dropout.h"
#include "linear.h"
#include "ops.h"

void attention_forward(float *out, float *weights, const float *q, const float *k, const float *v,
                       size_t T, size_t d, size_t ld, bool causal, double dropout, Rng *rng)
{
    /* attn_scores / d_k**0.5: PyTorch divides by sqrt(d) rounded to float. */
    float scale = (float)sqrt((double)d);
    for (size_t i = 0; i < T; i++) {
        /* Token i sees tokens [0, seen); the rest get no weight. */
        size_t seen = causal ? i + 1 : T;
        const float *q_i = q + i * ld;
        float *w_i = weights + i * T;
        for (size_t j = 0; j < seen; j++) {
            const float *k_j = k + j * ld;
            float dot = 0;
            for (size_t c = 0; c < d; c++)
                dot += q_i[c] * k_j[c];
            w_i[j] = dot / scale;
        }
        for (size_t j = seen; j < T; j++)
            w_i[j] = -INFINITY;
    }
    softmax_rows(weights, T, T);
    if (rng)
        dropout_forward(weights, T * T, dropout, rng);

    for (size_t i = 0; i < T; i++) {
        size_t seen = causal ? i + 1 : T;
        const float *w_i = weights + i * T;
        float *o = out + i * ld;
        for (size_t c = 0; c < d; c++)
            o[c] = 0;
        for (size_t j = 0; j < seen; j++) {
            const float *v_j = v + j * ld;
            for (size_t c = 0; c < d; c++)
                o[c] += w_i[j] * v_j[c];
        }
    }
}

void self_attention_init(const SelfAttention *sa, Rng *rng)
{
    linear_init(sa->w_query, sa->b_query, sa->d_in, sa->d_out, rng);
    linear_init(sa->w_key, sa->b_key, sa->d_in, sa->d_out, rng);
    linear_init(sa->w_value, sa->b_value, sa->d_in, sa->d_out, rng);
}

size_t self_attention_scratch_size(const SelfAttention *sa, size_t T)
{
    return 3 * T * sa->d_out + T * T;
}

void self_attention_forward(float *out, const float *x, size_t B, size_t T,
                            const SelfAttention *sa, float *scratch, Rng *rng)
{
    size_t d_in = sa->d_in, d_out = sa->d_out;
    float *q = scratch;
    float *k = q + T * d_out;
    float *v = k + T * d_out;
    float *weights = v + T * d_out;
    /* One sequence at a time: the projections act row by row, and dropout
     * reaches each sequence's weights in turn, as in one [B x T x T] draw. */
    for (size_t b = 0; b < B; b++) {
        const float *x_b = x + b * T * d_in;
        linear_forward(q, x_b, T, sa->w_query, sa->b_query, d_in, d_out);
        linear_forward(k, x_b, T, sa->w_key, sa->b_key, d_in, d_out);
        linear_forward(v, x_b, T, sa->w_value, sa->b_value, d_in, d_out);
        attention_forward(out + b * T * d_out, weights, q, k, v, T, d_out, d_out, sa->causal,
                          sa->dropout, rng);
    }
}

void multi_head_attention_init(const MultiHeadAttention *mha, Rng *rng)
{
    linear_init(mha->w_query, mha->b_query, mha->d_in, mha->d_out, rng);
    linear_init(mha->w_key, mha->b_key, mha->d_in, mha->d_out, rng);
    linear_init(mha->w_value, mha->b_value, mha->d_in, mha->d_out, rng);
    linear_init(mha->w_out, mha->b_out, mha->d_out, mha->d_out, rng);
}

size_t multi_head_attention_scratch_size(const MultiHeadAttention *mha, size_t T)
{
    return 4 * T * mha->d_out + T * T;
}

void multi_head_attention_forward(float *out, const float *x, size_t B, size_t T,
                                  const MultiHeadAttention *mha, float *scratch, Rng *rng)
{
    size_t d_in = mha->d_in, d_out = mha->d_out;
    size_t head_dim = d_out / mha->num_heads;
    float *q = scratch;
    float *k = q + T * d_out;
    float *v = k + T * d_out;
    float *context = v + T * d_out;
    float *weights = context + T * d_out;
    /* Sequence by sequence and head by head, so dropout draws in the order
     * of PyTorch's [B x num_heads x T x T] attention weights. */
    for (size_t b = 0; b < B; b++) {
        const float *x_b = x + b * T * d_in;
        linear_forward(q, x_b, T, mha->w_query, mha->b_query, d_in, d_out);
        linear_forward(k, x_b, T, mha->w_key, mha->b_key, d_in, d_out);
        linear_forward(v, x_b, T, mha->w_value, mha->b_value, d_in, d_out);
        /* Each head reads its columns of q, k and v and writes the same
         * columns of context, which leaves the heads concatenated. */
        for (size_t h = 0; h < mha->num_heads; h++) {
            size_t col = h * head_dim;
            attention_forward(context + col, weights, q + col, k + col, v + col, T, head_dim, d_out,
                              true, mha->dropout, rng);
        }
        linear_forward(out + b * T * d_out, context, T, mha->w_out, mha->b_out, d_out, d_out);
    }
}
