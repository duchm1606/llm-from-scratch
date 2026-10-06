#include "attention.h"

#include <math.h>

#include "linear.h"
#include "ops.h"

void attention_forward(float *out, float *weights, const float *q, const float *k, const float *v,
                       size_t T, size_t d)
{
    /* attn_scores / d_k**0.5: PyTorch divides by sqrt(d) rounded to float. */
    float scale = (float)sqrt((double)d);
    matmul_bt(weights, q, k, T, d, T);
    for (size_t i = 0; i < T * T; i++)
        weights[i] /= scale;
    softmax_rows(weights, T, T);
    matmul(out, weights, v, T, T, d);
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

void self_attention_forward(float *out, const float *x, size_t T, const SelfAttention *sa,
                            float *scratch)
{
    size_t d_in = sa->d_in, d_out = sa->d_out;
    float *q = scratch;
    float *k = q + T * d_out;
    float *v = k + T * d_out;
    float *weights = v + T * d_out;
    linear_forward(q, x, T, sa->w_query, sa->b_query, d_in, d_out);
    linear_forward(k, x, T, sa->w_key, sa->b_key, d_in, d_out);
    linear_forward(v, x, T, sa->w_value, sa->b_value, d_in, d_out);
    attention_forward(out, weights, q, k, v, T, d_out);
}
