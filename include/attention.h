#ifndef LLM_ATTENTION_H
#define LLM_ATTENTION_H

#include <stddef.h>

#include "rng.h"

/*
 * Attention (chapter 3). Matrices are row-major float arrays; callers own
 * all memory, scratch space included, and nothing here allocates.
 */

/*
 * Scaled dot-product attention over T tokens (section 3.4.1), with queries,
 * keys and values q, k and v of [T x d] each:
 *
 *   weights = softmax(q @ k.T / sqrt(d))    [T x T]
 *   out     = weights @ v                   [T x d]
 *
 * weights is room for the [T x T] attention weights and is left holding
 * them. out must not overlap the inputs.
 */
void attention_forward(float *out, float *weights, const float *q, const float *k, const float *v,
                       size_t T, size_t d);

/*
 * Self-attention with trainable weights, SelfAttention_v2 (section 3.4.2):
 * three nn.Linear(d_in, d_out) layers project the input to queries, keys and
 * values for attention_forward. The struct points at caller-owned arrays in
 * nn.Linear's layout.
 */
typedef struct {
    size_t d_in, d_out;
    float *w_query, *w_key, *w_value; /* [d_out x d_in] each */
    float *b_query, *b_key, *b_value; /* [d_out] each, or NULL for qkv_bias=False */
} SelfAttention;

/* SelfAttention_v2's initialization: linear_init on W_query, W_key and
 * W_value, in that order. */
void self_attention_init(const SelfAttention *sa, Rng *rng);

/* Floats of scratch self_attention_forward needs for T tokens. */
size_t self_attention_scratch_size(const SelfAttention *sa, size_t T);

/* Self-attention over x [T x d_in], giving out [T x d_out]. */
void self_attention_forward(float *out, const float *x, size_t T, const SelfAttention *sa,
                            float *scratch);

#endif
