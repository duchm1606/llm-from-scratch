#ifndef LLM_ATTENTION_H
#define LLM_ATTENTION_H

#include <stdbool.h>
#include <stddef.h>

#include "rng.h"

/*
 * Attention (chapter 3). Matrices are row-major float arrays; callers own
 * all memory, scratch space included, and nothing here allocates.
 */

/*
 * Scaled dot-product attention over T tokens (sections 3.4-3.5), with
 * queries, keys and values q, k and v of [T x d] each:
 *
 *   weights = dropout(softmax(mask(q @ k.T) / sqrt(d)))    [T x T]
 *   out     = weights @ v                                 [T x d]
 *
 * With causal, mask sets the score of token i for every later token j > i
 * to -inf, so each token attends only to itself and earlier tokens (section
 * 3.5.1); otherwise it does nothing. dropout is dropout_forward with
 * probability dropout, drawing from rng (section 3.5.2); a NULL rng means
 * evaluation mode, without dropout.
 *
 * weights is room for the [T x T] attention weights and is left holding
 * them. out must not overlap the inputs.
 */
void attention_forward(float *out, float *weights, const float *q, const float *k, const float *v,
                       size_t T, size_t d, bool causal, double dropout, Rng *rng);

/*
 * Single-head self-attention with trainable weights: three nn.Linear(d_in,
 * d_out) layers project the input to queries, keys and values for
 * attention_forward. With causal and dropout left 0 this is SelfAttention_v2
 * (section 3.4.2); with causal set it is CausalAttention (section 3.5.3),
 * whose mask buffer the index test in attention_forward replaces. The struct
 * points at caller-owned arrays in nn.Linear's layout.
 */
typedef struct {
    size_t d_in, d_out;
    float *w_query, *w_key, *w_value; /* [d_out x d_in] each */
    float *b_query, *b_key, *b_value; /* [d_out] each, or NULL for qkv_bias=False */
    bool causal;
    double dropout; /* rate on the attention weights */
} SelfAttention;

/* Initialization of both classes: linear_init on W_query, W_key and W_value,
 * in that order. */
void self_attention_init(const SelfAttention *sa, Rng *rng);

/* Floats of scratch self_attention_forward needs for T tokens, whatever the
 * batch size. */
size_t self_attention_scratch_size(const SelfAttention *sa, size_t T);

/*
 * Self-attention over a batch x [B x T x d_in], giving out [B x T x d_out].
 * Dropout draws from rng in the order of PyTorch's [B x T x T] attention
 * weights; a NULL rng means evaluation mode.
 */
void self_attention_forward(float *out, const float *x, size_t B, size_t T,
                            const SelfAttention *sa, float *scratch, Rng *rng);

#endif
