#ifndef LLM_TRANSFORMER_H
#define LLM_TRANSFORMER_H

#include <stdbool.h>
#include <stddef.h>

#include "attention.h"
#include "feedforward.h"
#include "layernorm.h"
#include "rng.h"

/*
 * The transformer block of GPT, TransformerBlock (section 4.5, Listing
 * 4.6). Two branches, each normalized first (Pre-LayerNorm), regularized by
 * dropout and wrapped in a shortcut connection:
 *
 *   x = x + dropout(att(norm1(x)))     tokens exchange information
 *   x = x + dropout(ff(norm2(x)))      each token is transformed on its own
 *
 * att is causal multi-head attention and ff the feed forward network, both
 * on emb_dim = att.d_in = att.d_out = ff.emb_dim, so the block maps
 * [B x T x emb_dim] to the same shape and blocks stack. drop_rate is the
 * block's drop_shortcut; attention's own dropout rate is att.dropout.
 */
typedef struct {
    LayerNorm norm1;
    MultiHeadAttention att;
    LayerNorm norm2;
    FeedForward ff;
    double drop_rate;
} TransformerBlock;

/*
 * Parameters of GPT's block with emb_dim dimensions, n_heads heads and
 * hidden_dim = 4 * emb_dim: 7,085,568 for GPT-2 small, of which attention
 * holds 2,360,064 and the feed forward network 4,722,432 (exercise 4.1).
 */
size_t transformer_block_num_params(size_t emb_dim, bool qkv_bias);

/*
 * Sets up tb as GPT's block on transformer_block_num_params(emb_dim,
 * qkv_bias) consecutive floats at params, in the order of PyTorch's
 * block.parameters(): att.W_query, W_key and W_value (each weight, then
 * bias if qkv_bias), att.out_proj, ff.layers[0], ff.layers[2], norm1 and
 * norm2. dropout is both drop_shortcut's and attention's rate.
 */
void transformer_block_bind(TransformerBlock *tb, float *params, size_t emb_dim, size_t n_heads,
                            bool qkv_bias, double dropout);

/* TransformerBlock's initialization: multi_head_attention_init on att,
 * feed_forward_init on ff, then layernorm_init on norm1 and norm2. */
void transformer_block_init(const TransformerBlock *tb, Rng *rng);

/* Floats of scratch transformer_block_forward needs for B sequences of T
 * tokens. It grows with B because each dropout covers the whole batch
 * before the next starts, as in PyTorch. */
size_t transformer_block_scratch_size(const TransformerBlock *tb, size_t B, size_t T);

/*
 * The block over a batch x [B x T x emb_dim], giving out [B x T x
 * emb_dim]; out must not overlap x. Dropout draws from rng in PyTorch's
 * order: attention's weights, the attention branch, then the feed forward
 * branch, each over the whole batch. A NULL rng means evaluation mode.
 */
void transformer_block_forward(float *out, const float *x, size_t B, size_t T,
                               const TransformerBlock *tb, float *scratch, Rng *rng);

#endif
