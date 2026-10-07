#ifndef LLM_GPT_H
#define LLM_GPT_H

#include <stdbool.h>
#include <stddef.h>

#include "layernorm.h"
#include "rng.h"
#include "transformer.h"

/*
 * The GPT model, GPTModel (section 4.6, Listing 4.7):
 *
 *   x      = dropout(tok_emb[ids] + pos_emb[0..T))   [B x T x emb_dim]
 *   x      = n_layers transformer blocks on x
 *   logits = out_head(final_norm(x))                  [B x T x vocab_size]
 *
 * Row t of the logits scores every vocabulary token as the one after token
 * t. out_head has no bias and its own weights, not tied to tok_emb.
 */
typedef struct {
    size_t vocab_size;     /* tokens in the vocabulary */
    size_t context_length; /* most tokens the positional embeddings cover */
    size_t emb_dim;        /* embedding dimension */
    size_t n_heads;        /* attention heads, dividing emb_dim */
    size_t n_layers;       /* transformer blocks */
    double drop_rate;      /* rate of every dropout layer */
    bool qkv_bias;         /* biases on the query, key and value projections */
} GPTConfig;

/* GPT_CONFIG_124M, GPT-2 small: 50257, 1024, 768, 12, 12, 0.1, false. */
extern const GPTConfig GPT_CONFIG_124M;

/* Parameters of the model: 163,009,536 for GPT_CONFIG_124M, or 124,412,160
 * without out_head, the count of GPT-2, which ties out_head to tok_emb. */
size_t gpt_num_params(const GPTConfig *cfg);

/*
 * The model's parameters live in one caller-owned buffer of
 * gpt_num_params(&cfg) floats, in the order of PyTorch's
 * model.parameters(): tok_emb, pos_emb, the blocks, final_norm and
 * out_head. tok_emb is the start of the buffer.
 */
typedef struct {
    GPTConfig cfg;
    float *tok_emb;       /* [vocab_size x emb_dim] */
    float *pos_emb;       /* [context_length x emb_dim] */
    float *blocks;        /* n_layers blocks, see gpt_block */
    LayerNorm final_norm; /* [emb_dim] scale and shift */
    float *out_head;      /* [vocab_size x emb_dim] */
} GPTModel;

/* Sets up model for cfg on gpt_num_params(cfg) floats at params. */
void gpt_bind(GPTModel *model, const GPTConfig *cfg, float *params);

/* Transformer block layer of the model, pointing into its parameters:
 * transformer_block_bind on transformer_block_num_params floats each. */
TransformerBlock gpt_block(const GPTModel *model, size_t layer);

/* GPTModel's initialization: embedding_init on tok_emb and pos_emb, each
 * block's transformer_block_init, layernorm_init on final_norm, then
 * linear_init on out_head. */
void gpt_init(const GPTModel *model, Rng *rng);

/* Floats of scratch gpt_forward needs for B sequences of T tokens. */
size_t gpt_scratch_size(const GPTModel *model, size_t B, size_t T);

/*
 * The model on a [B x T] batch of token ids, each < vocab_size, with
 * T <= context_length, giving logits [B x T x vocab_size]. Dropout draws
 * from rng in PyTorch's order, the embeddings' first; a NULL rng means
 * evaluation mode.
 */
void gpt_forward(float *logits, const int *ids, size_t B, size_t T, const GPTModel *model,
                 float *scratch, Rng *rng);

#endif
