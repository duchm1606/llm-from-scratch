#ifndef LLM_EMBEDDING_H
#define LLM_EMBEDDING_H

#include <stddef.h>

#include "rng.h"

/*
 * Embedding layers (sections 2.7-2.8). A table is a row-major [rows x dim]
 * float array: the layout of nn.Embedding.weight and of GPT-2's wte/wpe
 * checkpoint tensors. Callers own all memory; nothing here allocates.
 */

/* nn.Embedding's default initialization: every weight drawn from N(0, 1). */
void embedding_init(float *table, size_t rows, size_t dim, Rng *rng);

/* nn.Embedding forward: out[i] = table[ids[i]] for n ids, giving [n x dim].
 * Every id must be a row of table. */
void embedding_forward(float *out, const int *ids, size_t n, const float *table, size_t dim);

/*
 * GPT input embeddings for a [B x T] batch of token ids, giving [B x T x C]:
 *
 *   out[b][t] = wte[ids[b][t]] + wpe[t]
 *
 * wte is the token table [vocab_size x C], wpe the position table
 * [context_length x C] with context_length >= T. Every id must be < vocab_size.
 */
void input_embedding_forward(float *out, const int *ids, const float *wte, const float *wpe,
                             size_t B, size_t T, size_t C);

#endif
