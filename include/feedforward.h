#ifndef LLM_FEEDFORWARD_H
#define LLM_FEEDFORWARD_H

#include <stddef.h>

#include "rng.h"

/*
 * The transformer's feed forward network, FeedForward (section 4.3,
 * Listing 4.4): an nn.Linear(emb_dim, hidden_dim) that expands each token,
 * GELU, and an nn.Linear(hidden_dim, emb_dim) that contracts it back.
 * GPT-2 uses hidden_dim = 4 * emb_dim, 768 -> 3072 -> 768. Every token goes
 * through on its own, so the output has the input's shape. The struct
 * points at caller-owned arrays in nn.Linear's layout.
 */
typedef struct {
    size_t emb_dim, hidden_dim;
    float *w_in, *b_in;   /* layers[0]: [hidden_dim x emb_dim] and [hidden_dim] */
    float *w_out, *b_out; /* layers[2]: [emb_dim x hidden_dim] and [emb_dim] */
} FeedForward;

/* FeedForward's initialization: linear_init on layers[0], then layers[2]. */
void feed_forward_init(const FeedForward *ff, Rng *rng);

/* Floats of scratch feed_forward_forward needs, whatever the number of
 * tokens. */
size_t feed_forward_scratch_size(const FeedForward *ff);

/* Feed forward on n tokens x [n x emb_dim], giving out [n x emb_dim]. out
 * must not overlap x. */
void feed_forward_forward(float *out, const float *x, size_t n, const FeedForward *ff,
                          float *scratch);

#endif
