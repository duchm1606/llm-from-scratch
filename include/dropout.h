#ifndef LLM_DROPOUT_H
#define LLM_DROPOUT_H

#include <stddef.h>

#include "rng.h"

/*
 * nn.Dropout(p) in training mode (section 3.5.2), in place on n floats: each
 * is zeroed with probability p (0 <= p <= 1), and the rest are scaled by
 * 1/(1 - p) so the expected value is unchanged. Each value takes one
 * rng_bernoulli draw, except that p = 0 and p = 1 draw nothing, as in
 * PyTorch. In evaluation mode dropout does nothing, so there is no call.
 */
void dropout_forward(float *x, size_t n, double p, Rng *rng);

#endif
