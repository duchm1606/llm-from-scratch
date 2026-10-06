#ifndef LLM_LINEAR_H
#define LLM_LINEAR_H

#include <stddef.h>

#include "rng.h"

/*
 * Linear layer, nn.Linear(d_in, d_out) (section 3.4.2). The weight is a
 * row-major [d_out x d_in] float array and the bias has d_out entries: the
 * layout of nn.Linear.weight and nn.Linear.bias. A NULL bias means
 * bias=False. Callers own all memory; nothing here allocates.
 */

/*
 * nn.Linear's default initialization: the weight, then the bias, drawn from
 * U(-1/sqrt(d_in), 1/sqrt(d_in)). (PyTorch's kaiming_uniform_ with
 * a=sqrt(5) reduces to the same bound for the weight.)
 */
void linear_init(float *weight, float *bias, size_t d_in, size_t d_out, Rng *rng);

/*
 * nn.Linear forward on n rows: out = x @ weight.T + bias for x [n x d_in],
 * giving [n x d_out]. out must not overlap x.
 */
void linear_forward(float *out, const float *x, size_t n, const float *weight, const float *bias,
                    size_t d_in, size_t d_out);

#endif
