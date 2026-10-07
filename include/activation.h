#ifndef LLM_ACTIVATION_H
#define LLM_ACTIVATION_H

#include <stddef.h>

/*
 * GELU activation (section 4.3), in place on n floats, in the tanh
 * approximation GPT-2 was trained with (Listing 4.3):
 *
 *   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
 *
 * Like ReLU it passes large positive values and zeroes large negative ones,
 * but it is smooth, and small negative inputs give small negative outputs,
 * down to about -0.17 near x = -0.75.
 */
void gelu_forward(float *x, size_t n);

#endif
