#ifndef LLM_LAYERNORM_H
#define LLM_LAYERNORM_H

#include <stddef.h>

/*
 * Layer normalization, LayerNorm (section 4.2). Each row of dim values is
 * shifted to mean 0 and scaled to variance 1, then multiplied by scale and
 * offset by shift, two learnable [dim] vectors:
 *
 *   out = scale * (x - mean) / sqrt(var + eps) + shift
 *
 * var is the biased variance, a sum of squares divided by dim
 * (unbiased=False), as in GPT-2, and eps = 1e-5 keeps the division finite
 * for a constant row. The struct points at caller-owned arrays.
 */
#define LAYERNORM_EPS 1e-5f

typedef struct {
    size_t dim;
    float *scale; /* [dim] */
    float *shift; /* [dim] */
} LayerNorm;

/* LayerNorm's initialization: scale all ones and shift all zeros. Draws
 * nothing from a generator. */
void layernorm_init(const LayerNorm *ln);

/* Normalizes n rows of x [n x dim] into out [n x dim]. out may be x. */
void layernorm_forward(float *out, const float *x, size_t n, const LayerNorm *ln);

#endif
