#include "linear.h"

#include <math.h>

#include "ops.h"

void linear_init(float *weight, float *bias, size_t d_in, size_t d_out, Rng *rng)
{
    /* PyTorch computes the bound in double and rounds it to float once. */
    float bound = (float)(1 / sqrt((double)d_in));
    rng_uniform(rng, weight, d_out * d_in, -bound, bound);
    if (bias)
        rng_uniform(rng, bias, d_out, -bound, bound);
}

void linear_forward(float *out, const float *x, size_t n, const float *weight, const float *bias,
                    size_t d_in, size_t d_out)
{
    matmul_bt(out, x, weight, n, d_in, d_out);
    if (!bias)
        return;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < d_out; j++)
            out[i * d_out + j] += bias[j];
}
