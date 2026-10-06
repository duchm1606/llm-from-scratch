#include "dropout.h"

void dropout_forward(float *x, size_t n, double p, Rng *rng)
{
    if (p == 0)
        return;
    if (p == 1) {
        for (size_t i = 0; i < n; i++)
            x[i] = 0;
        return;
    }
    double keep = 1 - p;
    /* PyTorch multiplies by the float reciprocal, which can differ from
     * dividing by keep in the last bit. */
    float scale = 1 / (float)keep;
    for (size_t i = 0; i < n; i++)
        x[i] *= rng_bernoulli(rng, keep) ? scale : 0;
}
