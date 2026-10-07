#include "layernorm.h"

#include <math.h>

void layernorm_init(const LayerNorm *ln)
{
    for (size_t i = 0; i < ln->dim; i++) {
        ln->scale[i] = 1;
        ln->shift[i] = 0;
    }
}

void layernorm_forward(float *out, const float *x, size_t n, const LayerNorm *ln)
{
    size_t d = ln->dim;
    for (size_t r = 0; r < n; r++) {
        const float *x_r = x + r * d;
        float *o = out + r * d;
        /* PyTorch reduces in higher precision and rounds the mean and the
         * variance to float; the rest is float arithmetic. */
        double sum = 0;
        for (size_t i = 0; i < d; i++)
            sum += x_r[i];
        float mean = (float)(sum / (double)d);
        double sq = 0;
        for (size_t i = 0; i < d; i++) {
            double diff = (double)x_r[i] - mean;
            sq += diff * diff;
        }
        float var = (float)(sq / (double)d);
        float std = sqrtf(var + LAYERNORM_EPS);
        for (size_t i = 0; i < d; i++)
            o[i] = ln->scale[i] * ((x_r[i] - mean) / std) + ln->shift[i];
    }
}
