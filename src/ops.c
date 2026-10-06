#include "ops.h"

#include <math.h>

void matmul(float *out, const float *a, const float *b, size_t m, size_t k, size_t n)
{
    /* Row i of out accumulates a[i][p] * (row p of b), so both b and out are
     * walked along rows. */
    for (size_t i = 0; i < m; i++) {
        float *o = out + i * n;
        for (size_t j = 0; j < n; j++)
            o[j] = 0;
        for (size_t p = 0; p < k; p++) {
            float a_ip = a[i * k + p];
            const float *b_p = b + p * n;
            for (size_t j = 0; j < n; j++)
                o[j] += a_ip * b_p[j];
        }
    }
}

void matmul_bt(float *out, const float *a, const float *b, size_t m, size_t k, size_t n)
{
    for (size_t i = 0; i < m; i++) {
        const float *a_i = a + i * k;
        for (size_t j = 0; j < n; j++) {
            const float *b_j = b + j * k;
            float dot = 0;
            for (size_t p = 0; p < k; p++)
                dot += a_i[p] * b_j[p];
            out[i * n + j] = dot;
        }
    }
}

void softmax_rows(float *x, size_t rows, size_t n)
{
    for (size_t r = 0; r < rows; r++) {
        float *row = x + r * n;
        float max = -INFINITY;
        for (size_t j = 0; j < n; j++)
            if (row[j] > max)
                max = row[j];
        float sum = 0;
        for (size_t j = 0; j < n; j++) {
            row[j] = expf(row[j] - max);
            sum += row[j];
        }
        float inv = 1 / sum;
        for (size_t j = 0; j < n; j++)
            row[j] *= inv;
    }
}
