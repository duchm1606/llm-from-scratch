#include "embedding.h"

#include <string.h>

void embedding_init(float *table, size_t rows, size_t dim, Rng *rng)
{
    rng_normal(rng, table, rows * dim, 0, 1);
}

void embedding_forward(float *out, const int *ids, size_t n, const float *table, size_t dim)
{
    for (size_t i = 0; i < n; i++)
        memcpy(out + i * dim, table + (size_t)ids[i] * dim, dim * sizeof *out);
}

void input_embedding_forward(float *out, const int *ids, const float *wte, const float *wpe,
                             size_t B, size_t T, size_t C)
{
    for (size_t b = 0; b < B; b++) {
        for (size_t t = 0; t < T; t++) {
            float *o = out + (b * T + t) * C;
            const float *tok = wte + (size_t)ids[b * T + t] * C;
            const float *pos = wpe + t * C;
            for (size_t c = 0; c < C; c++)
                o[c] = tok[c] + pos[c];
        }
    }
}
