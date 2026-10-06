#ifndef LLM_OPS_H
#define LLM_OPS_H

#include <stddef.h>

/*
 * Matrix operations shared by the layers (section 3.3). Matrices are
 * row-major float arrays; callers own all memory, and out must not overlap
 * any input.
 */

/* out = a @ b for a [m x k] and b [k x n], giving [m x n]. */
void matmul(float *out, const float *a, const float *b, size_t m, size_t k, size_t n);

/*
 * out = a @ b.T for a [m x k] and b [n x k], giving [m x n]: out[i][j] is the
 * dot product of row i of a with row j of b. This is the shape of attention
 * scores (queries @ keys.T) and of nn.Linear, whose weight is [out x in].
 */
void matmul_bt(float *out, const float *a, const float *b, size_t m, size_t k, size_t n);

/*
 * torch.softmax(x, dim=-1) in place on a [rows x n] matrix: each row becomes
 * exp(x - max) / sum(exp(x - max)). Subtracting the row max keeps exp from
 * overflowing. -inf entries become exactly 0, so every row needs at least one
 * finite entry.
 */
void softmax_rows(float *x, size_t rows, size_t n);

#endif
