#include "activation.h"

#include <math.h>

void gelu_forward(float *x, size_t n)
{
    /* torch.sqrt(torch.tensor(2.0 / torch.pi)): the square root in float of
     * 2 / pi rounded to float. */
    const float c = sqrtf((float)(2 / 3.14159265358979323846));
    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        float cube = v * v * v;
        x[i] = 0.5f * v * (1 + tanhf(c * (v + 0.044715f * cube)));
    }
}
