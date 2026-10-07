#include <stdio.h>

#include "activation.h"
#include "test_util.h"

/* Expectations come from PyTorch 2.14 on CPU, running the book's GELU
 * class; the book only plots it. */
#define TOL 1e-6f

/* Listing 4.3 at points across figure 4.8's range and beyond. */
static void check_values(void)
{
    enum { N = 12 };
    float x[N] = {-10, -3, -2, -1, -0.75f, -0.5f, 0, 0.5f, 1, 2, 3, 10};
    static const float want[N] = {
        -0.0f,        -0.00363743305f, -0.0454022884f, -0.158808023f,
        -0.170039445f, -0.154285997f,  0,              0.345714003f,
        0.841192007f, 1.95459771f,     2.99636269f,    10,
    };
    gelu_forward(x, N);
    CHECK(floats_close(x, want, N, TOL));
}

/* Figure 4.8: on torch.linspace(-3, 3, 100) GELU stays above ReLU's 0 only
 * for positive inputs, and its minimum, about -0.17, sits near x = -0.75. */
static void check_shape(void)
{
    enum { N = 100 };
    float x[N], y[N];
    for (size_t i = 0; i < N; i++)
        x[i] = y[i] = -3 + 6 * (float)i / (N - 1);
    gelu_forward(y, N);
    size_t min = 0;
    bool sign_ok = true;
    for (size_t i = 0; i < N; i++) {
        if (y[i] < y[min])
            min = i;
        sign_ok &= x[i] > 0 ? y[i] > 0 && y[i] < x[i] : y[i] <= 0;
    }
    CHECK(sign_ok);
    CHECK(min == 37);
    CHECK(floats_close(&y[min], (float[]){-0.170035154f}, 1, TOL));
}

int main(void)
{
    check_values();
    check_shape();
    return test_report("test_activation");
}
