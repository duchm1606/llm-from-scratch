#include <stdio.h>
#include <string.h>

#include "layernorm.h"
#include "linear.h"
#include "rng.h"
#include "test_util.h"

/* Full-precision expectations come from PyTorch 2.14 on CPU; the book prints
 * four decimals. */
#define TOL 1e-6f
#define BOOK_TOL 5e-5f

enum { ROWS = 2, D = 5 };

/* Section 4.2: torch.manual_seed(123); batch_example = torch.randn(2, 5). */
static void batch_example(float *x)
{
    Rng rng;
    rng_seed(&rng, 123);
    rng_normal(&rng, x, ROWS * D, 0, 1);
}

/* Mean and biased variance of each row of x [ROWS x d]. */
static void row_stats(float *mean, float *var, const float *x, size_t d)
{
    for (size_t r = 0; r < ROWS; r++) {
        double sum = 0, sq = 0;
        for (size_t i = 0; i < d; i++)
            sum += x[r * d + i];
        double m = sum / (double)d;
        for (size_t i = 0; i < d; i++)
            sq += (x[r * d + i] - m) * (x[r * d + i] - m);
        mean[r] = (float)m;
        var[r] = (float)(sq / (double)d);
    }
}

/* Section 4.2's opening example: layer = nn.Sequential(nn.Linear(5, 6),
 * nn.ReLU()), created after batch_example, so its weights continue the same
 * generator. */
static void check_layer_outputs(void)
{
    enum { D_OUT = 6 };
    float x[ROWS * D], weight[D_OUT * D], bias[D_OUT], out[ROWS * D_OUT];
    Rng rng;
    rng_seed(&rng, 123);
    rng_normal(&rng, x, ROWS * D, 0, 1);
    linear_init(weight, bias, D, D_OUT, &rng);
    linear_forward(out, x, ROWS, weight, bias, D, D_OUT);
    for (size_t i = 0; i < ROWS * D_OUT; i++)
        out[i] = out[i] > 0 ? out[i] : 0;
    CHECK(floats_close(out,
                       (float[]){0.2260f, 0.3470f, 0.0000f, 0.2216f, 0.0000f, 0.0000f,
                                 0.2133f, 0.2394f, 0.0000f, 0.5198f, 0.3297f, 0.0000f},
                       ROWS * D_OUT, BOOK_TOL));
}

/* Listing 4.2: ln = LayerNorm(emb_dim=5) on batch_example. */
static void check_book_layernorm(void)
{
    float x[ROWS * D], scale[D], shift[D], out[ROWS * D];
    batch_example(x);
    CHECK(floats_close(x,
                       (float[]){-0.111467116f, 0.120362945f, -0.369634509f, -0.240417972f,
                                 -1.19692433f, 0.20926936f, -0.972355008f, -0.755045474f,
                                 0.323902756f, -0.108522631f},
                       ROWS * D, TOL));
    LayerNorm ln = {.dim = D, .scale = scale, .shift = shift};
    layernorm_init(&ln);
    layernorm_forward(out, x, ROWS, &ln);

    /* The book prints mean -0.0000 and 0.0000, variance 1.0000 for both
     * rows; eps keeps the variance just below 1. */
    float mean[ROWS], var[ROWS];
    row_stats(mean, var, out, D);
    CHECK(floats_close(mean, (float[]){0, 0}, ROWS, BOOK_TOL));
    CHECK(floats_close(var, (float[]){1, 1}, ROWS, BOOK_TOL));
    CHECK(floats_close(var, (float[]){0.999950409f, 0.999962628f}, ROWS, TOL));

    static const float want[ROWS * D] = {
        0.552836061f, 1.06931603f,  -0.0223191977f, 0.265554398f, -1.86538744f,
        0.908665478f, -1.37668276f, -0.956390142f,  1.13037491f,  0.294032484f,
    };
    CHECK(floats_close(out, want, ROWS * D, TOL));

    /* In place gives the same. */
    layernorm_forward(x, x, ROWS, &ln);
    CHECK(memcmp(x, out, sizeof out) == 0);
}

/* Trained scale and shift: ln.scale = [0.5, 1, 1.5, 2, -1] and ln.shift =
 * [0.1, -0.2, 0.3, 0, 1]. */
static void check_scale_shift(void)
{
    float x[ROWS * D], out[ROWS * D];
    float scale[D] = {0.5f, 1.0f, 1.5f, 2.0f, -1.0f};
    float shift[D] = {0.1f, -0.2f, 0.3f, 0.0f, 1.0f};
    batch_example(x);
    LayerNorm ln = {.dim = D, .scale = scale, .shift = shift};
    layernorm_forward(out, x, ROWS, &ln);
    static const float want[ROWS * D] = {
        0.376418024f, 0.869316041f,  0.266521215f, 0.531108797f, 2.86538744f,
        0.554332733f, -1.57668281f, -1.13458514f, 2.26074982f, 0.705967546f,
    };
    CHECK(floats_close(out, want, ROWS * D, TOL));

    /* A constant row has variance 0: eps keeps it finite, and every value
     * lands on its shift. */
    float row[D] = {3.25f, 3.25f, 3.25f, 3.25f, 3.25f};
    layernorm_forward(out, row, 1, &ln);
    CHECK(floats_close(out, shift, D, TOL));
}

/* torch.manual_seed(123); x = torch.rand(2, 768) * 10 - 3, then
 * LayerNorm(768): rows the size of GPT-2's embeddings. */
static void check_gpt2_size(void)
{
    enum { C = 768 };
    static float x[ROWS * C], out[ROWS * C], scale[C], shift[C];
    Rng rng;
    rng_seed(&rng, 123);
    rng_uniform(&rng, x, ROWS * C, 0, 1);
    for (size_t i = 0; i < ROWS * C; i++)
        x[i] = x[i] * 10 - 3;
    LayerNorm ln = {.dim = C, .scale = scale, .shift = shift};
    layernorm_init(&ln);
    layernorm_forward(out, x, ROWS, &ln);
    CHECK(floats_close(out, (float[]){-0.711680055f, 0.0561167561f, -0.866462469f}, 3, TOL));
    CHECK(floats_close(&out[ROWS * C - 3], (float[]){0.147266313f, -0.162256479f, 0.854961157f},
                       3, TOL));
    float mean[ROWS], var[ROWS];
    row_stats(mean, var, out, C);
    CHECK(floats_close(mean, (float[]){0, 0}, ROWS, TOL));
    CHECK(floats_close(var, (float[]){1, 1}, ROWS, 1e-4f));
}

int main(void)
{
    check_layer_outputs();
    check_book_layernorm();
    check_scale_shift();
    check_gpt2_size();
    return test_report("test_layernorm");
}
