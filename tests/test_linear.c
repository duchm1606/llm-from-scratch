#include <stdio.h>
#include <stdlib.h>

#include "ch03_inputs.h"
#include "linear.h"
#include "rng.h"
#include "test_util.h"

/* Expectations come from PyTorch 2.14 on CPU; the book prints no Linear
 * weights, only the attention outputs built on them. */
#define TOL 1e-6f

enum { D_OUT = 2 };

/* Section 3.4.2: torch.manual_seed(789); SelfAttention_v2(3, 2) creates
 * W_query, W_key and W_value as nn.Linear(3, 2, bias=False), in that order. */
static void check_without_bias(void)
{
    static const float want_query[D_OUT * D_IN] = {
        0.316059083f, 0.456808567f, 0.51183486f, -0.1682854f, -0.337877035f, -0.0917738676f,
    };
    static const float want_key[D_OUT * D_IN] = {
        0.405805826f, -0.470420539f, 0.236805201f, 0.213360742f, -0.260050654f, -0.510542989f,
    };
    static const float want_value[D_OUT * D_IN] = {
        0.252569884f, -0.14147827f, -0.196181342f, 0.519107401f, -0.0851675794f, -0.204327047f,
    };
    float w_query[D_OUT * D_IN], w_key[D_OUT * D_IN], w_value[D_OUT * D_IN];
    Rng rng;
    rng_seed(&rng, 789);
    linear_init(w_query, NULL, D_IN, D_OUT, &rng);
    linear_init(w_key, NULL, D_IN, D_OUT, &rng);
    linear_init(w_value, NULL, D_IN, D_OUT, &rng);
    CHECK(floats_close(w_query, want_query, D_OUT * D_IN, TOL));
    CHECK(floats_close(w_key, want_key, D_OUT * D_IN, TOL));
    CHECK(floats_close(w_value, want_value, D_OUT * D_IN, TOL));

    /* queries = W_query(inputs) */
    static const float want_queries[T * D_OUT] = {
        0.659959733f, -0.204723015f,
        0.909066975f, -0.447080731f,
        0.896015286f, -0.441853434f,
        0.503387451f, -0.263276845f,
        0.4087511f,   -0.223226398f,
        0.662759006f, -0.329191536f,
    };
    float queries[T * D_OUT];
    linear_forward(queries, inputs, T, w_query, NULL, D_IN, D_OUT);
    CHECK(floats_close(queries, want_queries, T * D_OUT, TOL));
}

/* torch.manual_seed(123); nn.Linear(3, 2): the weight is drawn before the
 * bias, from the same bound. */
static void check_with_bias(void)
{
    float weight[D_OUT * D_IN], bias[D_OUT];
    Rng rng;
    rng_seed(&rng, 123);
    linear_init(weight, bias, D_IN, D_OUT, &rng);
    CHECK(floats_close(weight,
                       (float[]){-0.235429645f, 0.0191244762f, -0.286745936f, 0.217726618f,
                                 -0.49193421f, 0.423223078f},
                       D_OUT * D_IN, TOL));
    CHECK(floats_close(bias, (float[]){-0.419641405f, -0.459017664f}, D_OUT, TOL));

    static const float want[T * D_OUT] = {
        -0.77321136f,  -0.0625168085f,
        -0.721741736f, -0.487923533f,
        -0.721097887f, -0.482194811f,
        -0.554969907f, -0.556776047f,
        -0.624815702f, -0.372029424f,
        -0.573823571f, -0.608906031f,
    };
    float out[T * D_OUT];
    linear_forward(out, inputs, T, weight, bias, D_IN, D_OUT);
    CHECK(floats_close(out, want, T * D_OUT, TOL));
}

enum { GPT2_DIM = 768 };

/* torch.manual_seed(123); nn.Linear(768, 768), the size of GPT-2's
 * projections, where the bound is 1/sqrt(768). */
static void check_gpt2_size(void)
{
    float *weight = malloc(sizeof(float) * GPT2_DIM * GPT2_DIM);
    float bias[GPT2_DIM];
    if (!weight) {
        fprintf(stderr, "FAIL out of memory\n");
        failures++;
        return;
    }
    Rng rng;
    rng_seed(&rng, 123);
    linear_init(weight, bias, GPT2_DIM, GPT2_DIM, &rng);
    CHECK(floats_close(weight, (float[]){-0.0147143528f, 0.00119527976f, -0.017921621f}, 3, TOL));
    CHECK(floats_close(&weight[GPT2_DIM * GPT2_DIM - 3],
                       (float[]){0.0334980674f, 0.0352249816f, 0.0215428453f}, 3, TOL));
    CHECK(floats_close(bias, (float[]){0.0151426587f, -0.0159845278f, 0.0273847431f}, 3, TOL));
    CHECK(floats_close(&bias[GPT2_DIM - 3],
                       (float[]){0.0172505826f, -0.00929774158f, -0.0128699364f}, 3, TOL));
    free(weight);
}

int main(void)
{
    check_without_bias();
    check_with_bias();
    check_gpt2_size();
    return test_report("test_linear");
}
