#include <stdio.h>
#include <stdlib.h>

#include "attention.h"
#include "ch03_inputs.h"
#include "ops.h"
#include "rng.h"
#include "test_util.h"

/* Full-precision expectations come from PyTorch 2.14 on CPU; the book prints
 * four decimals. */
#define TOL 1e-6f
#define BOOK_TOL 5e-5f

enum { D_OUT = 2, GUARD = 16 };

#define SENTINEL -1234.5f

/* self_attention_forward on the chapter's inputs. GUARD sentinels follow the
 * scratch it asks for and must come back untouched. */
static void run_self_attention(float *out, const SelfAttention *sa)
{
    size_t n = self_attention_scratch_size(sa, T);
    float *scratch = malloc(sizeof(float) * (n + GUARD));
    if (!scratch) {
        fprintf(stderr, "FAIL out of memory\n");
        failures++;
        return;
    }
    for (size_t i = 0; i < GUARD; i++)
        scratch[n + i] = SENTINEL;
    self_attention_forward(out, inputs, T, sa, scratch);
    bool intact = true;
    for (size_t i = 0; i < GUARD; i++)
        intact &= scratch[n + i] == SENTINEL;
    CHECK(intact);
    free(scratch);
}

/* Sections 3.4.1-3.4.2: torch.manual_seed(123), then three torch.rand(3, 2)
 * matrices, the weights of SelfAttention_v1. They are [d_in x d_out], the
 * transpose of nn.Linear's layout, so projecting is x @ W. */
static void check_v1(void)
{
    float w_query[D_IN * D_OUT], w_key[D_IN * D_OUT], w_value[D_IN * D_OUT];
    Rng rng;
    rng_seed(&rng, 123);
    rng_uniform(&rng, w_query, D_IN * D_OUT, 0, 1);
    rng_uniform(&rng, w_key, D_IN * D_OUT, 0, 1);
    rng_uniform(&rng, w_value, D_IN * D_OUT, 0, 1);
    CHECK(floats_close(w_query,
                       (float[]){0.296111941f, 0.516562283f, 0.251670718f, 0.68855679f,
                                 0.0739724636f, 0.866521955f},
                       D_IN * D_OUT, TOL));
    CHECK(floats_close(w_key,
                       (float[]){0.136579871f, 0.102479041f, 0.184056461f, 0.726446748f,
                                 0.315253913f, 0.687106669f},
                       D_IN * D_OUT, TOL));
    CHECK(floats_close(w_value,
                       (float[]){0.075635314f, 0.196638167f, 0.316411972f, 0.401740134f,
                                 0.118568301f, 0.82739538f},
                       D_IN * D_OUT, TOL));

    /* query_2 = x_2 @ W_query */
    float query_2[D_OUT];
    matmul(query_2, &inputs[1 * D_IN], w_query, 1, D_IN, D_OUT);
    CHECK(floats_close(query_2, (float[]){0.4306f, 1.4551f}, D_OUT, BOOK_TOL));

    float queries[T * D_OUT], keys[T * D_OUT], values[T * D_OUT];
    matmul(queries, inputs, w_query, T, D_IN, D_OUT);
    matmul(keys, inputs, w_key, T, D_IN, D_OUT);
    matmul(values, inputs, w_value, T, D_IN, D_OUT);

    /* attn_scores_2 = query_2 @ keys.T, whose second entry is attn_score_22 */
    float scores_2[T];
    matmul_bt(scores_2, query_2, keys, 1, D_OUT, T);
    CHECK(floats_close(&scores_2[1], (float[]){1.8524f}, 1, BOOK_TOL));
    CHECK(floats_close(scores_2, (float[]){1.2705f, 1.8524f, 1.8111f, 1.0795f, 0.5577f, 1.5440f},
                       T, BOOK_TOL));
    CHECK(floats_close(scores_2,
                       (float[]){1.27048302f, 1.85238445f, 1.81110668f, 1.07951725f,
                                 0.557730556f, 1.54397058f},
                       T, TOL));

    /* attn_weights_2 and context_vec_2 are row 2 of attention over all
     * queries at once. */
    float weights[T * T], context[T * D_OUT];
    attention_forward(context, weights, queries, keys, values, T, D_OUT);
    CHECK(floats_close(&weights[1 * T],
                       (float[]){0.1500f, 0.2264f, 0.2199f, 0.1311f, 0.0906f, 0.1820f}, T,
                       BOOK_TOL));
    CHECK(floats_close(&weights[1 * T],
                       (float[]){0.150019512f, 0.22638379f, 0.21987164f, 0.131070077f,
                                 0.0906289071f, 0.182026118f},
                       T, TOL));
    CHECK(floats_close(&context[1 * D_OUT], (float[]){0.3061f, 0.8210f}, D_OUT, BOOK_TOL));

    /* sa_v1(inputs) */
    static const float book_v1[T * D_OUT] = {
        0.2996f, 0.8053f,
        0.3061f, 0.8210f,
        0.3058f, 0.8203f,
        0.2948f, 0.7939f,
        0.2927f, 0.7891f,
        0.2990f, 0.8040f,
    };
    static const float torch_v1[T * D_OUT] = {
        0.299582064f, 0.805314124f,
        0.306100219f, 0.821030378f,
        0.305781096f, 0.820295751f,
        0.294765949f, 0.793866336f,
        0.292706072f, 0.789084315f,
        0.299010098f, 0.804036856f,
    };
    CHECK(floats_close(context, book_v1, T * D_OUT, BOOK_TOL));
    CHECK(floats_close(context, torch_v1, T * D_OUT, TOL));
}

/* Section 3.4.2: torch.manual_seed(789); SelfAttention_v2(3, 2). */
static void check_v2(void)
{
    float w[3][D_OUT * D_IN];
    SelfAttention sa = {.d_in = D_IN, .d_out = D_OUT, .w_query = w[0], .w_key = w[1], .w_value = w[2]};
    Rng rng;
    rng_seed(&rng, 789);
    self_attention_init(&sa, &rng);

    static const float book_v2[T * D_OUT] = {
        -0.0739f, 0.0713f,
        -0.0748f, 0.0703f,
        -0.0749f, 0.0702f,
        -0.0760f, 0.0685f,
        -0.0763f, 0.0679f,
        -0.0754f, 0.0693f,
    };
    static const float torch_v2[T * D_OUT] = {
        -0.0738902614f, 0.0712899044f,
        -0.0748107135f, 0.0703092963f,
        -0.0748561993f, 0.0702416599f,
        -0.0760016292f, 0.0684501082f,
        -0.0763275996f, 0.0679428056f,
        -0.0754442811f, 0.0693049207f,
    };
    float out[T * D_OUT];
    run_self_attention(out, &sa);
    CHECK(floats_close(out, book_v2, T * D_OUT, BOOK_TOL));
    CHECK(floats_close(out, torch_v2, T * D_OUT, TOL));

    /* Exercise 3.1: SelfAttention_v1 given the transposes of these weights
     * computes the same context vectors. */
    float w_t[3][D_IN * D_OUT];
    for (size_t m = 0; m < 3; m++)
        for (size_t i = 0; i < D_OUT; i++)
            for (size_t j = 0; j < D_IN; j++)
                w_t[m][j * D_OUT + i] = w[m][i * D_IN + j];
    float queries[T * D_OUT], keys[T * D_OUT], values[T * D_OUT];
    matmul(queries, inputs, w_t[0], T, D_IN, D_OUT);
    matmul(keys, inputs, w_t[1], T, D_IN, D_OUT);
    matmul(values, inputs, w_t[2], T, D_IN, D_OUT);
    float weights[T * T], v1_out[T * D_OUT];
    attention_forward(v1_out, weights, queries, keys, values, T, D_OUT);
    CHECK(floats_close(v1_out, out, T * D_OUT, TOL));
}

/* torch.manual_seed(123); SelfAttention_v2(3, 2, qkv_bias=True): each
 * projection draws its weight, then its bias. */
static void check_qkv_bias(void)
{
    float w[3][D_OUT * D_IN], b[3][D_OUT];
    SelfAttention sa = {
        .d_in = D_IN, .d_out = D_OUT,
        .w_query = w[0], .w_key = w[1], .w_value = w[2],
        .b_query = b[0], .b_key = b[1], .b_value = b[2],
    };
    Rng rng;
    rng_seed(&rng, 123);
    self_attention_init(&sa, &rng);
    CHECK(floats_close(b[0], (float[]){-0.419641405f, -0.459017664f}, D_OUT, TOL));
    CHECK(floats_close(b[2], (float[]){0.157876849f, 0.557292342f}, D_OUT, TOL));

    static const float want[T * D_OUT] = {
        0.10589961f,  0.929572642f,
        0.114426538f, 0.935321629f,
        0.114333838f, 0.935251594f,
        0.118122526f, 0.936930597f,
        0.113788687f, 0.934276819f,
        0.118761346f, 0.937503457f,
    };
    float out[T * D_OUT];
    run_self_attention(out, &sa);
    CHECK(floats_close(out, want, T * D_OUT, TOL));
}

int main(void)
{
    check_v1();
    check_v2();
    check_qkv_bias();
    return test_report("test_attention");
}
