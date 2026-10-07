#include <stdio.h>
#include <stdlib.h>

#include "feedforward.h"
#include "rng.h"
#include "test_util.h"

/* Expectations come from PyTorch 2.14 on CPU, running the book's
 * FeedForward class; the book prints only the output shape. */
#define TOL 1e-6f

enum { GUARD = 16 };

#define SENTINEL -1234.5f

/* feed_forward_forward in scratch followed by GUARD sentinels, checking
 * that nothing wrote past the size it reports. */
static void run_feed_forward(float *out, const float *x, size_t n, const FeedForward *ff)
{
    size_t size = feed_forward_scratch_size(ff);
    float *scratch = malloc(sizeof(float) * (size + GUARD));
    if (!scratch) {
        fprintf(stderr, "FAIL out of memory\n");
        failures++;
        return;
    }
    for (size_t i = 0; i < GUARD; i++)
        scratch[size + i] = SENTINEL;
    feed_forward_forward(out, x, n, ff, scratch);
    bool intact = true;
    for (size_t i = 0; i < GUARD; i++)
        intact &= scratch[size + i] == SENTINEL;
    CHECK(intact);
    free(scratch);
}

/* torch.manual_seed(123); FeedForward({"emb_dim": 4}); x = torch.rand(1, 2,
 * 4): small enough to compare every output. */
static void check_small(void)
{
    enum { C = 4, H = 4 * C, N = 2 };
    float w_in[H * C], b_in[H], w_out[C * H], b_out[C], x[N * C], out[N * C];
    FeedForward ff = {
        .emb_dim = C, .hidden_dim = H,
        .w_in = w_in, .b_in = b_in, .w_out = w_out, .b_out = b_out,
    };
    Rng rng;
    rng_seed(&rng, 123);
    feed_forward_init(&ff, &rng);
    rng_uniform(&rng, x, N * C, 0, 1);
    CHECK(floats_close(x,
                       (float[]){0.129076898f, 0.124174893f, 0.174561501f, 0.330177605f,
                                 0.537008762f, 0.844305813f, 0.693669677f, 0.883059382f},
                       N * C, TOL));
    run_feed_forward(out, x, N, &ff);
    CHECK(floats_close(out,
                       (float[]){-0.0328679532f, -0.253316998f, 0.0935623646f, -0.107867017f,
                                 -0.00538514555f, -0.329127371f, 0.245998472f, -0.261282027f},
                       N * C, TOL));
}

/* Section 4.3 with a seed: torch.manual_seed(123); ffn =
 * FeedForward(GPT_CONFIG_124M); x = torch.rand(2, 3, 768). */
static void check_gpt2_size(void)
{
    enum { C = 768, H = 4 * C, N = 2 * 3 };
    /* 4,722,432 parameters: two weights of 768 x 3072 and the two biases. */
    float *params = malloc(sizeof(float) * (2 * H * C + H + C));
    float *x = malloc(sizeof(float) * N * C);
    float *out = malloc(sizeof(float) * N * C);
    if (!params || !x || !out) {
        fprintf(stderr, "FAIL out of memory\n");
        failures++;
        goto done;
    }
    FeedForward ff = {
        .emb_dim = C, .hidden_dim = H,
        .w_in = params, .b_in = params + H * C,
        .w_out = params + H * C + H, .b_out = params + 2 * H * C + H,
    };
    Rng rng;
    rng_seed(&rng, 123);
    feed_forward_init(&ff, &rng);
    CHECK(floats_close(ff.w_in, (float[]){-0.0147143528f, 0.00119527976f, -0.017921621f}, 3, TOL));
    CHECK(floats_close(&ff.b_in[H - 3], (float[]){0.0297883227f, 0.00706425076f, 0.0324309878f},
                       3, TOL));
    CHECK(floats_close(ff.w_out, (float[]){-0.002169393f, -0.00764360465f, -0.0111561473f}, 3,
                       TOL));
    CHECK(floats_close(&ff.b_out[C - 3],
                       (float[]){-0.0135413641f, -0.010913617f, -0.00988701079f}, 3, TOL));

    rng_uniform(&rng, x, N * C, 0, 1);
    run_feed_forward(out, x, N, &ff);
    CHECK(floats_close(out, (float[]){0.0138571328f, 0.013552363f, -0.140250191f}, 3, TOL));
    /* Token [1][0], from column 384 */
    CHECK(floats_close(&out[3 * C + 384], (float[]){0.0403278135f, -0.0987298638f, -0.098621048f},
                       3, TOL));
    CHECK(floats_close(&out[N * C - 3], (float[]){-0.0481129847f, -0.095178321f, -0.118922748f},
                       3, TOL));

done:
    free(params);
    free(x);
    free(out);
}

int main(void)
{
    check_small();
    check_gpt2_size();
    return test_report("test_feedforward");
}
