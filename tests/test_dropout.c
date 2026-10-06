#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dropout.h"
#include "rng.h"
#include "test_util.h"

/* Expectations come from PyTorch 2.14 on CPU without MKL (Apple silicon). */

static void fill(float *x, size_t n, float value)
{
    for (size_t i = 0; i < n; i++)
        x[i] = value;
}

/* Section 3.5.2: torch.manual_seed(123); nn.Dropout(0.5)(torch.ones(6, 6)). */
static void check_book_example(void)
{
    static const float want[6 * 6] = {
        2, 2, 0, 2, 2, 0,
        0, 0, 0, 2, 0, 2,
        2, 2, 2, 2, 0, 2,
        0, 2, 2, 0, 0, 2,
        0, 2, 0, 2, 0, 2,
        0, 2, 2, 2, 2, 0,
    };
    float x[6 * 6];
    fill(x, 6 * 6, 1);
    Rng rng;
    rng_seed(&rng, 123);
    dropout_forward(x, 6 * 6, 0.5, &rng);
    CHECK(memcmp(x, want, sizeof want) == 0);
}

enum { N_LARGE = 100000 };

/* torch.manual_seed(123); nn.Dropout(0.1), the rate GPT training uses. */
static void check_rate_0_1(void)
{
    Rng rng;

    /* Over torch.ones(64), bit i of the mask is whether value i survives. */
    float ones[64];
    fill(ones, 64, 1);
    rng_seed(&rng, 123);
    dropout_forward(ones, 64, 0.1, &rng);
    uint64_t mask = 0;
    bool scaled = true;
    for (size_t i = 0; i < 64; i++) {
        if (ones[i] != 0)
            mask |= (uint64_t)1 << i;
        scaled &= ones[i] == 0 || ones[i] == 1.11111116f;
    }
    CHECK(mask == 0xdbbedffffbfffeffull);
    CHECK(scaled);

    /* Over torch.arange(64) / 64 * 3 - 1, values where x * (1/0.9f) and
     * x / 0.9f differ show which one PyTorch computes. */
    float x[64];
    for (size_t i = 0; i < 64; i++)
        x[i] = (float)i / 64 * 3 - 1;
    rng_seed(&rng, 123);
    dropout_forward(x, 64, 0.1, &rng);
    CHECK(x[2] == -1.00694454f);
    CHECK(x[6] == -0.798611164f);
    CHECK(x[9] == -0.642361164f);
    CHECK(x[14] == -0.381944478f);

    /* torch.ones(100000) keeps exactly 89941. */
    float *big = malloc(sizeof(float) * N_LARGE);
    if (!big) {
        fprintf(stderr, "FAIL out of memory\n");
        failures++;
        return;
    }
    fill(big, N_LARGE, 1);
    rng_seed(&rng, 123);
    dropout_forward(big, N_LARGE, 0.1, &rng);
    size_t kept = 0;
    for (size_t i = 0; i < N_LARGE; i++)
        kept += big[i] != 0;
    CHECK(kept == 89941);
    free(big);
}

/* p = 0 leaves the input alone and p = 1 zeroes it; neither draws, so the
 * generator continues as if dropout had not run. */
static void check_no_draws(void)
{
    Rng rng, fresh;
    rng_seed(&fresh, 123);
    uint32_t first = rng_next_u32(&fresh);

    float x[10], ones[10];
    fill(x, 10, 1);
    fill(ones, 10, 1);
    rng_seed(&rng, 123);
    dropout_forward(x, 10, 0, &rng);
    CHECK(memcmp(x, ones, sizeof x) == 0);
    CHECK(rng_next_u32(&rng) == first);

    rng_seed(&rng, 123);
    dropout_forward(x, 10, 1, &rng);
    bool zero = true;
    for (size_t i = 0; i < 10; i++)
        zero &= x[i] == 0;
    CHECK(zero);
    CHECK(rng_next_u32(&rng) == first);
}

int main(void)
{
    check_book_example();
    check_rate_0_1();
    check_no_draws();
    return test_report("test_dropout");
}
