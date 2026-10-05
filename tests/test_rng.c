#include <stdint.h>
#include <stdio.h>

#include "rng.h"
#include "test_util.h"

/* Expected floats come from PyTorch 2.14 on CPU. The math library may change
 * the last bit, so floats compare within TOL. */
#define TOL 1e-6f
#define LEN(a) (sizeof(a) / sizeof((a)[0]))

/* Draws LEN(want) normals and compares them. */
#define CHECK_NORMAL(rng, want)                              \
    do {                                                     \
        float got_[LEN(want)];                               \
        rng_normal(rng, got_, LEN(want), 0, 1);              \
        CHECK(floats_close(got_, want, LEN(want), TOL));     \
    } while (0)

static void check_mt19937(void)
{
    /* std::mt19937 with its default seed; the C++ standard fixes the 10000th
     * output at 4123659995. */
    static const uint32_t first[] = {3499211612u, 581869302u, 3890346734u, 3586334585u, 545404204u};
    Rng rng;
    rng_seed(&rng, 5489);
    for (size_t i = 0; i < LEN(first); i++)
        CHECK(rng_next_u32(&rng) == first[i]);
    for (int i = 6; i < 10000; i++)
        rng_next_u32(&rng);
    CHECK(rng_next_u32(&rng) == 4123659995u);
}

static void check_torch_compat(void)
{
    Rng rng;

    /* torch.manual_seed(123); torch.rand(5) */
    static const float rand5[] = {0.296111941f, 0.516562283f, 0.251670718f, 0.68855679f,
                                  0.0739724636f};
    float got[5];
    rng_seed(&rng, 123);
    rng_uniform(&rng, got, 5, 0, 1);
    CHECK(floats_close(got, rand5, 5, TOL));

    /* torch.randn(3): fewer than 16 values take the double-precision path. */
    static const float randn3[] = {-0.111467116f, 0.120362945f, -0.369634509f};
    rng_seed(&rng, 123);
    CHECK_NORMAL(&rng, randn3);

    /* torch.randn(15): the largest size on that path. */
    static const float randn15[] = {
        -0.111467116f, 0.120362945f, -0.369634509f, -0.240417972f, -1.19692433f,
        0.20926936f,   -0.972355008f, -0.755045474f, 0.323902756f, -0.108522631f,
        0.210331157f,  -0.390842825f, 0.234973416f,  0.665260434f, 0.352820754f,
    };
    rng_seed(&rng, 123);
    CHECK_NORMAL(&rng, randn15);

    /* torch.randn(17): 16 or more take the float path, which redraws the last
     * 16 values when n is not a multiple of 16. */
    static const float randn17[] = {
        0.337370157f, -1.02047253f,  -0.168956161f, 0.917764008f, 1.5809691f,  1.30103993f,
        1.27530122f,  -0.200953096f, -0.160563797f, -1.57230389f, 0.966571569f, -1.14814448f,
        -1.15886867f, 0.325471014f,  -0.63150543f,  -2.83999395f, -0.784853339f,
    };
    rng_seed(&rng, 123);
    CHECK_NORMAL(&rng, randn17);

    /* A run of calls shares one generator. randn(3) leaves the second sample
     * of its last pair cached, and randn(1) returns it. */
    static const float mix_rand2a[] = {0.53492254f, 0.198803186f};
    static const float mix_randn3[] = {-0.295246035f, 1.96854782f, -1.25140011f};
    static const float mix_randn1[] = {-1.88406754f};
    static const float mix_rand2b[] = {0.854943156f, 0.550935209f};
    static const float mix_randn16[] = {
        0.609473467f,  -0.581133723f, 0.0490438677f, -0.320476919f, -1.58266461f, -0.324566692f,
        1.92636728f,   -0.330011725f, -0.551940501f, 0.352749139f,  -1.08420193f, -0.887490094f,
        -0.209340423f, -0.215341449f, -1.81572962f,  -0.345241964f,
    };
    rng_seed(&rng, 7);
    rng_uniform(&rng, got, 2, 0, 1);
    CHECK(floats_close(got, mix_rand2a, 2, TOL));
    CHECK_NORMAL(&rng, mix_randn3);
    CHECK_NORMAL(&rng, mix_randn1);
    rng_uniform(&rng, got, 2, 0, 1);
    CHECK(floats_close(got, mix_rand2b, 2, TOL));
    CHECK_NORMAL(&rng, mix_randn16);

    /* Reseeding drops the cached sample: randn(1) restarts at randn3[0]. */
    rng_seed(&rng, 123);
    rng_normal(&rng, got, 1, 0, 1);
    rng_seed(&rng, 123);
    rng_normal(&rng, got, 1, 0, 1);
    CHECK(floats_close(got, randn3, 1, TOL));

    /* uniform_(-0.5, 0.25), normal_(2, 0.5) on 4 values, normal_(-1, 3) on 20. */
    static const float unif[] = {0.122688904f, -0.405416816f, 0.180602267f, 0.114945278f};
    rng_seed(&rng, 5);
    rng_uniform(&rng, got, 4, -0.5f, 0.25f);
    CHECK(floats_close(got, unif, 4, TOL));

    static const float normal_small[] = {1.75660968f, 1.69808924f, 1.72095191f, 2.33376217f};
    static const float normal_big[] = {-0.225245088f, -6.64997864f, 2.84027624f, 2.39133859f};
    float big[20];
    rng_seed(&rng, 5);
    rng_normal(&rng, got, 4, 2.0f, 0.5f);
    CHECK(floats_close(got, normal_small, 4, TOL));
    rng_normal(&rng, big, 20, -1.0f, 3.0f);
    CHECK(floats_close(big, normal_big, 4, TOL));
}

static void check_below(void)
{
    /* Results stay in range, including when most 64-bit values are rejected. */
    static const uint64_t bounds[] = {1, 2, 3, 7, 1000, UINT64_MAX / 2 + 2, UINT64_MAX};
    Rng rng;
    rng_seed(&rng, 42);
    for (size_t b = 0; b < LEN(bounds); b++) {
        bool in_range = true;
        for (int i = 0; i < 1000; i++)
            in_range &= rng_below(&rng, bounds[b]) < bounds[b];
        CHECK(in_range);
    }

    /* Six faces, 60000 rolls: each count is ~10000 with a std of ~91. */
    long counts[6] = {0};
    rng_seed(&rng, 7);
    for (int i = 0; i < 60000; i++)
        counts[rng_below(&rng, 6)]++;
    for (int f = 0; f < 6; f++)
        CHECK(counts[f] > 9500 && counts[f] < 10500);
}

int main(void)
{
    check_mt19937();
    check_torch_compat();
    check_below();
    return test_report("test_rng");
}
