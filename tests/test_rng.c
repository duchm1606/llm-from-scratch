#include <stdint.h>
#include <stdio.h>

#include "rng.h"
#include "test_util.h"

int main(void)
{
    /* SplitMix64 reference outputs for seed 1234567. */
    static const uint64_t reference[] = {
        6457827717110365317ULL, 3203168211198807973ULL, 9817491932198370423ULL,
        4593380528125082431ULL, 16408922859458223821ULL,
    };
    Rng rng;
    rng_seed(&rng, 1234567);
    for (size_t i = 0; i < sizeof reference / sizeof reference[0]; i++)
        CHECK(rng_next(&rng) == reference[i]);

    /* Results stay in range, including when most 64-bit values are rejected. */
    static const uint64_t bounds[] = {1, 2, 3, 7, 1000, UINT64_MAX / 2 + 2, UINT64_MAX};
    rng_seed(&rng, 42);
    for (size_t b = 0; b < sizeof bounds / sizeof bounds[0]; b++) {
        int in_range = 1;
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

    return test_report("test_rng");
}
