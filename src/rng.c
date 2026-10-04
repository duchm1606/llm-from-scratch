#include "rng.h"

void rng_seed(Rng *rng, uint64_t seed)
{
    rng->state = seed;
}

uint64_t rng_next(Rng *rng)
{
    uint64_t z = (rng->state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

uint64_t rng_below(Rng *rng, uint64_t n)
{
    /* Drop the 2^64 mod n smallest values so every remainder is equally likely. */
    uint64_t threshold = (0 - n) % n;
    for (;;) {
        uint64_t x = rng_next(rng);
        if (x >= threshold)
            return x % n;
    }
}
