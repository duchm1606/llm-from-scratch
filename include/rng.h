#ifndef LLM_RNG_H
#define LLM_RNG_H

#include <stdint.h>

/*
 * Small seeded random number generator (SplitMix64). The same seed always
 * gives the same sequence, which plays the role of torch.manual_seed(). It
 * does not reproduce PyTorch's numbers.
 */
typedef struct {
    uint64_t state;
} Rng;

void rng_seed(Rng *rng, uint64_t seed);

/* Uniform over all 64-bit values. */
uint64_t rng_next(Rng *rng);

/* Uniform over [0, n), without modulo bias. n must be > 0. */
uint64_t rng_below(Rng *rng, uint64_t n);

#endif
