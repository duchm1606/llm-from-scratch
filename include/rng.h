#ifndef LLM_RNG_H
#define LLM_RNG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * Seeded random number generator that reproduces PyTorch's CPU generator, so
 * the random numbers printed in the book can serve as test vectors:
 *
 *   torch.manual_seed(s)          rng_seed(&rng, s)
 *   torch.rand(n)                 rng_uniform(&rng, out, n, 0, 1)
 *   tensor.uniform_(lo, hi)       rng_uniform(&rng, out, n, lo, hi)
 *   torch.randn(n), normal_()     rng_normal(&rng, out, n, 0, 1)
 *   tensor.bernoulli_(p)          rng_bernoulli(&rng, p), once per element
 *
 * The same sequence of calls gives the same float32 numbers as in PyTorch, up
 * to last-bit differences between math libraries. Underneath is the 32-bit
 * Mersenne Twister (MT19937).
 */
#define RNG_MT_SIZE 624

typedef struct {
    uint32_t mt[RNG_MT_SIZE];
    size_t next;         /* index of the next word of mt to output */
    bool has_normal;     /* PyTorch keeps the second Box-Muller sample of a */
    double next_normal;  /* scalar normal draw for the next call */
} Rng;

/* torch.manual_seed(seed). Only the low 32 bits of seed are used, as in PyTorch. */
void rng_seed(Rng *rng, uint64_t seed);

/* One MT19937 output (PyTorch's generator->random()). */
uint32_t rng_next_u32(Rng *rng);

/* Two outputs, the first in the high half (PyTorch's generator->random64()). */
uint64_t rng_next_u64(Rng *rng);

/* Uniform over [0, n), without modulo bias. n must be > 0. For shuffling;
 * PyTorch has no direct equivalent. */
uint64_t rng_below(Rng *rng, uint64_t n);

/* Fills out with uniform floats in [lo, hi), like tensor.uniform_(lo, hi). */
void rng_uniform(Rng *rng, float *out, size_t n, float lo, float hi);

/* Fills out with normal samples, like tensor.normal_(mean, std). */
void rng_normal(Rng *rng, float *out, size_t n, float mean, float std);

/* True with probability p: one element of tensor.bernoulli_(p), which draws
 * a double uniform per element and compares it with p. That is PyTorch's
 * path without MKL, as on Apple silicon, and the one the book's dropout
 * numbers match; x86 builds with MKL draw differently, which is why the book
 * warns that dropout output varies by OS. */
bool rng_bernoulli(Rng *rng, double p);

#endif
