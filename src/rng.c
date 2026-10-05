#include "rng.h"

#include <math.h>

#define PI 3.14159265358979323846

void rng_seed(Rng *rng, uint64_t seed)
{
    rng->mt[0] = (uint32_t)seed;
    for (uint32_t i = 1; i < RNG_MT_SIZE; i++)
        rng->mt[i] = 1812433253u * (rng->mt[i - 1] ^ (rng->mt[i - 1] >> 30)) + i;
    rng->next = RNG_MT_SIZE;
    rng->has_normal = false;
}

/* Regenerates all 624 words of state at once (the MT19937 "twist"). */
static void twist(Rng *rng)
{
    for (size_t i = 0; i < RNG_MT_SIZE; i++) {
        uint32_t y = (rng->mt[i] & 0x80000000u) | (rng->mt[(i + 1) % RNG_MT_SIZE] & 0x7fffffffu);
        rng->mt[i] = rng->mt[(i + 397) % RNG_MT_SIZE] ^ (y >> 1) ^ (y & 1 ? 0x9908b0dfu : 0);
    }
    rng->next = 0;
}

uint32_t rng_next_u32(Rng *rng)
{
    if (rng->next >= RNG_MT_SIZE)
        twist(rng);
    uint32_t y = rng->mt[rng->next++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

uint64_t rng_next_u64(Rng *rng)
{
    uint64_t hi = rng_next_u32(rng);
    return hi << 32 | rng_next_u32(rng);
}

uint64_t rng_below(Rng *rng, uint64_t n)
{
    /* Drop the 2^64 mod n smallest values so every remainder is equally likely. */
    uint64_t threshold = (0 - n) % n;
    for (;;) {
        uint64_t x = rng_next_u64(rng);
        if (x >= threshold)
            return x % n;
    }
}

/* PyTorch's uniform floats keep 24 random bits (a float's precision) and do
 * the scaling in double; doubles keep 53 bits. */
static double uniform_float_bits(Rng *rng)
{
    return (rng_next_u32(rng) & 0xFFFFFFu) * 0x1p-24;
}

static double uniform_double(Rng *rng)
{
    return (rng_next_u64(rng) & 0x1FFFFFFFFFFFFFull) * 0x1p-53;
}

void rng_uniform(Rng *rng, float *out, size_t n, float lo, float hi)
{
    float range = hi - lo;
    for (size_t i = 0; i < n; i++)
        out[i] = (float)(uniform_float_bits(rng) * range + lo);
}

/* Box-Muller on 16 uniforms in place: pairs (d[j], d[j + 8]) for j < 8. */
static void normal_fill_16(float *d, float mean, float std)
{
    for (int j = 0; j < 8; j++) {
        float u1 = 1 - d[j]; /* [0, 1) -> (0, 1] so the log is finite */
        float u2 = d[j + 8];
        float radius = sqrtf(-2 * logf(u1));
        float theta = (float)(2.0f * PI * u2);
        d[j] = radius * cosf(theta) * std + mean;
        d[j + 8] = radius * sinf(theta) * std + mean;
    }
}

/*
 * PyTorch has two code paths, and matching it means following both:
 *  - 16 or more values: fill with uniforms, transform blocks of 16 in float,
 *    and if n is not a multiple of 16, redraw and transform the last 16 again.
 *  - fewer: one double-precision Box-Muller pair per two values, keeping the
 *    unused second sample in the generator for the next call.
 */
void rng_normal(Rng *rng, float *out, size_t n, float mean, float std)
{
    if (n >= 16) {
        rng_uniform(rng, out, n, 0, 1);
        for (size_t i = 0; i + 16 <= n; i += 16)
            normal_fill_16(out + i, mean, std);
        if (n % 16 != 0) {
            rng_uniform(rng, out + n - 16, 16, 0, 1);
            normal_fill_16(out + n - 16, mean, std);
        }
        return;
    }

    for (size_t i = 0; i < n; i++) {
        double z;
        if (rng->has_normal) {
            z = rng->next_normal;
            rng->has_normal = false;
        } else {
            double u1 = uniform_double(rng);
            double u2 = uniform_double(rng);
            double radius = sqrt(-2.0 * log1p(-u2));
            double theta = 2.0 * PI * u1;
            rng->next_normal = radius * sin(theta);
            rng->has_normal = true;
            z = radius * cos(theta);
        }
        out[i] = (float)(z * std + mean);
    }
}
