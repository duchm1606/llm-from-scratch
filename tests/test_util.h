#ifndef LLM_TEST_UTIL_H
#define LLM_TEST_UTIL_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define ENCODER_JSON "data/gpt2/encoder.json"
#define VOCAB_BPE "data/gpt2/vocab.bpe"
#define THE_VERDICT "data/the-verdict.txt"

static int failures;

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                              \
        }                                                                            \
    } while (0)

/* Reads a whole file into a malloc'd buffer (not NUL-terminated), or NULL. */
static inline char *read_text(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    char *buf = NULL;
    size_t cap = 0, n = 0, got;
    do {
        if (n == cap) {
            cap = cap ? cap * 2 : 1 << 16;
            char *grown = realloc(buf, cap);
            if (!grown) {
                free(buf);
                fclose(f);
                return NULL;
            }
            buf = grown;
        }
        got = fread(buf + n, 1, cap - n, f);
        n += got;
    } while (got > 0);
    fclose(f);
    *len = n;
    return buf;
}

/* Whether got[i] is within tol of want[i] for all i, relative to |want[i]|
 * once that exceeds 1. */
static inline bool floats_close(const float *got, const float *want, size_t n, float tol)
{
    for (size_t i = 0; i < n; i++) {
        float scale = fabsf(want[i]) > 1 ? fabsf(want[i]) : 1;
        if (!(fabsf(got[i] - want[i]) <= tol * scale))
            return false;
    }
    return true;
}

/* Prints the summary line and returns the process exit code. */
static inline int test_report(const char *name)
{
    if (failures) {
        fprintf(stderr, "%s: %d failure(s)\n", name, failures);
        return 1;
    }
    printf("%s: all checks passed\n", name);
    return 0;
}

#endif
