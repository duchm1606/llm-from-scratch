#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizer.h"

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

/* Expected ids come from tiktoken.get_encoding("gpt2"); -1 ends each list. */
typedef struct {
    const char *name;
    const char *text;
    bool allow_special;
    int ids[32];
} Case;

static const Case cases[] = {
    {"book 2.5", "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.",
     true, {15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 286,
            617, 34680, 27271, 13, -1}},
    {"exercise 2.1", "Akwirw ier", false, {33901, 86, 343, 86, 220, 959, -1}},
    {"whitespace runs", "a   b\n\nc \t d  ", false,
     {64, 220, 220, 275, 198, 198, 66, 220, 197, 288, 220, 220, -1}},
    {"contractions", "It's O'Sullivan's, they'LL we're I'M don't 'twas", false,
     {1026, 338, 440, 6, 47572, 338, 11, 484, 6, 3069, 356, 821, 314, 6, 44, 836, 470, 705, 4246,
      292, -1}},
    {"numbers", "In 2024, pi = 3.14159; 1,000,000!", false,
     {818, 48609, 11, 31028, 796, 513, 13, 1415, 19707, 26, 352, 11, 830, 11, 830, 0, -1}},
    {"vietnamese", "Xin chào thế giới!", false,
     {55, 259, 442, 24247, 78, 294, 157, 118, 123, 308, 72, 157, 119, 249, 72, 0, -1}},
    {"emoji", "👍🏽 ok", false, {41840, 235, 8582, 237, 121, 12876, -1}},
    {"eot as plain text", "a<|endoftext|>b", false, {64, 27, 91, 437, 1659, 5239, 91, 29, 65, -1}},
    {"eot as special token", "a<|endoftext|>b", true, {64, 50256, 65, -1}},
    {"empty", "", false, {-1}},
};

static void print_ids(const char *label, const int *ids, size_t n)
{
    fprintf(stderr, "  %s:", label);
    for (size_t i = 0; i < n; i++)
        fprintf(stderr, " %d", ids[i]);
    fprintf(stderr, "\n");
}

/* Checks encode(text) against want (unless NULL) and that decode restores text. */
static void check_case(const Tokenizer *tok, const char *name, const char *text, size_t len,
                       bool allow_special, const int *want)
{
    size_t n;
    int *ids = tokenizer_encode(tok, text, len, allow_special, &n);
    if (!ids) {
        fprintf(stderr, "FAIL %s: encode returned NULL\n", name);
        failures++;
        return;
    }

    if (want) {
        size_t n_want = 0;
        while (want[n_want] >= 0)
            n_want++;
        if (n != n_want || memcmp(ids, want, n * sizeof *ids) != 0) {
            fprintf(stderr, "FAIL %s: wrong ids\n", name);
            print_ids("want", want, n_want);
            print_ids("got ", ids, n);
            failures++;
        }
    }

    size_t out_len;
    char *out = tokenizer_decode(tok, ids, n, &out_len);
    if (!out || out_len != len || memcmp(out, text, len) != 0) {
        fprintf(stderr, "FAIL %s: decode does not restore the input\n", name);
        failures++;
    }
    free(out);
    free(ids);
}

static char *read_text(const char *path, size_t *len)
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

/* Section 2.6: the story is 5,145 tokens, and the sliding-window example
 * starts at token 50 with " and established himself in a". */
static void check_the_verdict(const Tokenizer *tok)
{
    size_t len;
    char *text = read_text(THE_VERDICT, &len);
    if (!text) {
        fprintf(stderr, "FAIL cannot read %s (run `make data`)\n", THE_VERDICT);
        failures++;
        return;
    }

    size_t n;
    int *ids = tokenizer_encode(tok, text, len, false, &n);
    CHECK(ids != NULL);
    if (ids) {
        CHECK(n == 5145);
        static const int window[] = {290, 4920, 2241, 287, 257};
        CHECK(n >= 55 && memcmp(ids + 50, window, sizeof window) == 0);
    }
    check_case(tok, "the-verdict round trip", text, len, false, NULL);
    free(ids);
    free(text);
}

int main(void)
{
    Tokenizer *tok = tokenizer_load(ENCODER_JSON, VOCAB_BPE);
    if (!tok) {
        fprintf(stderr, "test_tokenizer: cannot load the GPT-2 vocabulary (run `make data`)\n");
        return 1;
    }

    CHECK(tokenizer_vocab_size(tok) == 50257);
    CHECK(tokenizer_eot_id(tok) == 50256);

    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        const Case *c = &cases[i];
        check_case(tok, c->name, c->text, strlen(c->text), c->allow_special, c->ids);
    }

    /* Bytes that are not UTF-8 still survive the round trip. */
    static const char invalid[] = "\xff\xfe abc \xc0\xaf \xf0\x9f\x98";
    check_case(tok, "invalid utf-8", invalid, sizeof invalid - 1, false, NULL);

    check_the_verdict(tok);

    static const int out_of_range[] = {50257};
    static const int negative[] = {-1};
    CHECK(tokenizer_decode(tok, out_of_range, 1, NULL) == NULL);
    CHECK(tokenizer_decode(tok, negative, 1, NULL) == NULL);

    tokenizer_free(tok);
    if (failures) {
        fprintf(stderr, "test_tokenizer: %d failure(s)\n", failures);
        return 1;
    }
    printf("test_tokenizer: all checks passed\n");
    return 0;
}
