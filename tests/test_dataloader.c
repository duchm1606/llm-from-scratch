#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dataloader.h"
#include "test_util.h"
#include "tokenizer.h"

/* Compares the next batch with the tensors the book prints. want_targets may
 * be NULL where the book only shows inputs. */
static void check_book_batch(DataLoader *dl, const char *name, size_t rows, size_t len,
                             const int *want_inputs, const int *want_targets)
{
    int inputs[64], targets[64];
    size_t got = dataloader_next(dl, inputs, targets);
    size_t size = rows * len * sizeof(int);
    if (got != rows || memcmp(inputs, want_inputs, size) != 0 ||
        (want_targets && memcmp(targets, want_targets, size) != 0)) {
        fprintf(stderr, "FAIL %s: batch differs from the book\n", name);
        failures++;
    }
}

/* The book's examples all use shuffle=False. */
static DataLoaderConfig in_order(size_t batch_size, size_t max_length, size_t stride,
                                 bool drop_last)
{
    return (DataLoaderConfig){.batch_size = batch_size,
                              .max_length = max_length,
                              .stride = stride,
                              .drop_last = drop_last};
}

static void check_book_examples(const int *ids, size_t n)
{
    /* Section 2.6, batch_size=1, max_length=4, stride=1. */
    DataLoader *dl = dataloader_create(ids, n, in_order(1, 4, 1, true));
    check_book_batch(dl, "2.6 first batch", 1, 4, (int[]){40, 367, 2885, 1464},
                     (int[]){367, 2885, 1464, 1807});
    check_book_batch(dl, "2.6 second batch", 1, 4, (int[]){367, 2885, 1464, 1807},
                     (int[]){2885, 1464, 1807, 3619});
    dataloader_free(dl);

    /* Section 2.6, batch_size=8, max_length=4, stride=4. */
    static const int inputs_8x4[] = {
        40,   367,  2885,  1464,
        1807, 3619, 402,   271,
        10899, 2138, 257,  7026,
        15632, 438, 2016,  257,
        922,  5891, 1576,  438,
        568,  340,  373,   645,
        1049, 5975, 284,   502,
        284,  3285, 326,   11,
    };
    static const int targets_8x4[] = {
        367,  2885, 1464,  1807,
        3619, 402,  271,   10899,
        2138, 257,  7026,  15632,
        438,  2016, 257,   922,
        5891, 1576, 438,   568,
        340,  373,  645,   1049,
        5975, 284,  502,   284,
        3285, 326,  11,    287,
    };
    dl = dataloader_create(ids, n, in_order(8, 4, 4, true));
    check_book_batch(dl, "2.6 batch of 8", 8, 4, inputs_8x4, targets_8x4);
    dataloader_free(dl);

    /* Exercise 2.2 (appendix C), which prints inputs only. */
    dl = dataloader_create(ids, n, in_order(4, 2, 2, true));
    check_book_batch(dl, "exercise 2.2, max_length=2", 4, 2,
                     (int[]){40, 367, 2885, 1464, 1807, 3619, 402, 271}, NULL);
    dataloader_free(dl);

    static const int inputs_4x8[] = {
        40,   367,  2885,  1464, 1807,  3619, 402,   271,
        2885, 1464, 1807,  3619, 402,   271,  10899, 2138,
        1807, 3619, 402,   271,  10899, 2138, 257,   7026,
        402,  271,  10899, 2138, 257,   7026, 15632, 438,
    };
    dl = dataloader_create(ids, n, in_order(4, 8, 2, true));
    check_book_batch(dl, "exercise 2.2, max_length=8", 4, 8, inputs_4x8, NULL);
    dataloader_free(dl);
}

/* Walks a whole unshuffled epoch and checks every row against ids directly:
 * row r of batch k is the window starting at (k * batch_size + r) * stride. */
static void check_epoch_in_order(const int *ids, size_t n, DataLoaderConfig cfg,
                                 size_t want_windows, size_t want_batches)
{
    DataLoader *dl = dataloader_create(ids, n, cfg);
    CHECK(dl != NULL);
    if (!dl)
        return;
    CHECK(dataloader_num_windows(dl) == want_windows);
    CHECK(dataloader_num_batches(dl) == want_batches);

    size_t len = cfg.max_length;
    int *inputs = malloc(cfg.batch_size * len * sizeof *inputs);
    int *targets = malloc(cfg.batch_size * len * sizeof *targets);
    size_t batches = 0, windows = 0, rows;
    while ((rows = dataloader_next(dl, inputs, targets)) > 0) {
        for (size_t r = 0; r < rows; r++, windows++) {
            size_t start = windows * cfg.stride;
            CHECK(start + len < n); /* the target never reads past the end */
            CHECK(memcmp(inputs + r * len, ids + start, len * sizeof *ids) == 0);
            CHECK(memcmp(targets + r * len, ids + start + 1, len * sizeof *ids) == 0);
        }
        batches++;
    }
    CHECK(batches == want_batches);
    CHECK(windows == (cfg.drop_last ? want_batches * cfg.batch_size : want_windows));
    CHECK(dataloader_next(dl, inputs, targets) == 0);

    /* A reset replays the same epoch. */
    size_t first_rows = want_windows >= cfg.batch_size ? cfg.batch_size
                        : cfg.drop_last                ? 0
                                                       : want_windows;
    dataloader_reset(dl);
    CHECK(dataloader_next(dl, inputs, targets) == first_rows);

    free(inputs);
    free(targets);
    dataloader_free(dl);
}

static void check_counts(const int *ids, size_t n)
{
    CHECK(n == 5145);
    /* len(range(0, 5145 - 4, 4)) == 1286 windows: 160 full batches of 8 plus 6 rows. */
    check_epoch_in_order(ids, n, in_order(8, 4, 4, false), 1286, 161);
    check_epoch_in_order(ids, n, in_order(8, 4, 4, true), 1286, 160);
    /* stride 1: every possible window, the last one ending at the final token. */
    check_epoch_in_order(ids, n, in_order(1, 4, 1, true), 5141, 5141);
    /* stride > max_length skips tokens between windows. */
    check_epoch_in_order(ids, n, in_order(3, 256, 300, false), 17, 6);

    /* Too short for a single (input, target) pair, and exactly long enough. */
    check_epoch_in_order(ids, 4, in_order(2, 4, 1, false), 0, 0);
    check_epoch_in_order(ids, 5, in_order(2, 4, 1, false), 1, 1);
    check_epoch_in_order(ids, 5, in_order(2, 4, 1, true), 1, 0);
}

enum { SYN_N = 103, SYN_LEN = 5, SYN_STRIDE = 3, SYN_BATCH = 4, SYN_WINDOWS = 33 };

/* With ids[i] == i, a row's first input token is its window's start offset.
 * Records the epoch's window order in starts[] and returns how many rows came
 * out, checking each row along the way. */
static size_t shuffled_epoch(DataLoader *dl, const int *ids, size_t *starts)
{
    int inputs[SYN_BATCH * SYN_LEN], targets[SYN_BATCH * SYN_LEN];
    size_t count = 0, rows;
    while ((rows = dataloader_next(dl, inputs, targets)) > 0) {
        for (size_t r = 0; r < rows; r++) {
            size_t start = (size_t)inputs[r * SYN_LEN];
            CHECK(start % SYN_STRIDE == 0 && start / SYN_STRIDE < SYN_WINDOWS);
            CHECK(memcmp(inputs + r * SYN_LEN, ids + start, sizeof(int) * SYN_LEN) == 0);
            CHECK(memcmp(targets + r * SYN_LEN, ids + start + 1, sizeof(int) * SYN_LEN) == 0);
            if (count < SYN_WINDOWS)
                starts[count] = start;
            count++;
        }
    }
    return count;
}

static int is_permutation_of_windows(const size_t *starts, size_t n)
{
    int seen[SYN_WINDOWS] = {0};
    for (size_t i = 0; i < n; i++) {
        size_t w = starts[i] / SYN_STRIDE;
        if (w >= SYN_WINDOWS || seen[w]++)
            return 0;
    }
    return 1;
}

static int is_in_order(const size_t *starts, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        if (starts[i] != i * SYN_STRIDE)
            return 0;
    }
    return 1;
}

static void check_shuffle(void)
{
    int ids[SYN_N];
    for (int i = 0; i < SYN_N; i++)
        ids[i] = i;

    DataLoaderConfig cfg = {
        .batch_size = SYN_BATCH, .max_length = SYN_LEN, .stride = SYN_STRIDE, .shuffle = true,
        .seed = 123};
    DataLoader *a = dataloader_create(ids, SYN_N, cfg);
    DataLoader *b = dataloader_create(ids, SYN_N, cfg);
    cfg.seed = 456;
    DataLoader *c = dataloader_create(ids, SYN_N, cfg);
    CHECK(a && b && c);
    if (!a || !b || !c)
        goto done;
    CHECK(dataloader_num_windows(a) == SYN_WINDOWS);

    size_t a1[SYN_WINDOWS], a2[SYN_WINDOWS], b1[SYN_WINDOWS], c1[SYN_WINDOWS];
    CHECK(shuffled_epoch(a, ids, a1) == SYN_WINDOWS);
    dataloader_reset(a);
    CHECK(shuffled_epoch(a, ids, a2) == SYN_WINDOWS);
    CHECK(shuffled_epoch(b, ids, b1) == SYN_WINDOWS);
    CHECK(shuffled_epoch(c, ids, c1) == SYN_WINDOWS);

    /* Every window exactly once per epoch, in an order that changes per
     * epoch and per seed but repeats for the same seed. */
    CHECK(is_permutation_of_windows(a1, SYN_WINDOWS));
    CHECK(is_permutation_of_windows(a2, SYN_WINDOWS));
    CHECK(!is_in_order(a1, SYN_WINDOWS));
    CHECK(memcmp(a1, a2, sizeof a1) != 0);
    CHECK(memcmp(a1, b1, sizeof a1) == 0);
    CHECK(memcmp(a1, c1, sizeof a1) != 0);

    /* A uniform shuffle leaves on average one window where the previous
     * epoch had it; an off-by-one Fisher-Yates (Sattolo's algorithm) never
     * does. Over 400 epochs expect about 400, so 200 is a safe floor. */
    size_t prev[SYN_WINDOWS], cur[SYN_WINDOWS], fixed = 0;
    memcpy(prev, a2, sizeof prev);
    for (int epoch = 0; epoch < 400; epoch++) {
        dataloader_reset(a);
        shuffled_epoch(a, ids, cur);
        for (size_t i = 0; i < SYN_WINDOWS; i++)
            fixed += cur[i] == prev[i];
        memcpy(prev, cur, sizeof prev);
    }
    CHECK(fixed > 200);

    /* drop_last with shuffling: 33 windows -> 8 full batches, 1 window unused. */
    cfg.drop_last = true;
    DataLoader *d = dataloader_create(ids, SYN_N, cfg);
    size_t d1[SYN_WINDOWS];
    CHECK(d && shuffled_epoch(d, ids, d1) == 32);
    dataloader_free(d);

done:
    dataloader_free(a);
    dataloader_free(b);
    dataloader_free(c);
}

static void check_invalid_config(void)
{
    int ids[8] = {0};
    CHECK(dataloader_create(ids, 8, in_order(0, 4, 1, false)) == NULL);
    CHECK(dataloader_create(ids, 8, in_order(1, 0, 1, false)) == NULL);
    CHECK(dataloader_create(ids, 8, in_order(1, 4, 0, false)) == NULL);
}

int main(void)
{
    Tokenizer *tok = tokenizer_load(ENCODER_JSON, VOCAB_BPE);
    size_t len, n = 0;
    char *text = read_text(THE_VERDICT, &len);
    int *ids = tok && text ? tokenizer_encode(tok, text, len, false, &n) : NULL;
    if (!ids) {
        fprintf(stderr, "test_dataloader: cannot tokenize %s (run `make data`)\n", THE_VERDICT);
        return 1;
    }

    check_book_examples(ids, n);
    check_counts(ids, n);
    check_shuffle();
    check_invalid_config();

    free(ids);
    free(text);
    tokenizer_free(tok);
    return test_report("test_dataloader");
}
