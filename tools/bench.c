/*
 * Benchmark of generation on GPT-2 124M, the chat's weights, with and
 * without the KV cache. Every timed run is a CSV row on stdout, after
 * comment lines naming the build and the machine; the summary goes to
 * stderr. docs/benchmark.md describes the method and the results.
 *
 * Phases, each timed on the first n tokens of the-verdict.txt:
 *   load     reading the weights file (n = 0)
 *   prefill  gpt_forward_cached on n tokens into an empty cache: the time
 *            to the first token of an n-token prompt
 *   decode   gpt_forward_cached on token n with n - 1 cached: one step of
 *            generation with the cache, at context n
 *   nocache  generate_text_simple for one token after n: one step of
 *            generation without the cache, at context n
 *   generate_cached, generate_nocache
 *            GENERATED greedy tokens after a PROMPT-token prompt, n = PROMPT
 *   drift    decode at DRIFT_N again at the end, to catch throttling
 *
 * Greedy tokens must match between the cached and uncached runs, and the
 * prefill and decode logits must be identical, or the bench fails.
 *
 * Usage: bench [-m weights.bin] > results.csv
 * Run from the repository root after `make data weights bench`.
 */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <time.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#include "gpt.h"
#include "kv_cache.h"
#include "tokenizer.h"

#define DEFAULT_WEIGHTS "data/gpt2_alpaca.bin"
#define ENCODER_JSON "data/gpt2/encoder.json"
#define VOCAB_BPE "data/gpt2/vocab.bpe"
#define TEXT "data/the-verdict.txt"

#ifndef BENCH_COMMIT
#define BENCH_COMMIT "unknown"
#endif
#ifndef BENCH_CFLAGS
#define BENCH_CFLAGS "unknown"
#endif

enum { VOCAB_SIZE = 50260, PROMPT = 64, GENERATED = 32, DRIFT_N = 32, MAX_REPS = 20 };

/* Context lengths of the sweep, up to the model's 1024. */
static const size_t SWEEP[] = {32, 64, 128, 256, 512, 1024};
#define N_SWEEP (sizeof SWEEP / sizeof SWEEP[0])

/* Untimed warm-up runs, then timed runs, of one measurement. A decode step
 * is short, so it gets more runs. */
typedef struct {
    size_t warmup, reps;
} Plan;

static const Plan LOAD = {1, 5}, PREFILL = {1, 5}, DECODE = {3, 20}, NOCACHE = {1, 5},
                  GENERATE = {0, 1};

typedef struct {
    const char *phase;
    size_t n;
    size_t reps;
    double seconds[MAX_REPS];
} Measurement;

static Measurement results[32];
static size_t n_results;

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static Measurement *begin(const char *phase, size_t n)
{
    Measurement *m = &results[n_results++];
    *m = (Measurement){.phase = phase, .n = n};
    fprintf(stderr, "%-16s n=%-5zu", phase, n);
    return m;
}

/* Prints run r of a measurement, keeping it unless r is a warm-up. */
static void record(Measurement *m, const Plan *plan, size_t r, double seconds)
{
    if (r < plan->warmup)
        return;
    printf("%s,%zu,%zu,%.9f\n", m->phase, m->n, m->reps, seconds);
    fflush(stdout);
    m->seconds[m->reps++] = seconds;
    fputc('.', stderr);
    if (r + 1 == plan->warmup + plan->reps)
        fputc('\n', stderr);
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(const Measurement *m)
{
    double s[MAX_REPS];
    memcpy(s, m->seconds, sizeof(double) * m->reps);
    qsort(s, m->reps, sizeof(double), cmp_double);
    return m->reps % 2 ? s[m->reps / 2] : (s[m->reps / 2 - 1] + s[m->reps / 2]) / 2;
}

static const Measurement *find(const char *phase, size_t n)
{
    for (size_t i = 0; i < n_results; i++)
        if (strcmp(results[i].phase, phase) == 0 && results[i].n == n)
            return &results[i];
    return NULL;
}

/* Peak resident memory of the process so far, in bytes. */
static double peak_rss(void)
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    return (double)ru.ru_maxrss;
#else
    return 1024.0 * (double)ru.ru_maxrss;
#endif
}

static void print_metadata(const char *weights)
{
    char date[32];
    time_t t = time(NULL);
    strftime(date, sizeof date, "%Y-%m-%dT%H:%M:%S%z", localtime(&t));
    struct utsname u;
    uname(&u);
    printf("# date: %s\n# commit: %s\n# compiler: %s\n# cflags: %s\n# os: %s %s %s\n", date,
           BENCH_COMMIT, __VERSION__, BENCH_CFLAGS, u.sysname, u.release, u.machine);
#ifdef __APPLE__
    const char *keys[] = {"hw.model", "machdep.cpu.brand_string", "kern.osproductversion"};
    for (size_t i = 0; i < sizeof keys / sizeof keys[0]; i++) {
        char value[256];
        size_t len = sizeof value;
        if (sysctlbyname(keys[i], value, &len, NULL, 0) == 0)
            printf("# %s: %s\n", keys[i], value);
    }
#endif
    printf("# weights: %s\n# threads: 1\nphase,n,rep,seconds\n", weights);
}

/* Reads the exported weights into params, n floats; false after printing
 * why. */
static bool read_weights(const char *path, float *params, size_t n)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        fprintf(stderr, "Run `make weights` to download and export them.\n");
        return false;
    }
    bool ok = fread(params, sizeof(float), n, f) == n && fgetc(f) == EOF;
    fclose(f);
    if (!ok)
        fprintf(stderr, "%s: expected %zu float32s of GPT-2 124M\n", path, n);
    return ok;
}

/* The first highest of n logits, as generate_text_simple picks. */
static int argmax(const float *x, size_t n)
{
    size_t best = 0;
    for (size_t i = 1; i < n; i++)
        if (x[i] > x[best])
            best = i;
    return (int)best;
}

int main(int argc, char **argv)
{
    const char *weights = DEFAULT_WEIGHTS;
    if (argc == 3 && strcmp(argv[1], "-m") == 0) {
        weights = argv[2];
    } else if (argc != 1) {
        fprintf(stderr, "usage: %s [-m weights.bin] > results.csv\n", argv[0]);
        return 2;
    }

    GPTConfig cfg = GPT_CONFIG_124M;
    cfg.vocab_size = VOCAB_SIZE;
    cfg.qkv_bias = true;
    size_t V = cfg.vocab_size, ctx = cfg.context_length, n_params = gpt_num_params(&cfg);

    /* The token ids of the book's short story, the text of every phase. */
    Tokenizer *tok = tokenizer_load(ENCODER_JSON, VOCAB_BPE);
    FILE *f = fopen(TEXT, "rb");
    if (!tok || !f) {
        fprintf(stderr, "Run `make data` to download the tokenizer files and the text.\n");
        return 1;
    }
    static char text[64 * 1024];
    size_t text_len = fread(text, 1, sizeof text, f);
    fclose(f);
    size_t n_ids;
    int *ids = tokenizer_encode(tok, text, text_len, false, &n_ids);
    tokenizer_free(tok);
    if (!ids || n_ids < ctx) {
        fprintf(stderr, "%s: expected at least %zu tokens\n", TEXT, ctx);
        return 1;
    }

    float *params = malloc(sizeof(float) * n_params);
    GPTModel model;
    gpt_bind(&model, &cfg, params);
    float *cache_mem = malloc(sizeof(float) * kv_cache_size(&cfg, ctx));
    float *scratch = malloc(sizeof(float) * gpt_cached_scratch_size(&model, ctx));
    float *logits = malloc(sizeof(float) * V);
    float *expected = malloc(sizeof(float) * V);
    int *buf = malloc(sizeof(int) * (ctx + 1));
    if (!params || !cache_mem || !scratch || !logits || !expected || !buf) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    print_metadata(weights);

    Measurement *m = begin("load", 0);
    for (size_t r = 0; r < LOAD.warmup + LOAD.reps; r++) {
        double t = now();
        if (!read_weights(weights, params, n_params))
            return 1;
        record(m, &LOAD, r, now() - t);
    }
    double rss_weights = peak_rss();
    KVCache cache;
    kv_cache_bind(&cache, &cfg, ctx, cache_mem);

    /* With the cache: prefill and one decode step at every n. Both give the
     * logits after token n, which must be the same. */
    int next[N_SWEEP];
    float max_diff = 0;
    bool ok = true;
    for (size_t s = 0; s < N_SWEEP; s++) {
        size_t n = SWEEP[s];
        m = begin("prefill", n);
        for (size_t r = 0; r < PREFILL.warmup + PREFILL.reps; r++) {
            cache.len = 0;
            double t = now();
            gpt_forward_cached(logits, ids, n, &model, &cache, scratch);
            record(m, &PREFILL, r, now() - t);
        }
        memcpy(expected, logits, sizeof(float) * V);
        next[s] = argmax(logits, V);

        m = begin("decode", n);
        for (size_t r = 0; r < DECODE.warmup + DECODE.reps; r++) {
            cache.len = n - 1;
            double t = now();
            gpt_forward_cached(logits, ids + n - 1, 1, &model, &cache, scratch);
            record(m, &DECODE, r, now() - t);
        }
        for (size_t v = 0; v < V; v++)
            max_diff = fmaxf(max_diff, fabsf(logits[v] - expected[v]));
    }

    int cached[PROMPT + GENERATED];
    memcpy(cached, ids, sizeof(int) * PROMPT);
    m = begin("generate_cached", PROMPT);
    for (size_t r = 0; r < GENERATE.warmup + GENERATE.reps; r++) {
        cache.len = 0;
        double t = now();
        gpt_forward_cached(logits, cached, PROMPT, &model, &cache, scratch);
        for (size_t i = 0; i < GENERATED; i++) {
            int id = argmax(logits, V);
            cached[PROMPT + i] = id;
            if (i + 1 < GENERATED)
                gpt_forward_cached(logits, &id, 1, &model, &cache, scratch);
        }
        record(m, &GENERATE, r, now() - t);
    }
    double rss_cached = peak_rss();

    /* Without the cache: the book's generate_text_simple, which runs the
     * whole context through the model for every token. */
    float *gen_scratch = malloc(sizeof(float) * generate_scratch_size(&model, ctx));
    if (!gen_scratch) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    for (size_t s = 0; s < N_SWEEP; s++) {
        size_t n = SWEEP[s];
        m = begin("nocache", n);
        for (size_t r = 0; r < NOCACHE.warmup + NOCACHE.reps; r++) {
            memcpy(buf, ids, sizeof(int) * n);
            double t = now();
            generate_text_simple(buf, n, 1, ctx, &model, gen_scratch);
            record(m, &NOCACHE, r, now() - t);
        }
        if (buf[n] != next[s]) {
            fprintf(stderr, "FAIL n=%zu: token %d without the cache, %d with it\n", n, buf[n],
                    next[s]);
            ok = false;
        }
    }

    m = begin("generate_nocache", PROMPT);
    for (size_t r = 0; r < GENERATE.warmup + GENERATE.reps; r++) {
        memcpy(buf, ids, sizeof(int) * PROMPT);
        double t = now();
        generate_text_simple(buf, PROMPT, GENERATED, ctx, &model, gen_scratch);
        record(m, &GENERATE, r, now() - t);
    }
    if (memcmp(buf, cached, sizeof cached) != 0) {
        fprintf(stderr, "FAIL the %d greedy tokens differ with and without the cache\n",
                GENERATED);
        ok = false;
    }
    double rss_all = peak_rss();

    cache.len = 0;
    gpt_forward_cached(logits, ids, DRIFT_N - 1, &model, &cache, scratch);
    m = begin("drift", DRIFT_N);
    for (size_t r = 0; r < DECODE.warmup + DECODE.reps; r++) {
        cache.len = DRIFT_N - 1;
        double t = now();
        gpt_forward_cached(logits, ids + DRIFT_N - 1, 1, &model, &cache, scratch);
        record(m, &DECODE, r, now() - t);
    }

    if (max_diff != 0) {
        fprintf(stderr, "FAIL prefill and decode logits differ by up to %g\n", max_diff);
        ok = false;
    }

    /* The summary: medians in ms, with the range of the timed runs. */
    fprintf(stderr, "\n| phase | n | runs | median ms | min ms | max ms |\n|---|--:|--:|--:|--:|--:|\n");
    for (size_t i = 0; i < n_results; i++) {
        const Measurement *r = &results[i];
        double lo = r->seconds[0], hi = r->seconds[0];
        for (size_t j = 1; j < r->reps; j++) {
            lo = fmin(lo, r->seconds[j]);
            hi = fmax(hi, r->seconds[j]);
        }
        fprintf(stderr, "| %s | %zu | %zu | %.2f | %.2f | %.2f |\n", r->phase, r->n, r->reps,
                1e3 * median(r), 1e3 * lo, 1e3 * hi);
    }

    fprintf(stderr, "\n| n | prefill ms | prefill tok/s | decode ms | decode tok/s | nocache ms "
                    "| nocache / decode |\n|--:|--:|--:|--:|--:|--:|--:|\n");
    for (size_t s = 0; s < N_SWEEP; s++) {
        double pre = median(find("prefill", SWEEP[s]));
        double dec = median(find("decode", SWEEP[s]));
        double no = median(find("nocache", SWEEP[s]));
        fprintf(stderr, "| %zu | %.1f | %.0f | %.2f | %.1f | %.1f | %.0fx |\n", SWEEP[s],
                1e3 * pre, SWEEP[s] / pre, 1e3 * dec, 1 / dec, 1e3 * no, no / dec);
    }

    double gc = median(find("generate_cached", PROMPT));
    double gn = median(find("generate_nocache", PROMPT));
    fprintf(stderr, "\n%d tokens after a %d-token prompt: %.2f s with the cache, %.2f s without "
                    "(%.1fx)\n",
            GENERATED, PROMPT, gc, gn, gn / gc);
    fprintf(stderr, "drift: decode at n=%d %.2f ms at the start, %.2f ms at the end\n", DRIFT_N,
            1e3 * median(find("decode", DRIFT_N)), 1e3 * median(find("drift", DRIFT_N)));

    double MB = 1024.0 * 1024.0;
    fprintf(stderr,
            "\nmemory (MiB): weights %.1f, KV cache %.1f, cached scratch %.1f, "
            "generate_text_simple scratch %.1f\n"
            "peak RSS (MiB): after load %.1f, after the cached phases %.1f, at the end %.1f\n",
            sizeof(float) * n_params / MB, sizeof(float) * kv_cache_size(&cfg, ctx) / MB,
            sizeof(float) * gpt_cached_scratch_size(&model, ctx) / MB,
            sizeof(float) * generate_scratch_size(&model, ctx) / MB, rss_weights / MB,
            rss_cached / MB, rss_all / MB);
    fprintf(stderr, "\ncorrectness: %s\n",
            ok ? "greedy tokens match with and without the cache, prefill and decode logits "
                 "identical"
               : "FAILED");

    free(gen_scratch);
    free(buf);
    free(expected);
    free(logits);
    free(scratch);
    free(cache_mem);
    free(params);
    free(ids);
    return ok ? 0 : 1;
}
