/*
 * Chat with GPT-2 124M fine-tuned to follow instructions on the Alpaca
 * dataset (vicgalle/gpt2-alpaca). Each message becomes an instruction in
 * the prompt format the model was trained on, and the reply is sampled
 * token by token with temperature scaling and top-k (section 5.3), on a KV
 * cache that keeps the whole conversation.
 *
 * Usage: chat [-m weights.bin] [-t temperature] [-k top_k] [-n max_tokens] [-s seed]
 * Run from the repository root after `make data weights chat`.
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gpt.h"
#include "kv_cache.h"
#include "rng.h"
#include "tokenizer.h"

#define DEFAULT_WEIGHTS "data/gpt2_alpaca.bin"
#define ENCODER_JSON "data/gpt2/encoder.json"
#define VOCAB_BPE "data/gpt2/vocab.bpe"

/*
 * The model's prompt format, with three tokens it added after GPT-2's
 * <|endoftext|> (50256):
 *
 *   Below is an instruction that describes a task. ... the request.\n\n
 *   ### Instruction:\n{message}\n\n### Response:\n{reply}\n\n### End
 *
 * Earlier turns stay in the context, each followed by "\n\n" and the next.
 */
enum { TOKEN_END = 50257, TOKEN_INSTRUCTION = 50258, TOKEN_RESPONSE = 50259, VOCAB_SIZE = 50260 };

static const char HEADER[] = "Below is an instruction that describes a task. "
                             "Write a response that appropriately completes the request.\n\n";

typedef struct {
    const char *weights;
    float temperature; /* 0 for greedy decoding */
    size_t top_k;      /* 0 for the whole vocabulary */
    size_t max_tokens; /* most tokens per reply */
    uint64_t seed;
} Options;

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s [options]\n"
            "  -m PATH   weights from `make weights` (default " DEFAULT_WEIGHTS ")\n"
            "  -t TEMP   temperature, 0 for greedy decoding (default 0.7)\n"
            "  -k K      sample from the K most likely tokens, 0 for all (default 40)\n"
            "  -n N      most tokens per reply (default 256)\n"
            "  -s SEED   random seed (default: the time)\n",
            prog);
}

/* Parses argv into opts; false on a bad option, after printing why. */
static bool parse_options(int argc, char **argv, Options *opts)
{
    *opts = (Options){DEFAULT_WEIGHTS, 0.7f, 40, 256, (uint64_t)time(NULL)};
    for (int i = 1; i < argc; i += 2) {
        const char *flag = argv[i], *value = argv[i + 1];
        if (strcmp(flag, "-h") == 0 || strcmp(flag, "--help") == 0) {
            usage(argv[0]);
            exit(0);
        }
        if (!value) {
            usage(argv[0]);
            return false;
        }
        char *end = NULL;
        bool bad = value[0] == '-';
        if (strcmp(flag, "-m") == 0) {
            opts->weights = value;
        } else if (strcmp(flag, "-t") == 0) {
            opts->temperature = strtof(value, &end);
        } else if (strcmp(flag, "-k") == 0) {
            opts->top_k = strtoul(value, &end, 10);
        } else if (strcmp(flag, "-n") == 0) {
            opts->max_tokens = strtoul(value, &end, 10);
        } else if (strcmp(flag, "-s") == 0) {
            opts->seed = strtoull(value, &end, 10);
        } else {
            usage(argv[0]);
            return false;
        }
        if (bad || (end && (end == value || *end))) {
            fprintf(stderr, "%s: bad value for %s: %s\n", argv[0], flag, value);
            return false;
        }
    }
    return true;
}

/* Reads the exported weights into a malloc'd buffer of gpt_num_params(cfg)
 * floats, or returns NULL after printing why. */
static float *load_weights(const char *path, const GPTConfig *cfg)
{
    size_t n = gpt_num_params(cfg);
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        fprintf(stderr, "Run `make weights` to download and export them.\n");
        return NULL;
    }
    float *params = malloc(sizeof(float) * n);
    bool ok = params && fread(params, sizeof(float), n, f) == n && fgetc(f) == EOF;
    fclose(f);
    if (!ok) {
        fprintf(stderr, "%s: expected %zu float32s of GPT-2 124M with a %zu-token vocabulary\n",
                path, n, cfg->vocab_size);
        free(params);
        return NULL;
    }
    return params;
}

/*
 * The next token after logits [V] (section 5.3). Temperature 0 picks the
 * most likely one. Otherwise only tokens whose logit reaches the top_k-th
 * highest stay, as the book masks the rest to -inf, and one is drawn from
 * softmax(logits / temperature) over them. probs is room for V floats.
 */
static int sample(const float *logits, float *probs, size_t V, float temperature, size_t top_k,
                  Rng *rng)
{
    size_t best = 0;
    for (size_t v = 1; v < V; v++)
        if (logits[v] > logits[best])
            best = v;
    if (temperature == 0)
        return (int)best;

    /* The top_k highest logits so far, in descending order, in probs. */
    float threshold = -INFINITY;
    if (top_k > 0 && top_k < V) {
        size_t kept = 0;
        for (size_t v = 0; v < V; v++) {
            if (kept == top_k && logits[v] <= probs[kept - 1])
                continue;
            size_t i = kept < top_k ? kept++ : kept - 1;
            for (; i > 0 && probs[i - 1] < logits[v]; i--)
                probs[i] = probs[i - 1];
            probs[i] = logits[v];
        }
        threshold = probs[top_k - 1];
    }

    float sum = 0;
    for (size_t v = 0; v < V; v++) {
        probs[v] = logits[v] >= threshold ? expf((logits[v] - logits[best]) / temperature) : 0;
        sum += probs[v];
    }
    float u;
    rng_uniform(rng, &u, 1, 0, 1);
    float target = u * sum, cumulative = 0;
    for (size_t v = 0; v < V; v++) {
        cumulative += probs[v];
        if (probs[v] > 0 && cumulative > target)
            return (int)v;
    }
    return (int)best;
}

/* A line of stdin without its newline, in *buf of *cap bytes grown as
 * needed; false at the end of input. */
static bool read_line(char **buf, size_t *cap)
{
    size_t len = 0;
    for (;;) {
        if (*cap - len < 2) {
            size_t grown = *cap ? *cap * 2 : 256;
            char *p = realloc(*buf, grown);
            if (!p)
                return false;
            *buf = p;
            *cap = grown;
        }
        if (!fgets(*buf + len, (int)(*cap - len), stdin))
            return len > 0;
        len += strlen(*buf + len);
        if (len > 0 && (*buf)[len - 1] == '\n') {
            (*buf)[--len] = '\0';
            if (len > 0 && (*buf)[len - 1] == '\r')
                (*buf)[--len] = '\0';
            return true;
        }
    }
}

/* A growable token list. */
typedef struct {
    int *ids;
    size_t len, cap;
} Tokens;

static bool push(Tokens *t, const int *ids, size_t n)
{
    if (t->len + n > t->cap) {
        size_t cap = t->cap ? t->cap : 64;
        while (cap < t->len + n)
            cap *= 2;
        int *p = realloc(t->ids, sizeof(int) * cap);
        if (!p)
            return false;
        t->ids = p;
        t->cap = cap;
    }
    memcpy(t->ids + t->len, ids, sizeof(int) * n);
    t->len += n;
    return true;
}

static bool push_text(Tokens *t, const Tokenizer *tok, const char *text)
{
    size_t n;
    int *ids = tokenizer_encode(tok, text, strlen(text), false, &n);
    bool ok = ids && push(t, ids, n);
    free(ids);
    return ok;
}

static bool push_id(Tokens *t, int id)
{
    return push(t, &id, 1);
}

/*
 * The tokens of one turn into turn: the header to start a conversation, or
 * the end of the previous reply, then message as the instruction. Text
 * between the added tokens is encoded on its own, as the Hugging Face
 * tokenizer splits it.
 */
static bool build_turn(Tokens *turn, const Tokenizer *tok, const char *message, bool first)
{
    turn->len = 0;
    bool ok = first ? push_text(turn, tok, HEADER)
                    : push_id(turn, TOKEN_END) && push_text(turn, tok, "\n\n");
    char *body = malloc(strlen(message) + 4);
    ok = ok && body;
    if (ok) {
        sprintf(body, "\n%s\n\n", message);
        ok = push_id(turn, TOKEN_INSTRUCTION) && push_text(turn, tok, body) &&
             push_id(turn, TOKEN_RESPONSE);
    }
    free(body);
    return ok;
}

/*
 * Prints a reply as it streams, without the whitespace around it: the
 * model starts with a newline and ends with "\n\n" before ### End.
 * Whitespace waits in pending until something follows it.
 */
typedef struct {
    bool started;
    char pending[256];
    size_t n_pending;
} Printer;

static void print_bytes(Printer *pr, const char *s, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        char c = s[i];
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
            if (!pr->started)
                continue;
            if (pr->n_pending == sizeof pr->pending) {
                fwrite(pr->pending, 1, pr->n_pending, stdout);
                pr->n_pending = 0;
            }
            pr->pending[pr->n_pending++] = c;
        } else {
            fwrite(pr->pending, 1, pr->n_pending, stdout);
            pr->n_pending = 0;
            putchar(c);
            pr->started = true;
        }
    }
    fflush(stdout);
}

int main(int argc, char **argv)
{
    Options opts;
    if (!parse_options(argc, argv, &opts))
        return 2;

    GPTConfig cfg = GPT_CONFIG_124M;
    cfg.vocab_size = VOCAB_SIZE;
    cfg.qkv_bias = true; /* GPT-2's checkpoints have query, key and value biases */
    size_t V = cfg.vocab_size, capacity = cfg.context_length;

    Tokenizer *tok = tokenizer_load(ENCODER_JSON, VOCAB_BPE);
    if (!tok) {
        fprintf(stderr, "Run `make data` to download the tokenizer files.\n");
        return 1;
    }
    int eot = tokenizer_eot_id(tok);
    float *params = load_weights(opts.weights, &cfg);
    if (!params)
        return 1;
    GPTModel model;
    gpt_bind(&model, &cfg, params);

    KVCache cache;
    float *cache_mem = malloc(sizeof(float) * kv_cache_size(&cfg, capacity));
    float *scratch = malloc(sizeof(float) * gpt_cached_scratch_size(&model, capacity));
    float *logits = malloc(sizeof(float) * V);
    float *probs = malloc(sizeof(float) * V);
    if (!cache_mem || !scratch || !logits || !probs) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    kv_cache_bind(&cache, &cfg, capacity, cache_mem);
    Rng rng;
    rng_seed(&rng, opts.seed);

    printf("GPT-2 124M fine-tuned on Alpaca instructions, seed %llu.\n"
           "Type a message. /reset starts a new conversation, /quit or Ctrl-D exits.\n",
           (unsigned long long)opts.seed);

    char *line = NULL;
    size_t line_cap = 0;
    Tokens turn = {0};
    for (;;) {
        printf("\n> ");
        fflush(stdout);
        if (!read_line(&line, &line_cap)) {
            putchar('\n');
            break;
        }
        /* The message without surrounding whitespace. */
        char *message = line + strspn(line, " \t");
        size_t len = strlen(message);
        while (len > 0 && (message[len - 1] == ' ' || message[len - 1] == '\t'))
            message[--len] = '\0';
        if (len == 0)
            continue;
        if (strcmp(message, "/quit") == 0 || strcmp(message, "/exit") == 0)
            break;
        if (strcmp(message, "/reset") == 0) {
            cache.len = 0;
            printf("(new conversation)\n");
            continue;
        }

        bool ok = build_turn(&turn, tok, message, cache.len == 0);
        /* Keep room for a whole reply, starting over if the context lacks it. */
        if (ok && cache.len > 0 && cache.len + turn.len + opts.max_tokens > capacity) {
            printf("(context full: starting a new conversation)\n");
            cache.len = 0;
            ok = build_turn(&turn, tok, message, true);
        }
        if (!ok) {
            fprintf(stderr, "out of memory\n");
            break;
        }
        if (turn.len >= capacity) {
            printf("(message too long: %zu tokens, the context holds %zu)\n", turn.len, capacity);
            continue;
        }

        gpt_forward_cached(logits, turn.ids, turn.len, &model, &cache, scratch);
        Printer pr = {0};
        for (size_t i = 0; i < opts.max_tokens; i++) {
            int id = sample(logits, probs, V, opts.temperature, opts.top_k, &rng);
            /* <|endoftext|> or one of the format's tokens ends the reply. */
            if (id >= eot)
                break;
            size_t n_bytes = 0;
            char *text = tokenizer_decode(tok, &id, 1, &n_bytes);
            print_bytes(&pr, text, text ? n_bytes : 0);
            free(text);
            if (cache.len == capacity)
                break;
            gpt_forward_cached(logits, &id, 1, &model, &cache, scratch);
        }
        putchar('\n');
    }

    free(line);
    free(turn.ids);
    free(probs);
    free(logits);
    free(scratch);
    free(cache_mem);
    free(params);
    tokenizer_free(tok);
    return 0;
}
