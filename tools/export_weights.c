/*
 * Exports a GPT-2 checkpoint in Hugging Face's safetensors format, such as
 * vicgalle/gpt2-alpaca's model.safetensors, to the file the C model loads:
 * gpt_num_params(cfg) float32s in gpt_bind's order, with qkv_bias. This is
 * the mapping of the book's load_weights_into_gpt (section 5.5).
 *
 * Usage: export_weights model.safetensors out.bin
 * `make weights` downloads vicgalle/gpt2-alpaca and runs this.
 *
 * A safetensors file is an 8-byte little-endian header length, a JSON
 * header mapping each tensor's name to its dtype, shape and byte range
 * [begin, end) in the data, then the data. GPT-2's layers are Conv1D, which
 * store weights as [in x out], the transpose of nn.Linear's [out x in], and
 * c_attn holds the query, key and value projections side by side.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_DIMS 4

typedef struct {
    char name[128];
    char dtype[16];
    size_t ndim;
    uint64_t shape[MAX_DIMS];
    uint64_t begin, end;
} Tensor;

static void die(const char *msg, const char *detail)
{
    fprintf(stderr, "export_weights: %s%s%s\n", msg, detail ? ": " : "", detail ? detail : "");
    exit(1);
}

/* ---- The JSON header: an object of objects, strings, arrays of integers. */

typedef struct {
    const char *p, *end;
} Json;

static void skip_ws(Json *j)
{
    while (j->p < j->end && (*j->p == ' ' || *j->p == '\t' || *j->p == '\n' || *j->p == '\r'))
        j->p++;
}

static void expect(Json *j, char c)
{
    skip_ws(j);
    if (j->p == j->end || *j->p != c)
        die("malformed header", NULL);
    j->p++;
}

/* Whether the next character is c, consuming it if so. */
static bool accept(Json *j, char c)
{
    skip_ws(j);
    if (j->p < j->end && *j->p == c) {
        j->p++;
        return true;
    }
    return false;
}

/* A string into out, cap bytes with the NUL. Names and dtypes are ASCII, so
 * an escape just keeps the character after the backslash. */
static void parse_string(Json *j, char *out, size_t cap)
{
    size_t n = 0;
    expect(j, '"');
    while (j->p < j->end && *j->p != '"') {
        if (*j->p == '\\' && ++j->p == j->end)
            break;
        if (n + 1 < cap)
            out[n++] = *j->p;
        j->p++;
    }
    expect(j, '"');
    out[n] = '\0';
}

static uint64_t parse_uint(Json *j)
{
    skip_ws(j);
    if (j->p == j->end || *j->p < '0' || *j->p > '9')
        die("malformed header", "expected an integer");
    uint64_t v = 0;
    while (j->p < j->end && *j->p >= '0' && *j->p <= '9')
        v = v * 10 + (uint64_t)(*j->p++ - '0');
    return v;
}

/* Skips any value, such as the "__metadata__" object of strings. */
static void skip_value(Json *j)
{
    char buf[8];
    skip_ws(j);
    if (j->p == j->end)
        die("malformed header", NULL);
    char c = *j->p;
    if (c == '"') {
        parse_string(j, buf, sizeof buf);
    } else if (c == '{' || c == '[') {
        char close = c == '{' ? '}' : ']';
        j->p++;
        if (accept(j, close))
            return;
        do {
            if (c == '{') {
                parse_string(j, buf, sizeof buf);
                expect(j, ':');
            }
            skip_value(j);
        } while (accept(j, ','));
        expect(j, close);
    } else {
        /* A number, true, false or null. */
        while (j->p < j->end && !strchr(",}] \t\n\r", *j->p))
            j->p++;
    }
}

/* An array of up to max integers into out, returning how many. */
static size_t parse_uints(Json *j, uint64_t *out, size_t max)
{
    size_t n = 0;
    expect(j, '[');
    if (accept(j, ']'))
        return 0;
    do {
        if (n == max)
            die("malformed header", "array too long");
        out[n++] = parse_uint(j);
    } while (accept(j, ','));
    expect(j, ']');
    return n;
}

/* {"dtype": ..., "shape": [...], "data_offsets": [begin, end]} */
static void parse_tensor(Json *j, Tensor *t)
{
    char key[32];
    uint64_t offsets[2];
    bool has_offsets = false;
    expect(j, '{');
    do {
        parse_string(j, key, sizeof key);
        expect(j, ':');
        if (strcmp(key, "dtype") == 0) {
            parse_string(j, t->dtype, sizeof t->dtype);
        } else if (strcmp(key, "shape") == 0) {
            t->ndim = parse_uints(j, t->shape, MAX_DIMS);
        } else if (strcmp(key, "data_offsets") == 0) {
            if (parse_uints(j, offsets, 2) != 2)
                die("malformed data_offsets of", t->name);
            has_offsets = true;
        } else {
            skip_value(j);
        }
    } while (accept(j, ','));
    expect(j, '}');
    if (!has_offsets || offsets[1] < offsets[0])
        die("missing data_offsets for", t->name);
    t->begin = offsets[0];
    t->end = offsets[1];
}

/* Every tensor of the header, in a malloc'd array of *n. */
static Tensor *parse_header(const char *text, size_t len, size_t *n)
{
    Json j = {text, text + len};
    size_t cap = 256;
    Tensor *tensors = malloc(cap * sizeof *tensors);
    if (!tensors)
        die("out of memory", NULL);
    *n = 0;
    expect(&j, '{');
    if (accept(&j, '}'))
        return tensors;
    do {
        if (*n == cap) {
            cap *= 2;
            tensors = realloc(tensors, cap * sizeof *tensors);
            if (!tensors)
                die("out of memory", NULL);
        }
        Tensor *t = &tensors[*n];
        memset(t, 0, sizeof *t);
        parse_string(&j, t->name, sizeof t->name);
        expect(&j, ':');
        if (strcmp(t->name, "__metadata__") == 0) {
            skip_value(&j);
            continue;
        }
        parse_tensor(&j, t);
        (*n)++;
    } while (accept(&j, ','));
    expect(&j, '}');
    return tensors;
}

/* ---- The checkpoint. */

typedef struct {
    FILE *f;
    uint64_t data_start; /* file offset of the data, after the header */
    Tensor *tensors;
    size_t n_tensors;
} Checkpoint;

static void open_checkpoint(Checkpoint *ck, const char *path)
{
    ck->f = fopen(path, "rb");
    if (!ck->f) {
        perror(path);
        exit(1);
    }
    unsigned char len_bytes[8];
    if (fread(len_bytes, 1, 8, ck->f) != 8)
        die("not a safetensors file", path);
    uint64_t len = 0;
    for (int i = 7; i >= 0; i--)
        len = len << 8 | len_bytes[i];
    if (len == 0 || len > 100u << 20)
        die("not a safetensors file", path);
    char *text = malloc(len);
    if (!text)
        die("out of memory", NULL);
    if (fread(text, 1, len, ck->f) != len)
        die("truncated header in", path);
    ck->data_start = 8 + len;
    ck->tensors = parse_header(text, len, &ck->n_tensors);
    free(text);
}

/* GPT2LMHeadModel saves names under "transformer.", GPT2Model without. */
static const Tensor *find(const Checkpoint *ck, const char *name)
{
    char full[160];
    snprintf(full, sizeof full, "transformer.%s", name);
    for (size_t i = 0; i < ck->n_tensors; i++)
        if (strcmp(ck->tensors[i].name, full) == 0 || strcmp(ck->tensors[i].name, name) == 0)
            return &ck->tensors[i];
    return NULL;
}

/* The float32 tensor name of shape [rows x cols], or [rows] for cols 0, in a
 * malloc'd array. */
static float *load(const Checkpoint *ck, const char *name, uint64_t rows, uint64_t cols)
{
    const Tensor *t = find(ck, name);
    if (!t)
        die("missing tensor", name);
    if (strcmp(t->dtype, "F32") != 0)
        die("expected dtype F32 for", name);
    size_t ndim = cols ? 2 : 1;
    if (t->ndim != ndim || t->shape[0] != rows || (cols && t->shape[1] != cols))
        die("unexpected shape of", name);
    uint64_t n = rows * (cols ? cols : 1);
    if (t->end - t->begin != n * sizeof(float))
        die("byte range does not match the shape of", name);
    float *data = malloc(n * sizeof(float));
    if (!data)
        die("out of memory", NULL);
    if (fseek(ck->f, (long)(ck->data_start + t->begin), SEEK_SET) != 0 ||
        fread(data, sizeof(float), n, ck->f) != n)
        die("truncated data of", name);
    return data;
}

/* ---- The output, in gpt_bind's order. */

static FILE *out;
static uint64_t written;

static void write_floats(const float *x, uint64_t n)
{
    if (fwrite(x, sizeof(float), n, out) != n)
        die("write failed", NULL);
    written += n;
}

/* Columns [col, col + n) of w [rows x cols], transposed to [n x rows]: a
 * Conv1D weight as the nn.Linear weight it stands for. */
static void write_transposed(const float *w, uint64_t rows, uint64_t cols, uint64_t col,
                             uint64_t n)
{
    float *row = malloc(rows * sizeof(float));
    if (!row)
        die("out of memory", NULL);
    for (uint64_t j = col; j < col + n; j++) {
        for (uint64_t i = 0; i < rows; i++)
            row[i] = w[i * cols + j];
        write_floats(row, rows);
    }
    free(row);
}

/* Loads, writes and frees one tensor. */
static void copy(const Checkpoint *ck, const char *name, uint64_t rows, uint64_t cols)
{
    float *x = load(ck, name, rows, cols);
    write_floats(x, rows * (cols ? cols : 1));
    free(x);
}

/* Loads a Conv1D weight [rows x cols] and writes it as nn.Linear's. */
static void copy_transposed(const Checkpoint *ck, const char *name, uint64_t rows, uint64_t cols)
{
    float *w = load(ck, name, rows, cols);
    write_transposed(w, rows, cols, 0, cols);
    free(w);
}

/* One transformer block in transformer_block_bind's order: W_query, W_key
 * and W_value with their biases, out_proj, the feed forward network's two
 * layers, then norm1 and norm2. */
static void write_block(const Checkpoint *ck, size_t l, uint64_t d)
{
    char name[96];
#define NAME(suffix) (snprintf(name, sizeof name, "h.%zu.%s", l, suffix), name)
    float *qkv_w = load(ck, NAME("attn.c_attn.weight"), d, 3 * d);
    float *qkv_b = load(ck, NAME("attn.c_attn.bias"), 3 * d, 0);
    for (uint64_t part = 0; part < 3; part++) {
        write_transposed(qkv_w, d, 3 * d, part * d, d);
        write_floats(qkv_b + part * d, d);
    }
    free(qkv_w);
    free(qkv_b);
    copy_transposed(ck, NAME("attn.c_proj.weight"), d, d);
    copy(ck, NAME("attn.c_proj.bias"), d, 0);
    copy_transposed(ck, NAME("mlp.c_fc.weight"), d, 4 * d);
    copy(ck, NAME("mlp.c_fc.bias"), 4 * d, 0);
    copy_transposed(ck, NAME("mlp.c_proj.weight"), 4 * d, d);
    copy(ck, NAME("mlp.c_proj.bias"), d, 0);
    copy(ck, NAME("ln_1.weight"), d, 0);
    copy(ck, NAME("ln_1.bias"), d, 0);
    copy(ck, NAME("ln_2.weight"), d, 0);
    copy(ck, NAME("ln_2.bias"), d, 0);
#undef NAME
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s model.safetensors out.bin\n", argv[0]);
        return 2;
    }
    /* Both the checkpoint and the output are little-endian float32s. */
    uint32_t one = 1;
    unsigned char first_byte;
    memcpy(&first_byte, &one, 1);
    if (first_byte != 1)
        die("big-endian hosts are not supported", NULL);

    Checkpoint ck;
    open_checkpoint(&ck, argv[1]);

    const Tensor *wte = find(&ck, "wte.weight"), *wpe = find(&ck, "wpe.weight");
    if (!wte || wte->ndim != 2 || !wpe || wpe->ndim != 2)
        die("not a GPT-2 checkpoint", argv[1]);
    uint64_t vocab_size = wte->shape[0], d = wte->shape[1], context_length = wpe->shape[0];
    size_t n_layers = 0;
    for (char name[64];; n_layers++) {
        snprintf(name, sizeof name, "h.%zu.ln_1.weight", n_layers);
        if (!find(&ck, name))
            break;
    }
    printf("vocab_size %llu, context_length %llu, emb_dim %llu, n_layers %zu\n",
           (unsigned long long)vocab_size, (unsigned long long)context_length,
           (unsigned long long)d, n_layers);

    /* Written to a temporary name first, so an interrupted export leaves no
     * file that looks complete. */
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s.part", argv[2]);
    out = fopen(tmp, "wb");
    if (!out) {
        perror(tmp);
        return 1;
    }
    copy(&ck, "wte.weight", vocab_size, d);
    copy(&ck, "wpe.weight", context_length, d);
    for (size_t l = 0; l < n_layers; l++)
        write_block(&ck, l, d);
    copy(&ck, "ln_f.weight", d, 0);
    copy(&ck, "ln_f.bias", d, 0);
    /* GPT-2 ties the output head to the token embeddings, so checkpoints
     * usually leave lm_head out. */
    copy(&ck, find(&ck, "lm_head.weight") ? "lm_head.weight" : "wte.weight", vocab_size, d);
    if (fclose(out) != 0)
        die("write failed", tmp);
    fclose(ck.f);
    free(ck.tensors);
    if (rename(tmp, argv[2]) != 0) {
        perror(argv[2]);
        return 1;
    }
    printf("wrote %s: %llu floats (%.2f MB)\n", argv[2], (unsigned long long)written,
           (double)written * sizeof(float) / (1024 * 1024));
    return 0;
}
