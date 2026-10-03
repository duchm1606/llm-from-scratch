#include "tokenizer.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "unicode_tables.h"

#define EOT_TEXT "<|endoftext|>"
#define EOT_TEXT_LEN (sizeof(EOT_TEXT) - 1)

typedef struct {
    uint64_t key;    /* left id << 32 | right id */
    int32_t rank;    /* line in vocab.bpe; lower merges first */
    int32_t merged;  /* id of the merged token; -1 marks an empty slot */
} MergeSlot;

struct Tokenizer {
    int vocab_size;
    int eot_id;
    int byte_ids[256];   /* id of the single-byte token for each byte value */
    char *bytes;         /* raw bytes of every token, back to back */
    uint32_t *tok_off;   /* token id -> start of its bytes */
    uint32_t *tok_len;   /* token id -> number of bytes */
    MergeSlot *merges;   /* open addressing, keyed by (left id, right id) */
    size_t merges_cap;   /* power of two */
};

/* ---- UTF-8 and character classes -------------------------------------- */

/*
 * Decodes one codepoint from s[0..n), n >= 1. Returns its length in bytes, or
 * 0 if the bytes are not valid UTF-8 (overlong forms and surrogates included).
 */
static size_t utf8_decode(const unsigned char *s, size_t n, uint32_t *cp)
{
    size_t len;
    uint32_t v, min;

    if (s[0] < 0x80) {
        *cp = s[0];
        return 1;
    } else if ((s[0] & 0xE0) == 0xC0) {
        len = 2, v = s[0] & 0x1F, min = 0x80;
    } else if ((s[0] & 0xF0) == 0xE0) {
        len = 3, v = s[0] & 0x0F, min = 0x800;
    } else if ((s[0] & 0xF8) == 0xF0) {
        len = 4, v = s[0] & 0x07, min = 0x10000;
    } else {
        return 0;
    }
    if (n < len)
        return 0;
    for (size_t i = 1; i < len; i++) {
        if ((s[i] & 0xC0) != 0x80)
            return 0;
        v = v << 6 | (s[i] & 0x3F);
    }
    if (v < min || v > 0x10FFFF || (v >= 0xD800 && v <= 0xDFFF))
        return 0;
    *cp = v;
    return len;
}

enum { CLS_LETTER, CLS_NUMBER, CLS_SPACE, CLS_OTHER };

static bool in_ranges(uint32_t cp, const CodepointRange *r, size_t n)
{
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < r[mid].lo)
            hi = mid;
        else if (cp > r[mid].hi)
            lo = mid + 1;
        else
            return true;
    }
    return false;
}

#define IN_TABLE(cp, table) in_ranges(cp, table, sizeof(table) / sizeof(table[0]))

/* The classes behind \s, \p{L} and \p{N} in the pre-tokenizer regex. */
static int classify(uint32_t cp)
{
    if (IN_TABLE(cp, unicode_white_space))
        return CLS_SPACE;
    if (IN_TABLE(cp, unicode_letters))
        return CLS_LETTER;
    if (IN_TABLE(cp, unicode_numbers))
        return CLS_NUMBER;
    return CLS_OTHER;
}

typedef struct {
    uint32_t cp;
    size_t len;
    int cls;
} Char;

/* Reads the character at s[i], i < n. An invalid UTF-8 byte reads as a
 * one-byte character of class CLS_OTHER. */
static Char char_at(const unsigned char *s, size_t n, size_t i)
{
    Char c;
    c.len = utf8_decode(s + i, n - i, &c.cp);
    if (c.len == 0) {
        c.cp = 0xFFFD;
        c.len = 1;
        c.cls = CLS_OTHER;
    } else {
        c.cls = classify(c.cp);
    }
    return c;
}

/* ---- Pre-tokenizer ------------------------------------------------------ */

/* Length of 's 't 'm 'd 'll 've 're at s[0] == '\'', or 0. Case-sensitive,
 * like GPT-2: "'S" is not a contraction. */
static size_t contraction_len(const unsigned char *s, size_t n)
{
    if (n >= 2 && (s[1] == 's' || s[1] == 't' || s[1] == 'm' || s[1] == 'd'))
        return 2;
    if (n >= 3 && ((s[1] == 'l' && s[2] == 'l') || (s[1] == 'v' && s[2] == 'e') ||
                   (s[1] == 'r' && s[2] == 'e')))
        return 3;
    return 0;
}

/*
 * Returns the end of the pre-token starting at s[i], i < n. This is GPT-2's
 * split regex, whose alternatives are tried in order:
 *
 *   's|'t|'re|'ve|'m|'ll|'d     contractions
 *    ?\p{L}+                     letters, with at most one leading space
 *    ?\p{N}+                     numbers, likewise
 *    ?[^\s\p{L}\p{N}]+           anything else, likewise
 *   \s+(?!\S)                    whitespace, leaving the last char to the next word
 *   \s+                          a single whitespace char before a word
 *
 * So "a   b" splits into "a", "  ", " b": the space before "b" stays with it.
 */
static size_t pretoken_end(const unsigned char *s, size_t n, size_t i)
{
    Char c = char_at(s, n, i);
    if (c.cp == '\'') {
        size_t k = contraction_len(s + i, n - i);
        if (k > 0)
            return i + k;
    }

    size_t j = i;
    Char run = c;
    if (c.cp == ' ' && i + 1 < n) {
        Char next = char_at(s, n, i + 1);
        if (next.cls != CLS_SPACE) {
            j = i + 1;
            run = next;
        }
    }
    if (run.cls != CLS_SPACE) {
        size_t end = j + run.len;
        while (end < n) {
            Char d = char_at(s, n, end);
            if (d.cls != run.cls)
                break;
            end += d.len;
        }
        return end;
    }

    size_t last = i, end = i;
    while (end < n) {
        Char d = char_at(s, n, end);
        if (d.cls != CLS_SPACE)
            break;
        last = end;
        end += d.len;
    }
    if (end < n && last > i)
        return last;
    return end;
}

/* ---- BPE ---------------------------------------------------------------- */

static uint64_t mix64(uint64_t x)
{
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static uint64_t pair_key(int left, int right)
{
    return (uint64_t)(uint32_t)left << 32 | (uint32_t)right;
}

static const MergeSlot *merge_find(const Tokenizer *tok, int left, int right)
{
    uint64_t key = pair_key(left, right);
    size_t mask = tok->merges_cap - 1;
    for (size_t i = mix64(key) & mask;; i = (i + 1) & mask) {
        const MergeSlot *m = &tok->merges[i];
        if (m->merged < 0)
            return NULL;
        if (m->key == key)
            return m;
    }
}

/*
 * Applies the merge rules to ids[0..n) in place, as encoder.py's bpe() does:
 * repeatedly take the adjacent pair with the lowest rank and merge every
 * occurrence of it, until no adjacent pair has a rule. Returns the new length.
 */
static size_t bpe_merge(const Tokenizer *tok, int *ids, size_t n)
{
    while (n >= 2) {
        const MergeSlot *best = NULL;
        int left = 0, right = 0;
        for (size_t i = 0; i + 1 < n; i++) {
            const MergeSlot *m = merge_find(tok, ids[i], ids[i + 1]);
            if (m && (!best || m->rank < best->rank)) {
                best = m;
                left = ids[i];
                right = ids[i + 1];
            }
        }
        if (!best)
            break;

        size_t w = 0;
        for (size_t i = 0; i < n;) {
            if (i + 1 < n && ids[i] == left && ids[i + 1] == right) {
                ids[w++] = best->merged;
                i += 2;
            } else {
                ids[w++] = ids[i++];
            }
        }
        n = w;
    }
    return n;
}

/* Encodes s[start..end), which holds no special tokens, appending to ids[n..].
 * Returns the new number of ids. */
static size_t encode_ordinary(const Tokenizer *tok, const unsigned char *s, size_t start,
                              size_t end, int *ids, size_t n)
{
    for (size_t i = start; i < end;) {
        size_t piece_end = pretoken_end(s, end, i);
        size_t m = 0;
        for (size_t k = i; k < piece_end; k++)
            ids[n + m++] = tok->byte_ids[s[k]];
        n += bpe_merge(tok, ids + n, m);
        i = piece_end;
    }
    return n;
}

static size_t find_eot(const unsigned char *s, size_t len, size_t from)
{
    for (size_t i = from; i + EOT_TEXT_LEN <= len; i++) {
        if (memcmp(s + i, EOT_TEXT, EOT_TEXT_LEN) == 0)
            return i;
    }
    return len;
}

int *tokenizer_encode(const Tokenizer *tok, const char *text, size_t len,
                      bool allow_special, size_t *n_ids)
{
    /* Every token covers at least one byte, so len ids always suffice. */
    if (len > SIZE_MAX / sizeof(int))
        return NULL;
    int *ids = malloc((len > 0 ? len : 1) * sizeof *ids);
    if (!ids)
        return NULL;

    const unsigned char *s = (const unsigned char *)text;
    size_t n = 0;
    for (size_t i = 0; i < len;) {
        /* Special tokens are cut out first; the regex never sees them. */
        size_t seg_end = allow_special && tok->eot_id >= 0 ? find_eot(s, len, i) : len;
        n = encode_ordinary(tok, s, i, seg_end, ids, n);
        i = seg_end;
        if (i < len) {
            ids[n++] = tok->eot_id;
            i += EOT_TEXT_LEN;
        }
    }
    *n_ids = n;
    return ids;
}

char *tokenizer_decode(const Tokenizer *tok, const int *ids, size_t n_ids, size_t *len)
{
    size_t total = 0;
    for (size_t i = 0; i < n_ids; i++) {
        if (ids[i] < 0 || ids[i] >= tok->vocab_size)
            return NULL;
        total += tok->tok_len[ids[i]];
    }

    char *out = malloc(total + 1);
    if (!out)
        return NULL;
    char *p = out;
    for (size_t i = 0; i < n_ids; i++) {
        memcpy(p, tok->bytes + tok->tok_off[ids[i]], tok->tok_len[ids[i]]);
        p += tok->tok_len[ids[i]];
    }
    *p = '\0';
    if (len)
        *len = total;
    return out;
}

int tokenizer_vocab_size(const Tokenizer *tok)
{
    return tok->vocab_size;
}

int tokenizer_eot_id(const Tokenizer *tok)
{
    return tok->eot_id;
}

void tokenizer_free(Tokenizer *tok)
{
    if (!tok)
        return;
    free(tok->bytes);
    free(tok->tok_off);
    free(tok->tok_len);
    free(tok->merges);
    free(tok);
}

/* ---- Loading ------------------------------------------------------------ */

/*
 * The vocabulary files spell each token with every byte mapped to a printable
 * codepoint (bytes_to_unicode() in encoder.py): printable Latin-1 bytes stand
 * for themselves, the other 68 bytes become U+0100 onwards in byte order. A
 * space, for example, is written "Ġ" (U+0120). byte_of[] inverts that map.
 */
#define BYTE_ALPHABET_SIZE (256 + 68)

static void build_byte_decoder(int byte_of[BYTE_ALPHABET_SIZE])
{
    int next = 256;
    for (int b = 0; b < 256; b++) {
        int printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || b >= 0xAE;
        byte_of[b] = printable ? b : -1;
        if (!printable)
            byte_of[next++] = b;
    }
}

/* Maps one character of that alphabet back to its byte, or returns -1. */
static int alphabet_byte(const int *byte_of, uint32_t cp)
{
    return cp < BYTE_ALPHABET_SIZE ? byte_of[cp] : -1;
}

/* Decodes a UTF-8 token spelling in s[0..n) to raw bytes in out, which needs
 * room for n bytes. Returns the byte count, or -1 on a character outside the
 * alphabet. */
static long decode_spelling(const char *s, size_t n, const int *byte_of, char *out)
{
    const unsigned char *u = (const unsigned char *)s;
    long len = 0;
    for (size_t i = 0; i < n;) {
        uint32_t cp;
        size_t k = utf8_decode(u + i, n - i, &cp);
        int b = k > 0 ? alphabet_byte(byte_of, cp) : -1;
        if (b < 0)
            return -1;
        out[len++] = (char)b;
        i += k;
    }
    return len;
}

static char *read_file(const char *path, size_t *size)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "tokenizer: cannot open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    char *buf = NULL;
    long end = -1;
    if (fseek(f, 0, SEEK_END) == 0 && (end = ftell(f)) >= 0 && fseek(f, 0, SEEK_SET) == 0)
        buf = malloc((size_t)end + 1);
    if (!buf || fread(buf, 1, (size_t)end, f) != (size_t)end) {
        fprintf(stderr, "tokenizer: cannot read %s\n", path);
        free(buf);
        buf = NULL;
    } else {
        buf[end] = '\0';
        *size = (size_t)end;
    }
    fclose(f);
    return buf;
}

static void skip_ws(const char **p, const char *end)
{
    while (*p < end && (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r'))
        (*p)++;
}

static int hex_value(char c)
{
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    return -1;
}

/*
 * Parses the JSON string at *p, decoding its spelling to raw bytes appended
 * at out + *used. Each output byte consumes at least one input byte, so out
 * never needs more room than the JSON text itself.
 */
static bool parse_token_key(const char **p, const char *end, const int *byte_of, char *out,
                            size_t *used)
{
    const char *s = *p;
    if (s == end || *s++ != '"')
        return false;
    while (s < end && *s != '"') {
        uint32_t cp;
        if (*s == '\\') {
            if (end - s < 2)
                return false;
            char e = s[1];
            s += 2;
            if (e == '"' || e == '\\' || e == '/') {
                cp = (uint32_t)e;
            } else if (e == 'u' && end - s >= 4) {
                cp = 0;
                for (int k = 0; k < 4; k++) {
                    int h = hex_value(*s++);
                    if (h < 0)
                        return false;
                    cp = cp << 4 | (uint32_t)h;
                }
            } else {
                return false;
            }
        } else {
            size_t k = utf8_decode((const unsigned char *)s, (size_t)(end - s), &cp);
            if (k == 0)
                return false;
            s += k;
        }
        int b = alphabet_byte(byte_of, cp);
        if (b < 0)
            return false;
        out[(*used)++] = (char)b;
    }
    if (s == end)
        return false;
    *p = s + 1;
    return true;
}

static bool parse_id(const char **p, const char *end, int *id)
{
    const char *s = *p;
    long v = 0;
    if (s == end || *s < '0' || *s > '9')
        return false;
    while (s < end && *s >= '0' && *s <= '9') {
        v = v * 10 + (*s++ - '0');
        if (v > INT_MAX / 10)
            return false;
    }
    *id = (int)v;
    *p = s;
    return true;
}

typedef struct {
    uint32_t off, len;
    int id;
} Entry;

/* Reads encoder.json ({"token": id, ...}) into tok->bytes/tok_off/tok_len. */
static bool load_encoder(Tokenizer *tok, const char *path, const int *byte_of)
{
    size_t size;
    char *json = read_file(path, &size);
    if (!json)
        return false;

    const char *err = "malformed JSON";
    Entry *entries = NULL;
    size_t n = 0, cap = 0, used = 0;
    bool ok = false;
    const char *p = json, *end = json + size;

    if (size > UINT32_MAX || !(tok->bytes = malloc(size + 1))) {
        err = "out of memory";
        goto done;
    }
    skip_ws(&p, end);
    if (p == end || *p++ != '{')
        goto done;
    for (;;) {
        Entry e;
        e.off = (uint32_t)used;
        skip_ws(&p, end);
        if (!parse_token_key(&p, end, byte_of, tok->bytes, &used))
            goto done;
        e.len = (uint32_t)(used - e.off);
        skip_ws(&p, end);
        if (p == end || *p++ != ':')
            goto done;
        skip_ws(&p, end);
        if (!parse_id(&p, end, &e.id))
            goto done;
        if (e.len == 0) {
            err = "empty token";
            goto done;
        }
        if (n == cap) {
            cap = cap ? cap * 2 : 1024;
            Entry *grown = realloc(entries, cap * sizeof *entries);
            if (!grown) {
                err = "out of memory";
                goto done;
            }
            entries = grown;
        }
        entries[n++] = e;
        skip_ws(&p, end);
        if (p < end && *p == ',') {
            p++;
            continue;
        }
        if (p < end && *p == '}')
            break;
        goto done;
    }

    tok->vocab_size = (int)n;
    tok->tok_off = malloc(n * sizeof *tok->tok_off);
    tok->tok_len = calloc(n, sizeof *tok->tok_len);
    if (!tok->tok_off || !tok->tok_len) {
        err = "out of memory";
        goto done;
    }
    for (size_t i = 0; i < n; i++) {
        int id = entries[i].id;
        if ((size_t)id >= n || tok->tok_len[id] != 0) {
            err = "token ids are not exactly 0..N-1";
            goto done;
        }
        tok->tok_off[id] = entries[i].off;
        tok->tok_len[id] = entries[i].len;
    }
    ok = true;

done:
    if (!ok)
        fprintf(stderr, "tokenizer: %s: %s\n", path, err);
    free(entries);
    free(json);
    return ok;
}

static uint64_t hash_bytes(const char *s, size_t n)
{
    uint64_t h = 0xcbf29ce484222325ULL; /* FNV-1a */
    for (size_t i = 0; i < n; i++) {
        h ^= (unsigned char)s[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static size_t pow2_at_least(size_t n)
{
    size_t cap = 1;
    while (cap < n)
        cap *= 2;
    return cap;
}

/* Token bytes -> id, only needed while loading. */
typedef struct {
    int32_t *slots;  /* token id, or -1 if empty */
    size_t cap;      /* power of two */
} TokenIndex;

static int index_find(const TokenIndex *ix, const Tokenizer *tok, const char *s, size_t n)
{
    size_t mask = ix->cap - 1;
    for (size_t i = hash_bytes(s, n) & mask;; i = (i + 1) & mask) {
        int id = ix->slots[i];
        if (id < 0)
            return -1;
        if (tok->tok_len[id] == n && memcmp(tok->bytes + tok->tok_off[id], s, n) == 0)
            return id;
    }
}

static bool build_index(TokenIndex *ix, const Tokenizer *tok, const char *path)
{
    ix->cap = pow2_at_least(2 * (size_t)tok->vocab_size);
    ix->slots = malloc(ix->cap * sizeof *ix->slots);
    if (!ix->slots) {
        fprintf(stderr, "tokenizer: out of memory\n");
        return false;
    }
    memset(ix->slots, 0xff, ix->cap * sizeof *ix->slots);

    size_t mask = ix->cap - 1;
    for (int id = 0; id < tok->vocab_size; id++) {
        const char *s = tok->bytes + tok->tok_off[id];
        size_t n = tok->tok_len[id];
        if (index_find(ix, tok, s, n) >= 0) {
            fprintf(stderr, "tokenizer: %s: token %d is a duplicate\n", path, id);
            return false;
        }
        size_t i = hash_bytes(s, n) & mask;
        while (ix->slots[i] >= 0)
            i = (i + 1) & mask;
        ix->slots[i] = id;
    }
    return true;
}

/* Reads vocab.bpe: a "#version" header, then one "left right" merge per line,
 * highest priority first. */
static bool load_merges(Tokenizer *tok, const TokenIndex *ix, const char *path,
                        const int *byte_of)
{
    size_t size;
    char *text = read_file(path, &size);
    if (!text)
        return false;

    size_t lines = 1;
    for (size_t i = 0; i < size; i++)
        lines += text[i] == '\n';
    tok->merges_cap = pow2_at_least(2 * lines);
    tok->merges = malloc(tok->merges_cap * sizeof *tok->merges);
    char *buf = malloc(size + 1);
    if (!tok->merges || !buf) {
        fprintf(stderr, "tokenizer: out of memory\n");
        free(buf);
        free(text);
        return false;
    }
    for (size_t i = 0; i < tok->merges_cap; i++)
        tok->merges[i].merged = -1;

    bool ok = true;
    int32_t rank = 0;
    size_t line_no = 0;
    for (char *line = text; line < text + size; line_no++) {
        char *eol = memchr(line, '\n', (size_t)(text + size - line));
        if (!eol)
            eol = text + size;
        size_t len = (size_t)(eol - line);
        if (len > 0 && line[len - 1] == '\r')
            len--;
        char *next = eol + 1;

        if (len == 0 || (line_no == 0 && strncmp(line, "#version", 8) == 0)) {
            line = next;
            continue;
        }

        /* Spellings never contain a raw space (it is written "Ġ"). */
        char *space = memchr(line, ' ', len);
        long left_len = space ? decode_spelling(line, (size_t)(space - line), byte_of, buf) : -1;
        long right_len = left_len > 0 ? decode_spelling(space + 1, (size_t)(line + len - space - 1),
                                                        byte_of, buf + left_len)
                                      : -1;
        if (right_len <= 0) {
            fprintf(stderr, "tokenizer: %s:%zu: malformed merge rule\n", path, line_no + 1);
            ok = false;
            break;
        }

        int left = index_find(ix, tok, buf, (size_t)left_len);
        int right = index_find(ix, tok, buf + left_len, (size_t)right_len);
        int merged = index_find(ix, tok, buf, (size_t)(left_len + right_len));
        if (left < 0 || right < 0 || merged < 0) {
            fprintf(stderr, "tokenizer: %s:%zu: merge uses a token missing from the vocabulary\n",
                    path, line_no + 1);
            ok = false;
            break;
        }

        uint64_t key = pair_key(left, right);
        size_t mask = tok->merges_cap - 1;
        size_t i = mix64(key) & mask;
        while (tok->merges[i].merged >= 0 && tok->merges[i].key != key)
            i = (i + 1) & mask;
        if (tok->merges[i].merged < 0) { /* on a repeated pair the first rule wins */
            tok->merges[i].key = key;
            tok->merges[i].rank = rank;
            tok->merges[i].merged = merged;
        }
        rank++;
        line = next;
    }

    free(buf);
    free(text);
    return ok;
}

Tokenizer *tokenizer_load(const char *encoder_json_path, const char *vocab_bpe_path)
{
    int byte_of[BYTE_ALPHABET_SIZE];
    build_byte_decoder(byte_of);

    Tokenizer *tok = calloc(1, sizeof *tok);
    if (!tok) {
        fprintf(stderr, "tokenizer: out of memory\n");
        return NULL;
    }

    TokenIndex ix = {0};
    bool ok = load_encoder(tok, encoder_json_path, byte_of) &&
              build_index(&ix, tok, encoder_json_path) &&
              load_merges(tok, &ix, vocab_bpe_path, byte_of);

    for (int b = 0; ok && b < 256; b++) {
        char c = (char)b;
        tok->byte_ids[b] = index_find(&ix, tok, &c, 1);
        if (tok->byte_ids[b] < 0) {
            fprintf(stderr, "tokenizer: %s: no token for byte 0x%02x\n", encoder_json_path, b);
            ok = false;
        }
    }
    if (ok)
        tok->eot_id = index_find(&ix, tok, EOT_TEXT, EOT_TEXT_LEN);

    free(ix.slots);
    if (!ok) {
        tokenizer_free(tok);
        return NULL;
    }
    return tok;
}
