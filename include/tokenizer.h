#ifndef LLM_TOKENIZER_H
#define LLM_TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>

/*
 * GPT-2 byte-level BPE tokenizer: the same encoding as
 * tiktoken.get_encoding("gpt2") used throughout the book (chapter 2.5 on).
 *
 * The vocabulary comes from the two files OpenAI ships with the GPT-2 weights:
 *   encoder.json  token string -> id (50,257 entries)
 *   vocab.bpe     the 50,000 merge rules, highest priority first
 */
typedef struct Tokenizer Tokenizer;

/* Returns NULL on failure, after printing the reason to stderr. */
Tokenizer *tokenizer_load(const char *encoder_json_path, const char *vocab_bpe_path);
void tokenizer_free(Tokenizer *tok);

int tokenizer_vocab_size(const Tokenizer *tok);

/* Id of <|endoftext|> (50256 for GPT-2), or -1 if the vocabulary lacks it. */
int tokenizer_eot_id(const Tokenizer *tok);

/*
 * Encodes len bytes of UTF-8 text. Returns a malloc'd array of *n_ids token
 * ids that the caller frees, or NULL if memory runs out.
 *
 * allow_special = true turns the literal text "<|endoftext|>" into the EOT id,
 * like tiktoken's encode(text, allowed_special={"<|endoftext|>"}). With false
 * it is encoded as ordinary text, like disallowed_special=().
 *
 * Invalid UTF-8 bytes are tokenized as punctuation-like characters, so
 * decode(encode(x)) == x holds for any byte string.
 */
int *tokenizer_encode(const Tokenizer *tok, const char *text, size_t len,
                      bool allow_special, size_t *n_ids);

/*
 * Concatenates the bytes of each token. Returns a malloc'd, NUL-terminated
 * buffer that the caller frees and stores its length (excluding the NUL) in
 * *len unless len is NULL. Returns NULL if an id is out of range or memory
 * runs out.
 */
char *tokenizer_decode(const Tokenizer *tok, const int *ids, size_t n_ids, size_t *len);

#endif
