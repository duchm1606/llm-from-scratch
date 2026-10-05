#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "embedding.h"
#include "rng.h"
#include "test_util.h"

/* Full-precision expectations come from PyTorch 2.14 on CPU; the book prints
 * four decimals. */
#define TOL 1e-6f
#define BOOK_TOL 5e-5f

/* Section 2.7: torch.manual_seed(123); nn.Embedding(6, 3). */
static void check_small_table(void)
{
    static const float book_weight[] = {
        0.3374f,  -0.1778f, -0.1690f,
        0.9178f,  1.5810f,  1.3010f,
        1.2753f,  -0.2010f, -0.1606f,
        -0.4015f, 0.9666f,  -1.1481f,
        -1.1589f, 0.3255f,  -0.6315f,
        -2.8400f, -0.7849f, -1.4096f,
    };
    static const float torch_weight[] = {
        0.337370157f,  -0.177777216f, -0.168956161f, 0.917764008f,  1.5809691f,    1.30103993f,
        1.27530122f,   -0.200953096f, -0.160563797f, -0.401487917f, 0.966571569f,  -1.14814448f,
        -1.15886867f,  0.325471014f,  -0.63150543f,  -2.83999395f,  -0.784853339f, -1.40957236f,
    };
    float weight[6 * 3];
    Rng rng;
    rng_seed(&rng, 123);
    embedding_init(weight, 6, 3, &rng);
    CHECK(floats_close(weight, book_weight, 18, BOOK_TOL));
    CHECK(floats_close(weight, torch_weight, 18, TOL));

    /* embedding_layer(torch.tensor([3])) is row 3. */
    float out[4 * 3];
    embedding_forward(out, (int[]){3}, 1, weight, 3);
    CHECK(floats_close(out, (float[]){-0.4015f, 0.9666f, -1.1481f}, 3, BOOK_TOL));

    /* embedding_layer(torch.tensor([2, 3, 5, 1])) is rows 2, 3, 5, 1. */
    static const float book_lookup[] = {
        1.2753f,  -0.2010f, -0.1606f,
        -0.4015f, 0.9666f,  -1.1481f,
        -2.8400f, -0.7849f, -1.4096f,
        0.9178f,  1.5810f,  1.3010f,
    };
    embedding_forward(out, (int[]){2, 3, 5, 1}, 4, weight, 3);
    CHECK(floats_close(out, book_lookup, 12, BOOK_TOL));
}

enum { VOCAB = 50257, DIM = 256, B = 8, T = 4 };

/* Section 2.8: a [50257 x 256] token table, then a [4 x 256] position table
 * drawn from the same generator, applied to the batch the book prints. */
static void check_input_embeddings(void)
{
    static const int ids[B * T] = {
        40,    367,  2885, 1464,
        1807,  3619, 402,  271,
        10899, 2138, 257,  7026,
        15632, 438,  2016, 257,
        922,   5891, 1576, 438,
        568,   340,  373,  645,
        1049,  5975, 284,  502,
        284,   3285, 326,  11,
    };
    float *wte = malloc(sizeof(float) * VOCAB * DIM);
    float *wpe = malloc(sizeof(float) * T * DIM);
    float *out = malloc(sizeof(float) * B * T * DIM);
    float *tok = malloc(sizeof(float) * B * T * DIM);
    if (!wte || !wpe || !out || !tok) {
        fprintf(stderr, "FAIL out of memory\n");
        failures++;
        goto done;
    }

    Rng rng;
    rng_seed(&rng, 123);
    embedding_init(wte, VOCAB, DIM, &rng);
    embedding_init(wpe, T, DIM, &rng);

    /* Spot checks against nn.Embedding(50257, 256) and nn.Embedding(4, 256). */
    static const size_t wte_idx[] = {0, 1, 255, 256, 12345, 1000003, VOCAB * DIM - 1};
    static const float wte_want[] = {0.337370157f, -0.177777216f, -0.0521811359f, 0.238623038f,
                                     0.593931079f, -1.03467131f,  0.300565988f};
    for (size_t i = 0; i < sizeof wte_idx / sizeof wte_idx[0]; i++)
        CHECK(floats_close(&wte[wte_idx[i]], &wte_want[i], 1, TOL));
    CHECK(floats_close(wpe, (float[]){0.542322636f, -0.122386917f, -1.41496706f}, 3, TOL));
    CHECK(floats_close(&wpe[T * DIM - 1], (float[]){0.108031996f}, 1, TOL));

    /* input_embeddings = token_embeddings + pos_embeddings, shape [8, 4, 256]. */
    input_embedding_forward(out, ids, wte, wpe, B, T, DIM);
    CHECK(floats_close(out, (float[]){0.478358746f, 0.209353104f, -1.30799139f}, 3, TOL));
    CHECK(floats_close(&out[B * T * DIM - 3],
                       (float[]){0.941353261f, -1.79466558f, -0.209866703f}, 3, TOL));

    /* Every vector is exactly its token's row plus its position's row. */
    embedding_forward(tok, ids, B * T, wte, DIM);
    bool exact = true;
    for (size_t b = 0; b < B; b++)
        for (size_t t = 0; t < T; t++)
            for (size_t c = 0; c < DIM; c++) {
                size_t i = (b * T + t) * DIM + c;
                exact &= out[i] == tok[i] + wpe[t * DIM + c];
            }
    CHECK(exact);

    /* Token 257 sits at position 2 of row 2 and at position 3 of row 3: the
     * same token embedding, but different input embeddings. */
    size_t at_2 = (2 * T + 2) * DIM, at_3 = (3 * T + 3) * DIM;
    CHECK(ids[2 * T + 2] == 257 && ids[3 * T + 3] == 257);
    CHECK(memcmp(&tok[at_2], &tok[at_3], DIM * sizeof *tok) == 0);
    CHECK(memcmp(&out[at_2], &out[at_3], DIM * sizeof *out) != 0);

done:
    free(wte);
    free(wpe);
    free(out);
    free(tok);
}

int main(void)
{
    check_small_table();
    check_input_embeddings();
    return test_report("test_embedding");
}
