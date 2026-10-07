#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpt.h"
#include "rng.h"
#include "test_util.h"
#include "tokenizer.h"

/* Full-precision expectations come from PyTorch 2.14 on CPU, running the
 * book's classes; the book prints four decimals. */
#define TOL 1e-6f
#define BOOK_TOL 5e-5f
/* After twelve blocks GPT-2's logits are up to about 1.2e-6 from PyTorch's,
 * whose matrix products add in another order. */
#define DEEP_TOL 5e-6f

enum { GUARD = 16 };

#define SENTINEL -1234.5f

/* gpt_forward in scratch followed by GUARD sentinels, checking that nothing
 * wrote past the size it reports. */
static void run_gpt(float *logits, const int *ids, size_t B, size_t T, const GPTModel *model,
                    Rng *rng)
{
    size_t size = gpt_scratch_size(model, B, T);
    float *scratch = malloc(sizeof(float) * (size + GUARD));
    if (!scratch) {
        fprintf(stderr, "FAIL out of memory\n");
        failures++;
        return;
    }
    for (size_t i = 0; i < GUARD; i++)
        scratch[size + i] = SENTINEL;
    gpt_forward(logits, ids, B, T, model, scratch, rng);
    bool intact = true;
    for (size_t i = 0; i < GUARD; i++)
        intact &= scratch[size + i] == SENTINEL;
    CHECK(intact);
    free(scratch);
}

/* generate_text_simple in guarded scratch, as run_gpt. */
static size_t run_generate(int *ids, size_t n, size_t max_new_tokens, size_t context_size,
                           const GPTModel *model)
{
    size_t size = generate_scratch_size(model, context_size);
    float *scratch = malloc(sizeof(float) * (size + GUARD));
    if (!scratch) {
        fprintf(stderr, "FAIL out of memory\n");
        failures++;
        return 0;
    }
    for (size_t i = 0; i < GUARD; i++)
        scratch[size + i] = SENTINEL;
    n = generate_text_simple(ids, n, max_new_tokens, context_size, model, scratch);
    bool intact = true;
    for (size_t i = 0; i < GUARD; i++)
        intact &= scratch[size + i] == SENTINEL;
    CHECK(intact);
    free(scratch);
    return n;
}

/* Section 4.6 and exercise 4.2: parameter counts and sizes. */
static void check_sizes(void)
{
    const GPTConfig *cfg = &GPT_CONFIG_124M;
    size_t total = gpt_num_params(cfg);
    CHECK(total == 163009536);
    /* Without out_head, which has tok_emb's shape [50257 x 768]: the
     * weight-tied count of GPT-2. */
    CHECK(total - cfg->vocab_size * cfg->emb_dim == 124412160);
    char size[32];
    snprintf(size, sizeof size, "%.2f MB", (double)total * 4 / (1024 * 1024));
    CHECK(strcmp(size, "621.83 MB") == 0);

    /* GPT-2 medium, large and XL: the same code with wider embeddings,
     * more blocks and more heads. */
    static const struct {
        size_t emb_dim, n_layers, n_heads, total;
    } larger[] = {
        {1024, 24, 16, 406212608},
        {1280, 36, 20, 838220800},
        {1600, 48, 25, 1637792000},
    };
    for (size_t i = 0; i < sizeof larger / sizeof larger[0]; i++) {
        GPTConfig big = *cfg;
        big.emb_dim = larger[i].emb_dim;
        big.n_layers = larger[i].n_layers;
        big.n_heads = larger[i].n_heads;
        CHECK(gpt_num_params(&big) == larger[i].total);
    }
}

/* torch.manual_seed(123); GPTModel(cfg) with an 11-token vocabulary,
 * context_length 5, emb_dim 8, 2 heads, 2 layers, drop_rate 0.25 and
 * qkv_bias=True, on ids [[1, 4, 10, 0], [7, 7, 2, 9]]: small enough to
 * compare every logit, in training mode and in evaluation mode. */
static void check_small(void)
{
    enum { B = 2, T = 4, V = 11 };
    const GPTConfig cfg = {
        .vocab_size = V, .context_length = 5, .emb_dim = 8, .n_heads = 2, .n_layers = 2,
        .drop_rate = 0.25, .qkv_bias = true,
    };
    size_t n_params = gpt_num_params(&cfg);
    CHECK(n_params == 1976);
    float *params = malloc(sizeof(float) * n_params);
    if (!params) {
        fprintf(stderr, "FAIL out of memory\n");
        failures++;
        return;
    }
    GPTModel model;
    gpt_bind(&model, &cfg, params);
    CHECK(model.out_head + V * 8 == params + n_params);
    Rng rng;
    rng_seed(&rng, 123);
    gpt_init(&model, &rng);

    static const int ids[B * T] = {1, 4, 10, 0, 7, 7, 2, 9};
    static const float train[B * T * V] = {
        0.563320458f,   0.0415273905f, 0.188116997f,  -0.00218901038f, 0.554419041f,
        0.539460242f,   -0.322630405f, -0.142891452f, -0.676548779f,   0.579368889f,
        -0.26517415f,   1.33117366f,   -0.214440107f, 0.677580357f,    0.188243255f,
        0.649773836f,   -0.442713737f, 0.922113061f,  -0.218722433f,   0.487657845f,
        -0.152822629f,  0.364511847f,  -1.16290081f,  -0.147200555f,   -0.781935513f,
        0.333232641f,   -0.385734767f, 0.726651549f,  0.220672518f,    -0.231538236f,
        -0.572040319f,  -0.631820202f, 0.551943481f,  0.773631096f,    -0.29070428f,
        0.118041486f,   0.866428137f,  1.0924952f,    -0.612677813f,   0.526978016f,
        -0.548702478f,  -0.491910756f, -0.98071897f,  0.269017935f,    0.171234787f,
        -0.00612585247f, -0.0119715333f, -0.554735363f, -0.54112339f,  0.644245148f,
        0.488500476f,   0.269471347f,  0.11759568f,   1.01808953f,     0.536658347f,
        0.85428369f,    -0.330919504f, 1.14650488f,   0.322615802f,    0.441646636f,
        0.193028271f,   1.09157991f,   -0.744497895f, 0.68243897f,     -0.246560454f,
        -0.192997679f,  -0.276888043f, -0.529729843f, -0.40919286f,    1.22835219f,
        0.54156822f,    0.587222934f,  0.426121175f,  0.0342551172f,   -0.360826939f,
        -1.27362323f,   0.345183551f,  1.38644099f,   -0.14581728f,    1.13652813f,
        0.0248199254f,  0.703495026f,  0.811593294f,  0.277674794f,    0.0501240194f,
        0.763491392f,   0.386504978f,  -0.428369254f,
    };
    static const float eval[B * T * V] = {
        0.444551468f,   -0.123819351f, 0.243805304f,  0.29678756f,   0.533041239f,
        0.0335992575f,  0.0287309289f, -0.457049221f, -0.928166389f, 0.484806657f,
        -0.341232836f,  1.32333541f,   -0.403309137f, 0.487808943f,  0.504295468f,
        0.750216603f,   -0.085379228f, 1.05328798f,   -0.0423756242f, 0.197520465f,
        -0.143790215f,  0.522408903f,  -1.21168041f,  -0.11044994f,  -0.767590165f,
        0.249847263f,   -0.457798064f, 0.867184877f,  0.104702473f,  -0.129353195f,
        -0.462730259f,  -0.559819579f, 0.482295781f,  1.05833387f,   -0.51043272f,
        0.149976775f,   1.02173364f,   1.09750843f,   -0.283736855f, 0.938969076f,
        -0.334455669f,  -0.574559867f, -0.618388355f, 0.501015484f,  -0.276240379f,
        0.291473925f,   -0.467179239f, -0.64018333f,  -0.552733302f, 0.0376358628f,
        -0.725400507f,  0.724313855f,  -0.201510042f, 1.00426972f,   -0.02470419f,
        0.687966704f,   0.362117112f,  0.492015958f,  -0.996563554f, -0.341937691f,
        -0.298636794f,  -0.279162169f, 0.493354559f,  0.87119633f,   1.06429815f,
        -0.183944046f,  -0.614413083f, -0.382243395f, -0.568069816f, 0.96788168f,
        0.338923752f,   0.949404776f,  0.159297675f,  -0.0381423235f, -0.549509227f,
        -1.08098531f,   0.321971804f,  1.28644609f,   -0.251985312f, 0.896867394f,
        0.209183857f,   0.798314333f,  1.11242568f,   0.5625453f,    -0.177794442f,
        0.307652295f,   0.251794428f,  -0.0541437492f,
    };
    float logits[B * T * V];
    run_gpt(logits, ids, B, T, &model, &rng);
    CHECK(floats_close(logits, train, B * T * V, TOL));
    run_gpt(logits, ids, B, T, &model, NULL);
    CHECK(floats_close(logits, eval, B * T * V, TOL));

    /* generate_text_simple(model, [[1, 4, 10]], 6, context_size): with the
     * full context of 5 tokens the window slides once the sequence
     * outgrows it, and with 2 the model sees only the last two tokens. */
    int gen[3 + 6] = {1, 4, 10};
    CHECK(run_generate(gen, 3, 6, 5, &model) == 9);
    CHECK(memcmp(gen, (int[]){1, 4, 10, 5, 0, 4, 3, 2, 3}, sizeof gen) == 0);
    CHECK(run_generate(gen, 3, 6, 2, &model) == 9);
    CHECK(memcmp(gen, (int[]){1, 4, 10, 6, 5, 0, 0, 0, 0}, sizeof gen) == 0);
    free(params);
}

/* Section 4.7 on section 4.6's model: model.eval(), then six greedy tokens
 * after "Hello, I am". Untrained, the model continues with gibberish. */
static void check_generate(const GPTModel *model, const Tokenizer *tok)
{
    enum { NEW = 6 };
    static const char start_context[] = "Hello, I am";
    size_t n;
    int *encoded = tokenizer_encode(tok, start_context, strlen(start_context), false, &n);
    CHECK(encoded && n == 4);
    if (!encoded || n != 4) {
        free(encoded);
        return;
    }
    int ids[4 + NEW];
    memcpy(ids, encoded, sizeof(int) * n);
    free(encoded);
    CHECK(memcmp(ids, (int[]){15496, 11, 314, 716}, sizeof(int) * 4) == 0);

    n = run_generate(ids, n, NEW, model->cfg.context_length, model);
    CHECK(n == 10);
    CHECK(memcmp(ids, (int[]){15496, 11, 314, 716, 27018, 24086, 47843, 30961, 42348, 7267},
                 sizeof ids) == 0);
    char *text = tokenizer_decode(tok, ids, n, NULL);
    CHECK(text && strcmp(text, "Hello, I am Featureiman Byeswickattribute argue") == 0);
    free(text);
}

/* Sections 4.1 and 4.6: torch.manual_seed(123); model =
 * GPTModel(GPT_CONFIG_124M); out = model(batch), in training mode, on
 * "Every effort moves you" and "Every day holds a"; then section 4.7 with
 * the same model. */
static void check_gpt2_small(void)
{
    enum { B = 2, T = 4 };
    const GPTConfig *cfg = &GPT_CONFIG_124M;
    size_t V = cfg->vocab_size;
    float *params = malloc(sizeof(float) * gpt_num_params(cfg));
    float *logits = malloc(sizeof(float) * B * T * V);
    Tokenizer *tok = tokenizer_load(ENCODER_JSON, VOCAB_BPE);
    if (!params || !logits || !tok) {
        fprintf(stderr, "FAIL out of memory or tokenizer files missing\n");
        failures++;
        goto done;
    }

    static const char *texts[B] = {"Every effort moves you", "Every day holds a"};
    int batch[B * T];
    for (size_t b = 0; b < B; b++) {
        size_t n;
        int *ids = tokenizer_encode(tok, texts[b], strlen(texts[b]), false, &n);
        CHECK(ids && n == T);
        if (ids && n == T)
            memcpy(&batch[b * T], ids, sizeof(int) * T);
        free(ids);
    }
    CHECK(memcmp(batch, (int[]){6109, 3626, 6100, 345, 6109, 1110, 6622, 257}, sizeof batch) == 0);

    GPTModel model;
    gpt_bind(&model, cfg, params);
    Rng rng;
    rng_seed(&rng, 123);
    gpt_init(&model, &rng);
    CHECK(floats_close(model.tok_emb, (float[]){0.337370157f, -0.177777216f, -0.303527534f}, 3,
                       TOL));
    CHECK(floats_close(&model.tok_emb[V * cfg->emb_dim - 3],
                       (float[]){-0.324433714f, 0.260553986f, 0.588852406f}, 3, TOL));
    CHECK(floats_close(model.pos_emb, (float[]){0.876873314f, 0.255025357f, 0.844123423f}, 3,
                       TOL));
    CHECK(floats_close(model.out_head, (float[]){0.0275941323f, -0.0321561359f, 0.0332383178f}, 3,
                       TOL));
    CHECK(floats_close(&model.out_head[V * cfg->emb_dim - 3],
                       (float[]){0.00635148492f, -0.005744569f, -0.012392085f}, 3, TOL));

    /* The first and last three logits of each token, as the book prints
     * them, then at full precision. */
    static const float book[B * T][6] = {
        {0.3613f, 0.4222f, -0.0711f, 0.3483f, 0.4661f, -0.2838f},
        {-0.1792f, -0.5660f, -0.9485f, 0.0477f, 0.5181f, -0.3168f},
        {0.7120f, 0.0332f, 0.1085f, 0.1018f, -0.4327f, -0.2553f},
        {-1.0076f, 0.3418f, -0.1190f, 0.7195f, 0.4023f, 0.0532f},
        {-0.2564f, 0.0900f, 0.0335f, 0.2659f, 0.4454f, -0.6806f},
        {0.1230f, 0.3653f, -0.2074f, 0.7705f, 0.2710f, 0.2246f},
        {1.0558f, 1.0318f, -0.2800f, 0.6936f, 0.3205f, -0.3178f},
        {-0.1565f, 0.3926f, 0.3288f, 1.2630f, -0.1858f, 0.0388f},
    };
    static const float torch[B * T][6] = {
        {0.361261249f, 0.422249347f, -0.0711290538f, 0.34830007f, 0.466112733f, -0.283778638f},
        {-0.179168329f, -0.566034913f, -0.948509037f, 0.0477248281f, 0.518086791f, -0.316802084f},
        {0.711966991f, 0.0331927948f, 0.108504578f, 0.101766609f, -0.432741642f, -0.255287915f},
        {-1.00756419f, 0.341829062f, -0.119012408f, 0.7195158f, 0.402290434f, 0.0532381609f},
        {-0.256404757f, 0.0900263488f, 0.033531256f, 0.265916228f, 0.445382684f, -0.680560231f},
        {0.122963622f, 0.365285665f, -0.207367644f, 0.770489216f, 0.271047384f, 0.224576145f},
        {1.05577528f, 1.03175569f, -0.279975921f, 0.693619013f, 0.320514292f, -0.31777668f},
        {-0.156516805f, 0.392635584f, 0.328826487f, 1.26301968f, -0.185847282f, 0.0387613587f},
    };
    run_gpt(logits, batch, B, T, &model, &rng);
    for (size_t t = 0; t < B * T; t++) {
        const float *row = logits + t * V;
        float ends[6] = {row[0], row[1], row[2], row[V - 3], row[V - 2], row[V - 1]};
        CHECK(floats_close(ends, book[t], 6, BOOK_TOL));
        CHECK(floats_close(ends, torch[t], 6, DEEP_TOL));
    }

    /* model.eval() */
    run_gpt(logits, batch, B, T, &model, NULL);
    CHECK(floats_close(logits, (float[]){0.0641638264f, 0.20442605f, -0.169454202f}, 3,
                       DEEP_TOL));
    CHECK(floats_close(&logits[B * T * V - 3],
                       (float[]){1.19152117f, -0.164312929f, 0.0370460413f}, 3, DEEP_TOL));

    check_generate(&model, tok);

done:
    tokenizer_free(tok);
    free(params);
    free(logits);
}

int main(void)
{
    check_sizes();
    check_small();
    check_gpt2_small();
    return test_report("test_gpt");
}
