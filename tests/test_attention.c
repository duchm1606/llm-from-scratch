#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "attention.h"
#include "ch03_inputs.h"
#include "linear.h"
#include "ops.h"
#include "rng.h"
#include "test_util.h"

/* Full-precision expectations come from PyTorch 2.14 on CPU; the book prints
 * four decimals. */
#define TOL 1e-6f
#define BOOK_TOL 5e-5f

enum { D_OUT = 2, GUARD = 16 };

#define SENTINEL -1234.5f

/* Scratch of n floats followed by GUARD sentinels, or NULL. */
static float *guarded_scratch(size_t n)
{
    float *scratch = malloc(sizeof(float) * (n + GUARD));
    if (!scratch) {
        fprintf(stderr, "FAIL out of memory\n");
        failures++;
        return NULL;
    }
    for (size_t i = 0; i < GUARD; i++)
        scratch[n + i] = SENTINEL;
    return scratch;
}

/* Checks that nothing wrote past the n floats asked for, then frees. */
static void release_scratch(float *scratch, size_t n)
{
    bool intact = true;
    for (size_t i = 0; i < GUARD; i++)
        intact &= scratch[n + i] == SENTINEL;
    CHECK(intact);
    free(scratch);
}

/* self_attention_forward on B sequences of T tokens, in guarded scratch. */
static void run_self_attention(float *out, const float *x, size_t B, const SelfAttention *sa,
                               Rng *rng)
{
    size_t n = self_attention_scratch_size(sa, T);
    float *scratch = guarded_scratch(n);
    if (!scratch)
        return;
    self_attention_forward(out, x, B, T, sa, scratch, rng);
    release_scratch(scratch, n);
}

/* multi_head_attention_forward on B sequences of t tokens, in guarded
 * scratch. */
static void run_multi_head(float *out, const float *x, size_t B, size_t t,
                           const MultiHeadAttention *mha, Rng *rng)
{
    size_t n = multi_head_attention_scratch_size(mha, t);
    float *scratch = guarded_scratch(n);
    if (!scratch)
        return;
    multi_head_attention_forward(out, x, B, t, mha, scratch, rng);
    release_scratch(scratch, n);
}

/* Sections 3.4.1-3.4.2: torch.manual_seed(123), then three torch.rand(3, 2)
 * matrices, the weights of SelfAttention_v1. They are [d_in x d_out], the
 * transpose of nn.Linear's layout, so projecting is x @ W. */
static void check_v1(void)
{
    float w_query[D_IN * D_OUT], w_key[D_IN * D_OUT], w_value[D_IN * D_OUT];
    Rng rng;
    rng_seed(&rng, 123);
    rng_uniform(&rng, w_query, D_IN * D_OUT, 0, 1);
    rng_uniform(&rng, w_key, D_IN * D_OUT, 0, 1);
    rng_uniform(&rng, w_value, D_IN * D_OUT, 0, 1);
    CHECK(floats_close(w_query,
                       (float[]){0.296111941f, 0.516562283f, 0.251670718f, 0.68855679f,
                                 0.0739724636f, 0.866521955f},
                       D_IN * D_OUT, TOL));
    CHECK(floats_close(w_key,
                       (float[]){0.136579871f, 0.102479041f, 0.184056461f, 0.726446748f,
                                 0.315253913f, 0.687106669f},
                       D_IN * D_OUT, TOL));
    CHECK(floats_close(w_value,
                       (float[]){0.075635314f, 0.196638167f, 0.316411972f, 0.401740134f,
                                 0.118568301f, 0.82739538f},
                       D_IN * D_OUT, TOL));

    /* query_2 = x_2 @ W_query */
    float query_2[D_OUT];
    matmul(query_2, &inputs[1 * D_IN], w_query, 1, D_IN, D_OUT);
    CHECK(floats_close(query_2, (float[]){0.4306f, 1.4551f}, D_OUT, BOOK_TOL));

    float queries[T * D_OUT], keys[T * D_OUT], values[T * D_OUT];
    matmul(queries, inputs, w_query, T, D_IN, D_OUT);
    matmul(keys, inputs, w_key, T, D_IN, D_OUT);
    matmul(values, inputs, w_value, T, D_IN, D_OUT);

    /* attn_scores_2 = query_2 @ keys.T, whose second entry is attn_score_22 */
    float scores_2[T];
    matmul_bt(scores_2, query_2, keys, 1, D_OUT, T);
    CHECK(floats_close(&scores_2[1], (float[]){1.8524f}, 1, BOOK_TOL));
    CHECK(floats_close(scores_2, (float[]){1.2705f, 1.8524f, 1.8111f, 1.0795f, 0.5577f, 1.5440f},
                       T, BOOK_TOL));
    CHECK(floats_close(scores_2,
                       (float[]){1.27048302f, 1.85238445f, 1.81110668f, 1.07951725f,
                                 0.557730556f, 1.54397058f},
                       T, TOL));

    /* attn_weights_2 and context_vec_2 are row 2 of attention over all
     * queries at once. */
    float weights[T * T], context[T * D_OUT];
    attention_forward(context, weights, queries, keys, values, T, D_OUT, D_OUT, false, 0, NULL);
    CHECK(floats_close(&weights[1 * T],
                       (float[]){0.1500f, 0.2264f, 0.2199f, 0.1311f, 0.0906f, 0.1820f}, T,
                       BOOK_TOL));
    CHECK(floats_close(&weights[1 * T],
                       (float[]){0.150019512f, 0.22638379f, 0.21987164f, 0.131070077f,
                                 0.0906289071f, 0.182026118f},
                       T, TOL));
    CHECK(floats_close(&context[1 * D_OUT], (float[]){0.3061f, 0.8210f}, D_OUT, BOOK_TOL));

    /* sa_v1(inputs) */
    static const float book_v1[T * D_OUT] = {
        0.2996f, 0.8053f,
        0.3061f, 0.8210f,
        0.3058f, 0.8203f,
        0.2948f, 0.7939f,
        0.2927f, 0.7891f,
        0.2990f, 0.8040f,
    };
    static const float torch_v1[T * D_OUT] = {
        0.299582064f, 0.805314124f,
        0.306100219f, 0.821030378f,
        0.305781096f, 0.820295751f,
        0.294765949f, 0.793866336f,
        0.292706072f, 0.789084315f,
        0.299010098f, 0.804036856f,
    };
    CHECK(floats_close(context, book_v1, T * D_OUT, BOOK_TOL));
    CHECK(floats_close(context, torch_v1, T * D_OUT, TOL));
}

/* Section 3.4.2: torch.manual_seed(789); SelfAttention_v2(3, 2). */
static void check_v2(void)
{
    float w[3][D_OUT * D_IN];
    SelfAttention sa = {.d_in = D_IN, .d_out = D_OUT, .w_query = w[0], .w_key = w[1], .w_value = w[2]};
    Rng rng;
    rng_seed(&rng, 789);
    self_attention_init(&sa, &rng);

    static const float book_v2[T * D_OUT] = {
        -0.0739f, 0.0713f,
        -0.0748f, 0.0703f,
        -0.0749f, 0.0702f,
        -0.0760f, 0.0685f,
        -0.0763f, 0.0679f,
        -0.0754f, 0.0693f,
    };
    static const float torch_v2[T * D_OUT] = {
        -0.0738902614f, 0.0712899044f,
        -0.0748107135f, 0.0703092963f,
        -0.0748561993f, 0.0702416599f,
        -0.0760016292f, 0.0684501082f,
        -0.0763275996f, 0.0679428056f,
        -0.0754442811f, 0.0693049207f,
    };
    float out[T * D_OUT];
    run_self_attention(out, inputs, 1, &sa, NULL);
    CHECK(floats_close(out, book_v2, T * D_OUT, BOOK_TOL));
    CHECK(floats_close(out, torch_v2, T * D_OUT, TOL));

    /* Exercise 3.1: SelfAttention_v1 given the transposes of these weights
     * computes the same context vectors. */
    float w_t[3][D_IN * D_OUT];
    for (size_t m = 0; m < 3; m++)
        for (size_t i = 0; i < D_OUT; i++)
            for (size_t j = 0; j < D_IN; j++)
                w_t[m][j * D_OUT + i] = w[m][i * D_IN + j];
    float queries[T * D_OUT], keys[T * D_OUT], values[T * D_OUT];
    matmul(queries, inputs, w_t[0], T, D_IN, D_OUT);
    matmul(keys, inputs, w_t[1], T, D_IN, D_OUT);
    matmul(values, inputs, w_t[2], T, D_IN, D_OUT);
    float weights[T * T], v1_out[T * D_OUT];
    attention_forward(v1_out, weights, queries, keys, values, T, D_OUT, D_OUT, false, 0, NULL);
    CHECK(floats_close(v1_out, out, T * D_OUT, TOL));
}

/* torch.manual_seed(123); SelfAttention_v2(3, 2, qkv_bias=True): each
 * projection draws its weight, then its bias. */
static void check_qkv_bias(void)
{
    float w[3][D_OUT * D_IN], b[3][D_OUT];
    SelfAttention sa = {
        .d_in = D_IN, .d_out = D_OUT,
        .w_query = w[0], .w_key = w[1], .w_value = w[2],
        .b_query = b[0], .b_key = b[1], .b_value = b[2],
    };
    Rng rng;
    rng_seed(&rng, 123);
    self_attention_init(&sa, &rng);
    CHECK(floats_close(b[0], (float[]){-0.419641405f, -0.459017664f}, D_OUT, TOL));
    CHECK(floats_close(b[2], (float[]){0.157876849f, 0.557292342f}, D_OUT, TOL));

    static const float want[T * D_OUT] = {
        0.10589961f,  0.929572642f,
        0.114426538f, 0.935321629f,
        0.114333838f, 0.935251594f,
        0.118122526f, 0.936930597f,
        0.113788687f, 0.934276819f,
        0.118761346f, 0.937503457f,
    };
    float out[T * D_OUT];
    run_self_attention(out, inputs, 1, &sa, NULL);
    CHECK(floats_close(out, want, T * D_OUT, TOL));
}

/* Queries, keys and values for x from SelfAttention_v2 at seed 789, which
 * section 3.5 reuses. */
static void v2_projections(float *queries, float *keys, float *values, const float *x)
{
    float w[3][D_OUT * D_IN];
    SelfAttention sa = {.d_in = D_IN, .d_out = D_OUT, .w_query = w[0], .w_key = w[1], .w_value = w[2]};
    Rng rng;
    rng_seed(&rng, 789);
    self_attention_init(&sa, &rng);
    linear_forward(queries, x, T, sa.w_query, NULL, D_IN, D_OUT);
    linear_forward(keys, x, T, sa.w_key, NULL, D_IN, D_OUT);
    linear_forward(values, x, T, sa.w_value, NULL, D_IN, D_OUT);
}

/* Causal attention weights at seed 789 (section 3.5.1). */
static const float book_causal[T * T] = {
    1.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.5517f, 0.4483f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.3800f, 0.3097f, 0.3103f, 0.0000f, 0.0000f, 0.0000f,
    0.2758f, 0.2460f, 0.2462f, 0.2319f, 0.0000f, 0.0000f,
    0.2175f, 0.1983f, 0.1984f, 0.1888f, 0.1971f, 0.0000f,
    0.1935f, 0.1663f, 0.1666f, 0.1542f, 0.1666f, 0.1529f,
};
static const float torch_causal[T * T] = {
    1,            0,            0,            0,            0,            0,
    0.551678061f, 0.448321968f, 0,            0,            0,            0,
    0.379967242f, 0.309713542f, 0.310319275f, 0,            0,            0,
    0.275842845f, 0.246028453f, 0.246247143f, 0.231881544f, 0,            0,
    0.217515364f, 0.198280916f, 0.198397934f, 0.188752919f, 0.197052807f, 0,
    0.193472371f, 0.16633299f,  0.166568086f, 0.154186234f, 0.166560829f, 0.152879536f,
};

/* Section 3.5.1: masking SelfAttention_v2's attention weights. */
static void check_causal(void)
{
    float queries[T * D_OUT], keys[T * D_OUT], values[T * D_OUT];
    v2_projections(queries, keys, values, inputs);

    /* attn_weights before masking */
    static const float book_weights[T * T] = {
        0.1921f, 0.1646f, 0.1652f, 0.1550f, 0.1721f, 0.1510f,
        0.2041f, 0.1659f, 0.1662f, 0.1496f, 0.1665f, 0.1477f,
        0.2036f, 0.1659f, 0.1662f, 0.1498f, 0.1664f, 0.1480f,
        0.1869f, 0.1667f, 0.1668f, 0.1571f, 0.1661f, 0.1564f,
        0.1830f, 0.1669f, 0.1670f, 0.1588f, 0.1658f, 0.1585f,
        0.1935f, 0.1663f, 0.1666f, 0.1542f, 0.1666f, 0.1529f,
    };
    float weights[T * T], context[T * D_OUT];
    attention_forward(context, weights, queries, keys, values, T, D_OUT, D_OUT, false, 0, NULL);
    CHECK(floats_close(weights, book_weights, T * T, BOOK_TOL));
    float unmasked_last[D_OUT];
    memcpy(unmasked_last, &context[(T - 1) * D_OUT], sizeof unmasked_last);

    /* masked_simple_norm: zero the weights above the diagonal, then
     * renormalize each row. */
    float renormalized[T * T];
    for (size_t i = 0; i < T; i++) {
        float sum = 0;
        for (size_t j = 0; j < T; j++) {
            renormalized[i * T + j] = j <= i ? weights[i * T + j] : 0;
            sum += renormalized[i * T + j];
        }
        for (size_t j = 0; j < T; j++)
            renormalized[i * T + j] /= sum;
    }
    CHECK(floats_close(renormalized, book_causal, T * T, BOOK_TOL));

    /* The efficient way: -inf scores above the diagonal, then the softmax.
     * Both ways agree, so the masked tokens leak nothing. */
    attention_forward(context, weights, queries, keys, values, T, D_OUT, D_OUT, true, 0, NULL);
    CHECK(floats_close(weights, book_causal, T * T, BOOK_TOL));
    CHECK(floats_close(weights, torch_causal, T * T, TOL));
    CHECK(floats_close(weights, renormalized, T * T, TOL));

    /* context_vec = attn_weights @ values */
    static const float torch_context[T * D_OUT] = {
        -0.0872180834f, 0.0285899844f,
        -0.0990691483f, 0.0500948392f,
        -0.0999450088f, 0.0633498728f,
        -0.0982549042f, 0.0489481352f,
        -0.0514459163f, 0.109843709f,
        -0.0754442811f, 0.0693049207f,
    };
    CHECK(floats_close(context, torch_context, T * D_OUT, TOL));

    /* The last token sees every token, masked or not. */
    CHECK(memcmp(&context[(T - 1) * D_OUT], unmasked_last, sizeof unmasked_last) == 0);

    /* Changing the last token changes its own context vector and no other. */
    float changed[T * D_IN], changed_context[T * D_OUT];
    memcpy(changed, inputs, sizeof changed);
    for (size_t c = 0; c < D_IN; c++)
        changed[(T - 1) * D_IN + c] += 1;
    v2_projections(queries, keys, values, changed);
    attention_forward(changed_context, weights, queries, keys, values, T, D_OUT, D_OUT, true, 0,
                      NULL);
    CHECK(memcmp(changed_context, context, (T - 1) * D_OUT * sizeof(float)) == 0);
    CHECK(memcmp(&changed_context[(T - 1) * D_OUT], &context[(T - 1) * D_OUT],
                 D_OUT * sizeof(float)) != 0);
}

/* Section 3.5.2: torch.manual_seed(123), then nn.Dropout(0.5) on the causal
 * attention weights. */
static void check_dropout(void)
{
    float queries[T * D_OUT], keys[T * D_OUT], values[T * D_OUT];
    v2_projections(queries, keys, values, inputs);

    static const float book_dropped[T * T] = {
        2.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
        0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
        0.7599f, 0.6194f, 0.6206f, 0.0000f, 0.0000f, 0.0000f,
        0.0000f, 0.4921f, 0.4925f, 0.0000f, 0.0000f, 0.0000f,
        0.0000f, 0.3966f, 0.0000f, 0.3775f, 0.0000f, 0.0000f,
        0.0000f, 0.3327f, 0.3331f, 0.3084f, 0.3331f, 0.0000f,
    };
    static const float torch_dropped[T * T] = {
        2, 0,            0,            0,            0,            0,
        0, 0,            0,            0,            0,            0,
        0.759934485f, 0.619427085f, 0.620638549f, 0, 0,            0,
        0, 0.492056906f, 0.492494285f, 0,            0,            0,
        0, 0.396561831f, 0,            0.377505839f, 0,            0,
        0, 0.33266598f,  0.333136171f, 0.308372468f, 0.333121657f, 0,
    };
    static const float torch_context[T * D_OUT] = {
        -0.174436167f,  0.0571799688f,
        0,              0,
        -0.199890018f,  0.126699746f,
        -0.106082864f,  0.083339341f,
        -0.0795107409f, 0.0293701198f,
        -0.0534033291f, 0.174804777f,
    };
    float weights[T * T], context[T * D_OUT];
    Rng rng;
    rng_seed(&rng, 123);
    attention_forward(context, weights, queries, keys, values, T, D_OUT, D_OUT, true, 0.5, &rng);
    CHECK(floats_close(weights, book_dropped, T * T, BOOK_TOL));
    CHECK(floats_close(weights, torch_dropped, T * T, TOL));
    CHECK(floats_close(context, torch_context, T * D_OUT, TOL));

    /* A NULL rng is evaluation mode: no dropout, whatever the rate. */
    attention_forward(context, weights, queries, keys, values, T, D_OUT, D_OUT, true, 0.5, NULL);
    CHECK(floats_close(weights, torch_causal, T * T, TOL));
}

enum { B = 2 };

/* Section 3.5.3: batch = torch.stack((inputs, inputs)). */
static void book_batch(float *batch)
{
    memcpy(batch, inputs, sizeof inputs);
    memcpy(batch + T * D_IN, inputs, sizeof inputs);
}

/* Two different sequences: the inputs, then the inputs with their tokens
 * reversed. */
static void mixed_batch(float *batch)
{
    memcpy(batch, inputs, sizeof inputs);
    for (size_t t = 0; t < T; t++)
        memcpy(&batch[(T + t) * D_IN], &inputs[(T - 1 - t) * D_IN], D_IN * sizeof(float));
}

/* Section 3.5.3: torch.manual_seed(123), then CausalAttention(3, 2, 6,
 * dropout) on the book's batch. */
static void check_causal_attention(void)
{
    float batch[B * T * D_IN];
    book_batch(batch);

    float w[3][D_OUT * D_IN];
    SelfAttention ca = {
        .d_in = D_IN, .d_out = D_OUT,
        .w_query = w[0], .w_key = w[1], .w_value = w[2],
        .causal = true,
    };
    Rng rng;
    rng_seed(&rng, 123);
    self_attention_init(&ca, &rng);

    /* dropout = 0.0: context_vecs has shape [2, 6, 2]. */
    static const float want[T * D_OUT] = {
        -0.451920271f, 0.221604794f,
        -0.587435067f, 0.00577611756f,
        -0.630023062f, -0.0631825998f,
        -0.567456722f, -0.0842531398f,
        -0.55256182f,  -0.0980682001f,
        -0.529900968f, -0.108067632f,
    };
    float out[B * T * D_OUT];
    run_self_attention(out, batch, B, &ca, &rng);
    CHECK(floats_close(out, want, T * D_OUT, TOL));
    /* Both sequences are the same text, so they get the same context vectors. */
    CHECK(memcmp(&out[T * D_OUT], out, T * D_OUT * sizeof(float)) == 0);

    /* dropout = 0.5 in training mode: dropout continues drawing from the
     * generator where initialization left it, and the two sequences get
     * different masks. */
    static const float want_dropout[B * T * D_OUT] = {
        -0.903840542f, 0.443209589f,
        -0.43679893f,  0.214189857f,
        -0.484927803f, -0.134101912f,
        -0.583358884f, 0.00813283492f,
        -0.621864796f, -0.0526335314f,
        -0.141713083f, -0.0504860617f,

        0,             0,
        -1.17487013f,  0.0115522351f,
        -0.773255587f, 0.00728325034f,
        -0.913953245f, -0.27685684f,
        -0.767860651f, -0.0735368133f,
        -0.674854577f, -0.0983852297f,
    };
    ca.dropout = 0.5;
    rng_seed(&rng, 123);
    self_attention_init(&ca, &rng);
    run_self_attention(out, batch, B, &ca, &rng);
    CHECK(floats_close(out, want_dropout, B * T * D_OUT, TOL));

    /* ca.eval(): without dropout the rate makes no difference. */
    run_self_attention(out, batch, B, &ca, NULL);
    CHECK(floats_close(out, want, T * D_OUT, TOL));
    CHECK(floats_close(&out[T * D_OUT], want, T * D_OUT, TOL));

    /* With different sequences, each gets what it would get on its own. */
    float mixed[B * T * D_IN], alone[T * D_OUT];
    mixed_batch(mixed);
    run_self_attention(out, mixed, B, &ca, NULL);
    for (size_t b = 0; b < B; b++) {
        run_self_attention(alone, &mixed[b * T * D_IN], 1, &ca, NULL);
        CHECK(memcmp(&out[b * T * D_OUT], alone, sizeof alone) == 0);
    }
}

/* MultiHeadAttentionWrapper (section 3.6.1), built from the pieces above:
 * each head runs over the whole batch in turn, and the heads' outputs are
 * concatenated along the last dimension. */
static void run_wrapper(float *out, const float *x, size_t B, const SelfAttention *heads,
                        size_t num_heads, Rng *rng)
{
    size_t d = heads[0].d_out, width = num_heads * d;
    float *head_out = malloc(sizeof(float) * B * T * d);
    if (!head_out) {
        fprintf(stderr, "FAIL out of memory\n");
        failures++;
        return;
    }
    for (size_t h = 0; h < num_heads; h++) {
        run_self_attention(head_out, x, B, &heads[h], rng);
        for (size_t row = 0; row < B * T; row++)
            memcpy(&out[row * width + h * d], &head_out[row * d], d * sizeof(float));
    }
    free(head_out);
}

enum { HEADS = 2 };

/* Section 3.6.1: torch.manual_seed(123), then
 * MultiHeadAttentionWrapper(3, d_out, 6, 0.0, num_heads=2), whose heads are
 * CausalAttention(3, d_out, 6, 0.0) initialized one after the other. */
static void check_wrapper(void)
{
    float batch[B * T * D_IN];
    book_batch(batch);
    float w[HEADS][3][D_OUT * D_IN];
    SelfAttention heads[HEADS];
    Rng rng;

    /* d_out = 2: context_vecs has shape [2, 6, 4], the first head's columns
     * first. */
    static const float book[T * HEADS * D_OUT] = {
        -0.4519f, 0.2216f,  0.4772f, 0.1063f,
        -0.5874f, 0.0058f,  0.5891f, 0.3257f,
        -0.6300f, -0.0632f, 0.6202f, 0.3860f,
        -0.5675f, -0.0843f, 0.5478f, 0.3589f,
        -0.5526f, -0.0981f, 0.5321f, 0.3428f,
        -0.5299f, -0.1081f, 0.5077f, 0.3493f,
    };
    static const float torch[T * HEADS * D_OUT] = {
        -0.451920271f, 0.221604794f,   0.477195919f, 0.106347829f,
        -0.587435067f, 0.00577611756f, 0.589073062f, 0.325660497f,
        -0.630023062f, -0.0631825998f, 0.620195925f, 0.385963023f,
        -0.567456722f, -0.0842531398f, 0.547769308f, 0.358924598f,
        -0.55256182f,  -0.0980682001f, 0.532089472f, 0.342797339f,
        -0.529900968f, -0.108067632f,  0.507653534f, 0.349348515f,
    };
    rng_seed(&rng, 123);
    for (size_t h = 0; h < HEADS; h++) {
        heads[h] = (SelfAttention){
            .d_in = D_IN, .d_out = D_OUT,
            .w_query = w[h][0], .w_key = w[h][1], .w_value = w[h][2],
            .causal = true,
        };
        self_attention_init(&heads[h], &rng);
    }
    float out[B * T * HEADS * D_OUT];
    run_wrapper(out, batch, B, heads, HEADS, &rng);
    CHECK(floats_close(out, book, T * HEADS * D_OUT, BOOK_TOL));
    CHECK(floats_close(out, torch, T * HEADS * D_OUT, TOL));
    CHECK(memcmp(&out[T * HEADS * D_OUT], out, sizeof torch) == 0);

    /* Exercise 3.2: d_out = 1 gives two-dimensional context vectors. */
    static const float torch_d1[T * HEADS] = {
        -0.573988438f, 0.221604809f,
        -0.731986284f, 0.0155245587f,
        -0.777364731f, -0.0546023101f,
        -0.697864592f, -0.0817382485f,
        -0.653844774f, -0.0957008749f,
        -0.642380595f, -0.106453821f,
    };
    rng_seed(&rng, 123);
    for (size_t h = 0; h < HEADS; h++) {
        heads[h].d_out = 1;
        self_attention_init(&heads[h], &rng);
    }
    run_wrapper(out, batch, B, heads, HEADS, &rng);
    CHECK(floats_close(out, torch_d1, T * HEADS, TOL));
}

/* Section 3.6.2: torch.manual_seed(123), then
 * MultiHeadAttention(3, 2, 6, dropout, num_heads=2) on the book's batch. */
static void check_multi_head(void)
{
    float batch[B * T * D_IN];
    book_batch(batch);
    float w[3][D_OUT * D_IN], w_out[D_OUT * D_OUT], b_out[D_OUT];
    MultiHeadAttention mha = {
        .d_in = D_IN, .d_out = D_OUT, .num_heads = HEADS,
        .w_query = w[0], .w_key = w[1], .w_value = w[2],
        .w_out = w_out, .b_out = b_out,
    };
    Rng rng;
    rng_seed(&rng, 123);
    multi_head_attention_init(&mha, &rng);
    /* out_proj comes after the three projections. */
    CHECK(floats_close(w_out, (float[]){-0.166757792f, 0.22697258f, 0.500025988f, 0.131738231f},
                       D_OUT * D_OUT, TOL));
    CHECK(floats_close(b_out, (float[]){0.193358868f, 0.682540953f}, D_OUT, TOL));

    /* dropout = 0.0: context_vecs has shape [2, 6, 2]. */
    static const float book[T * D_OUT] = {
        0.3190f, 0.4858f,
        0.2943f, 0.3897f,
        0.2856f, 0.3593f,
        0.2693f, 0.3873f,
        0.2639f, 0.3928f,
        0.2575f, 0.4028f,
    };
    static const float torch[T * D_OUT] = {
        0.319018304f, 0.485762894f,
        0.294346005f, 0.389676273f,
        0.285574675f, 0.359277695f,
        0.269263685f, 0.387326658f,
        0.263870537f, 0.392795712f,
        0.257473588f, 0.402782619f,
    };
    float out[B * T * D_OUT];
    run_multi_head(out, batch, B, T, &mha, &rng);
    CHECK(floats_close(out, book, T * D_OUT, BOOK_TOL));
    CHECK(floats_close(out, torch, T * D_OUT, TOL));
    CHECK(memcmp(&out[T * D_OUT], out, sizeof torch) == 0);

    /* dropout = 0.5 in training mode: the draws go sequence by sequence,
     * head by head. */
    static const float torch_dropout[B * T * D_OUT] = {
        0.344081342f, 0.230597198f,
        0.317791224f, 0.309428483f,
        0.275156021f, 0.43727088f,
        0.330518663f, 0.216208607f,
        0.176625133f, 0.566732824f,
        0.261481106f, 0.398100495f,

        0.293955296f, 0.74092859f,
        0.367964447f, 0.338549763f,
        0.248665184f, 0.423198789f,
        0.292671293f, 0.304582477f,
        0.210256726f, 0.40234524f,
        0.239944533f, 0.405447304f,
    };
    mha.dropout = 0.5;
    rng_seed(&rng, 123);
    multi_head_attention_init(&mha, &rng);
    run_multi_head(out, batch, B, T, &mha, &rng);
    CHECK(floats_close(out, torch_dropout, B * T * D_OUT, TOL));
}

/* torch.manual_seed(123); MultiHeadAttention(3, 4, 6, 0.0, num_heads=2,
 * qkv_bias=True): two heads of two dimensions each, unlike the book's
 * one-dimensional heads. */
static void check_head_split(void)
{
    enum { D = 4, HEAD_DIM = D / HEADS };
    float w[3][D * D_IN], b[3][D], w_out[D * D], b_out[D];
    MultiHeadAttention mha = {
        .d_in = D_IN, .d_out = D, .num_heads = HEADS,
        .w_query = w[0], .w_key = w[1], .w_value = w[2],
        .b_query = b[0], .b_key = b[1], .b_value = b[2],
        .w_out = w_out, .b_out = b_out,
    };
    Rng rng;
    rng_seed(&rng, 123);
    multi_head_attention_init(&mha, &rng);

    static const float torch[T * D] = {
        -0.0525227338f, -0.143321693f,  -0.566870332f, -0.0898127705f,
        -0.0871039927f, -0.128182203f,  -0.607390702f, -0.118989632f,
        -0.101819813f,  -0.118385062f,  -0.617670238f, -0.125602692f,
        -0.121667176f,  -0.105415672f,  -0.57005173f,  -0.126912773f,
        -0.142786086f,  -0.0666071326f, -0.502120972f, -0.0770646557f,
        -0.142120332f,  -0.0818738341f, -0.513251305f, -0.105440594f,
    };
    float batch[B * T * D_IN], out[B * T * D];
    book_batch(batch);
    run_multi_head(out, batch, B, T, &mha, NULL);
    CHECK(floats_close(out, torch, T * D, TOL));

    /* Figure 3.26: head h's weights are rows [h * HEAD_DIM, (h + 1) *
     * HEAD_DIM) of each big projection. Running those as the heads of
     * MultiHeadAttentionWrapper, then out_proj, gives the same numbers. */
    SelfAttention heads[HEADS];
    for (size_t h = 0; h < HEADS; h++)
        heads[h] = (SelfAttention){
            .d_in = D_IN, .d_out = HEAD_DIM,
            .w_query = w[0] + h * HEAD_DIM * D_IN,
            .w_key = w[1] + h * HEAD_DIM * D_IN,
            .w_value = w[2] + h * HEAD_DIM * D_IN,
            .b_query = b[0] + h * HEAD_DIM,
            .b_key = b[1] + h * HEAD_DIM,
            .b_value = b[2] + h * HEAD_DIM,
            .causal = true,
        };
    float concat[B * T * D], via_wrapper[B * T * D];
    mixed_batch(batch);
    run_multi_head(out, batch, B, T, &mha, NULL);
    run_wrapper(concat, batch, B, heads, HEADS, NULL);
    linear_forward(via_wrapper, concat, B * T, w_out, b_out, D, D);
    CHECK(memcmp(via_wrapper, out, sizeof out) == 0);
}

enum { GPT2_DIM = 768, GPT2_HEADS = 12, GPT2_T = 4 };

/* Exercise 3.3: torch.manual_seed(123); MultiHeadAttention(768, 768, 1024,
 * 0.0, num_heads=12), the size of GPT-2 small, then x = torch.rand(1, 4, 768). */
static void check_gpt2_size(void)
{
    /* Four 768 x 768 weights and out_proj's 768 biases: the 2,360,064
     * parameters PyTorch counts. */
    size_t dd = (size_t)GPT2_DIM * GPT2_DIM;
    float *params = malloc(sizeof(float) * (4 * dd + GPT2_DIM));
    float *x = malloc(sizeof(float) * GPT2_T * GPT2_DIM);
    float *out = malloc(sizeof(float) * GPT2_T * GPT2_DIM);
    if (!params || !x || !out) {
        fprintf(stderr, "FAIL out of memory\n");
        failures++;
        goto done;
    }
    MultiHeadAttention mha = {
        .d_in = GPT2_DIM, .d_out = GPT2_DIM, .num_heads = GPT2_HEADS,
        .w_query = params, .w_key = params + dd, .w_value = params + 2 * dd,
        .w_out = params + 3 * dd, .b_out = params + 4 * dd,
    };
    Rng rng;
    rng_seed(&rng, 123);
    multi_head_attention_init(&mha, &rng);
    CHECK(floats_close(mha.w_query, (float[]){-0.0147143528f, 0.00119527976f, -0.017921621f}, 3,
                       TOL));
    CHECK(floats_close(&mha.w_value[dd - 3],
                       (float[]){0.00142194377f, -0.0177231338f, -0.0271463916f}, 3, TOL));
    CHECK(floats_close(mha.w_out, (float[]){0.0148252351f, -0.018582128f, -0.016898388f}, 3, TOL));
    CHECK(floats_close(&mha.b_out[GPT2_DIM - 3],
                       (float[]){0.0290636551f, -0.0310064796f, 0.0355794132f}, 3, TOL));

    /* Twelve heads of 64 dimensions; the middle check lands in head 6. */
    rng_uniform(&rng, x, GPT2_T * GPT2_DIM, 0, 1);
    run_multi_head(out, x, 1, GPT2_T, &mha, NULL);
    CHECK(floats_close(out, (float[]){0.240990832f, 0.020827163f, 0.0829337016f}, 3, TOL));
    CHECK(floats_close(&out[GPT2_DIM + 384],
                       (float[]){-0.0797438473f, -0.126870573f, -0.262400866f}, 3, TOL));
    CHECK(floats_close(&out[GPT2_T * GPT2_DIM - 3],
                       (float[]){0.0381890871f, 0.12357641f, -0.0442726053f}, 3, TOL));

done:
    free(params);
    free(x);
    free(out);
}

int main(void)
{
    check_v1();
    check_v2();
    check_qkv_bias();
    check_causal();
    check_dropout();
    check_causal_attention();
    check_wrapper();
    check_multi_head();
    check_head_split();
    check_gpt2_size();
    return test_report("test_attention");
}
