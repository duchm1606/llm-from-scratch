#include <math.h>
#include <string.h>

#include "ops.h"
#include "test_util.h"

/* Full-precision expectations come from PyTorch 2.14 on CPU; the book prints
 * four decimals. */
#define TOL 1e-6f
#define BOOK_TOL 5e-5f

enum { T = 6, D = 3 };

/* Section 3.3.1: "Your journey starts with one step" as 3-d embeddings. */
static const float inputs[T * D] = {
    0.43f, 0.15f, 0.89f, /* Your    */
    0.55f, 0.87f, 0.66f, /* journey */
    0.57f, 0.85f, 0.64f, /* starts  */
    0.22f, 0.58f, 0.33f, /* with    */
    0.77f, 0.25f, 0.10f, /* one     */
    0.05f, 0.80f, 0.55f, /* step    */
};

static void check_shapes(void)
{
    /* [2 x 3] @ [3 x 4]; small integers, so every sum is exact. */
    static const float a[2 * 3] = {1, 2, 3, 4, 5, 6};
    static const float b[3 * 4] = {
        1, 0,  -1, 2,
        0, 1,  2,  -1,
        3, -2, 0,  1,
    };
    static const float b_t[4 * 3] = {
        1,  0,  3,
        0,  1,  -2,
        -1, 2,  0,
        2,  -1, 1,
    };
    static const float want[2 * 4] = {
        10, -4, 3, 3,
        22, -7, 6, 9,
    };
    float out[2 * 4];
    matmul(out, a, b, 2, 3, 4);
    CHECK(memcmp(out, want, sizeof want) == 0);
    matmul_bt(out, a, b_t, 2, 3, 4);
    CHECK(memcmp(out, want, sizeof want) == 0);
}

/* Section 3.3: attention without trainable weights, first for the query
 * "journey" alone (3.3.1), then for every query at once (3.3.2). */
static void check_simple_attention(void)
{
    /* attn_scores_2[i] = torch.dot(x_i, query) */
    float scores_2[T], context_2[D];
    matmul_bt(scores_2, &inputs[1 * D], inputs, 1, D, T);
    CHECK(floats_close(scores_2, (float[]){0.9544f, 1.4950f, 1.4754f, 0.8434f, 0.7070f, 1.0865f},
                       T, BOOK_TOL));

    /* attn_weights_2 = torch.softmax(attn_scores_2, dim=0) */
    softmax_rows(scores_2, 1, T);
    CHECK(floats_close(scores_2, (float[]){0.1385f, 0.2379f, 0.2333f, 0.1240f, 0.1082f, 0.1581f},
                       T, BOOK_TOL));

    /* context_vec_2 = sum of attn_weights_2[i] * x_i */
    matmul(context_2, scores_2, inputs, 1, T, D);
    CHECK(floats_close(context_2, (float[]){0.4419f, 0.6515f, 0.5683f}, D, BOOK_TOL));

    /* attn_scores = inputs @ inputs.T */
    static const float book_scores[T * T] = {
        0.9995f, 0.9544f, 0.9422f, 0.4753f, 0.4576f, 0.6310f,
        0.9544f, 1.4950f, 1.4754f, 0.8434f, 0.7070f, 1.0865f,
        0.9422f, 1.4754f, 1.4570f, 0.8296f, 0.7154f, 1.0605f,
        0.4753f, 0.8434f, 0.8296f, 0.4937f, 0.3474f, 0.6565f,
        0.4576f, 0.7070f, 0.7154f, 0.3474f, 0.6654f, 0.2935f,
        0.6310f, 1.0865f, 1.0605f, 0.6565f, 0.2935f, 0.9450f,
    };
    static const float torch_scores[T * T] = {
        0.999499977f, 0.954400003f, 0.942200005f, 0.475300014f, 0.457599968f, 0.631000042f,
        0.954400003f, 1.495f,       1.47540009f,  0.843400002f, 0.707000017f, 1.08650005f,
        0.942200005f, 1.47540009f,  1.45700002f,  0.829599977f, 0.715399981f, 1.06050003f,
        0.475300014f, 0.843400002f, 0.829599977f, 0.493699998f, 0.34739998f,  0.656499982f,
        0.457599968f, 0.707000017f, 0.715399981f, 0.34739998f,  0.665399969f, 0.293500006f,
        0.631000042f, 1.08650005f,  1.06050003f,  0.656499982f, 0.293500006f, 0.945000052f,
    };
    float weights[T * T], context[T * D];
    matmul_bt(weights, inputs, inputs, T, D, T);
    CHECK(floats_close(weights, book_scores, T * T, BOOK_TOL));
    CHECK(floats_close(weights, torch_scores, T * T, TOL));

    /* attn_weights = torch.softmax(attn_scores, dim=-1) */
    static const float book_weights[T * T] = {
        0.2098f, 0.2006f, 0.1981f, 0.1242f, 0.1220f, 0.1452f,
        0.1385f, 0.2379f, 0.2333f, 0.1240f, 0.1082f, 0.1581f,
        0.1390f, 0.2369f, 0.2326f, 0.1242f, 0.1108f, 0.1565f,
        0.1435f, 0.2074f, 0.2046f, 0.1462f, 0.1263f, 0.1720f,
        0.1526f, 0.1958f, 0.1975f, 0.1367f, 0.1879f, 0.1295f,
        0.1385f, 0.2184f, 0.2128f, 0.1420f, 0.0988f, 0.1896f,
    };
    static const float torch_weights[T * T] = {
        0.209834769f, 0.200581461f, 0.198149234f, 0.124228224f, 0.122048721f,  0.145157665f,
        0.138547584f, 0.237891287f, 0.233274043f, 0.123991594f, 0.108181871f,  0.158113614f,
        0.139007598f, 0.236921459f, 0.232601956f, 0.124204405f, 0.110800207f,  0.156464428f,
        0.143526882f, 0.207394421f, 0.204552025f, 0.146192238f, 0.126295254f,  0.172039196f,
        0.152610853f, 0.195838675f, 0.197490647f, 0.136686683f, 0.187858909f,  0.129514292f,
        0.138471171f, 0.218363717f, 0.212759435f, 0.142047569f, 0.0988063738f, 0.189551756f,
    };
    softmax_rows(weights, T, T);
    CHECK(floats_close(weights, book_weights, T * T, BOOK_TOL));
    CHECK(floats_close(weights, torch_weights, T * T, TOL));
    for (size_t i = 0; i < T; i++) {
        float sum = 0;
        for (size_t j = 0; j < T; j++)
            sum += weights[i * T + j];
        CHECK(fabsf(sum - 1) <= TOL);
    }

    /* all_context_vecs = attn_weights @ inputs */
    static const float book_context[T * D] = {
        0.4421f, 0.5931f, 0.5790f,
        0.4419f, 0.6515f, 0.5683f,
        0.4431f, 0.6496f, 0.5671f,
        0.4304f, 0.6298f, 0.5510f,
        0.4671f, 0.5910f, 0.5266f,
        0.4177f, 0.6503f, 0.5645f,
    };
    static const float torch_context[T * D] = {
        0.442059427f, 0.59309864f,  0.578989148f,
        0.441865742f, 0.651481986f, 0.56830889f,
        0.443127543f, 0.649594665f, 0.567073107f,
        0.430389732f, 0.629828095f, 0.55102706f,
        0.467101753f, 0.590992808f, 0.526596546f,
        0.41772449f,  0.650323212f, 0.56453526f,
    };
    matmul(context, weights, inputs, T, T, D);
    CHECK(floats_close(context, book_context, T * D, BOOK_TOL));
    CHECK(floats_close(context, torch_context, T * D, TOL));

    /* Row 2 is context_vec_2 exactly: each row is computed independently. */
    CHECK(memcmp(&context[1 * D], context_2, sizeof context_2) == 0);
}

static void check_softmax_edge_cases(void)
{
    /* softmax_naive overflows here (exp(1000) is inf, and inf / inf is nan).
     * Shifting by the max leaves the same numbers as for [0, 1, 2]. */
    float big[3] = {1000, 1001, 1002};
    float small[3] = {0, 1, 2};
    softmax_rows(big, 1, 3);
    softmax_rows(small, 1, 3);
    CHECK(floats_close(big, (float[]){0.0900305733f, 0.244728476f, 0.665240943f}, 3, TOL));
    CHECK(memcmp(big, small, sizeof big) == 0);

    /* -inf scores get weight exactly 0, as the causal mask of section 3.5 needs. */
    float masked[4] = {0.5f, -INFINITY, 1.5f, -INFINITY};
    softmax_rows(masked, 1, 4);
    CHECK(floats_close(masked, (float[]){0.268941432f, 0, 0.731058598f, 0}, 4, TOL));
    CHECK(masked[1] == 0 && masked[3] == 0);
}

int main(void)
{
    check_shapes();
    check_simple_attention();
    check_softmax_edge_cases();
    return test_report("test_ops");
}
