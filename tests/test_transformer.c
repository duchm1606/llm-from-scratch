#include <stdio.h>
#include <stdlib.h>

#include "rng.h"
#include "test_util.h"
#include "transformer.h"

/* Expectations come from PyTorch 2.14 on CPU, running the book's classes;
 * the book prints only the block's input and output shapes. */
#define TOL 1e-6f

enum { GUARD = 16 };

#define SENTINEL -1234.5f

/* transformer_block_forward in scratch followed by GUARD sentinels,
 * checking that nothing wrote past the size it reports. */
static void run_block(float *out, const float *x, size_t B, size_t T, const TransformerBlock *tb,
                      Rng *rng)
{
    size_t size = transformer_block_scratch_size(tb, B, T);
    float *scratch = malloc(sizeof(float) * (size + GUARD));
    if (!scratch) {
        fprintf(stderr, "FAIL out of memory\n");
        failures++;
        return;
    }
    for (size_t i = 0; i < GUARD; i++)
        scratch[size + i] = SENTINEL;
    transformer_block_forward(out, x, B, T, tb, scratch, rng);
    bool intact = true;
    for (size_t i = 0; i < GUARD; i++)
        intact &= scratch[size + i] == SENTINEL;
    CHECK(intact);
    free(scratch);
}

/* torch.manual_seed(123); TransformerBlock(cfg) with emb_dim 8, n_heads 2,
 * drop_rate 0.5 and qkv_bias=True, then x = torch.randn(2, 3, 8): small
 * enough to compare every output, in training mode and in evaluation
 * mode. */
static void check_small(void)
{
    enum { B = 2, T = 3, C = 8 };
    size_t n_params = transformer_block_num_params(C, true);
    /* Attention 4 * 8 * 8 weights and 4 * 8 biases, the feed forward
     * network 2 * 32 * 8 weights and 32 + 8 biases, two norms of 2 * 8. */
    CHECK(n_params == 288 + 552 + 32);
    float *params = malloc(sizeof(float) * n_params);
    if (!params) {
        fprintf(stderr, "FAIL out of memory\n");
        failures++;
        return;
    }
    TransformerBlock tb;
    transformer_block_bind(&tb, params, C, 2, true, 0.5);
    Rng rng;
    rng_seed(&rng, 123);
    transformer_block_init(&tb, &rng);

    float x[B * T * C], out[B * T * C];
    rng_normal(&rng, x, B * T * C, 0, 1);
    CHECK(floats_close(x, (float[]){-0.42742911f, -1.77539361f, 0.0727675408f}, 3, TOL));

    static const float train[B * T * C] = {
        -0.42742911f,  -2.11300063f,  2.92774129f,   2.27588701f,   1.21674871f,
        0.203757003f,  -2.55844975f,  0.466748893f,  -0.928381443f, 1.2453897f,
        -0.72416389f,  0.16707924f,   0.324812531f,  -0.516501188f, -0.500502467f,
        -1.60667348f,  -1.90505421f,  -2.32620907f,  -1.93837702f,  0.397396833f,
        1.92447233f,   -0.0132912993f, -0.537582338f, 0.224446058f, 0.981761873f,
        -1.46081543f,  -0.295695066f, -1.90441883f,  0.321291685f,  1.50444961f,
        0.338769466f,  -1.18388677f,  -0.303573638f, -0.0617395639f, 0.114269935f,
        -0.173478097f, -0.0794827715f, 0.0245802999f, -0.0530513525f, -1.68446493f,
        0.288930327f,  -0.269193351f, 0.899811029f,  1.01739073f,   0.321569026f,
        -0.0676457435f, -0.372678012f, 0.723124981f,
    };
    run_block(out, x, B, T, &tb, &rng);
    CHECK(floats_close(out, train, B * T * C, TOL));

    static const float eval[B * T * C] = {
        -0.680392981f, -2.06496501f,  0.829550385f,  2.42978072f,   0.0896595269f,
        0.0984253958f, -1.37944615f,  0.825263858f,  -0.989922345f, 0.924057126f,
        0.403270423f,  0.203762308f,  -0.532445192f, -0.217615515f, -1.20655656f,
        -0.406134158f, -2.02987766f,  -0.893013418f, -1.79364967f,  0.697246253f,
        2.14979124f,   0.482600451f,  -0.162253216f, 0.831068039f,  1.01046324f,
        -1.67520523f,  0.337228626f,  -2.39021587f,  0.329586506f,  1.86815107f,
        0.256925136f,  -1.02437067f,  -0.470636189f, 0.281942308f,  0.188624084f,
        -0.514790297f, -0.109460503f, -0.167735249f, -0.121015489f, -0.544937789f,
        0.499258816f,  -0.743461132f, 1.37415004f,   0.274436831f,  0.0292910933f,
        0.0254527628f, -0.71273315f,  0.282784551f,
    };
    run_block(out, x, B, T, &tb, NULL);
    CHECK(floats_close(out, eval, B * T * C, TOL));
    free(params);
}

/* Section 4.5: torch.manual_seed(123); x = torch.rand(2, 4, 768); block =
 * TransformerBlock(GPT_CONFIG_124M); output = block(x), in training mode
 * as in the book, then in evaluation mode. */
static void check_gpt2_size(void)
{
    enum { B = 2, T = 4, C = 768 };
    size_t n_params = transformer_block_num_params(C, false);
    CHECK(n_params == 7085568);

    float *params = malloc(sizeof(float) * n_params);
    float *x = malloc(sizeof(float) * B * T * C);
    float *out = malloc(sizeof(float) * B * T * C);
    if (!params || !x || !out) {
        fprintf(stderr, "FAIL out of memory\n");
        failures++;
        goto done;
    }
    TransformerBlock tb;
    transformer_block_bind(&tb, params, C, 12, false, 0.1);
    CHECK(tb.att.b_query == NULL && tb.ff.hidden_dim == 4 * C);
    CHECK(tb.norm2.shift + C == params + n_params);
    /* Exercise 4.1: attention comes first and holds 2,360,064 parameters;
     * the feed forward network after it holds twice as many. */
    CHECK(tb.ff.w_in - params == 2360064);
    CHECK(tb.norm1.scale - tb.ff.w_in == 4722432);

    Rng rng;
    rng_seed(&rng, 123);
    rng_uniform(&rng, x, B * T * C, 0, 1);
    CHECK(floats_close(x, (float[]){0.296111941f, 0.516562283f, 0.251670718f}, 3, TOL));
    transformer_block_init(&tb, &rng);

    run_block(out, x, B, T, &tb, &rng);
    CHECK(floats_close(out, (float[]){-0.0055410862f, 0.0972206295f, -0.112203076f}, 3, TOL));
    /* Token [1][0], from column 384 */
    CHECK(floats_close(&out[T * C + 384], (float[]){0.651830912f, 0.263298482f, 0.198077708f}, 3,
                       TOL));
    CHECK(floats_close(&out[B * T * C - 3], (float[]){1.19630373f, 1.26495934f, 0.224327385f}, 3,
                       TOL));

    run_block(out, x, B, T, &tb, NULL);
    CHECK(floats_close(out, (float[]){0.129034653f, 0.266071439f, -0.0603248999f}, 3, TOL));
    CHECK(floats_close(&out[T * C + 384], (float[]){0.520124555f, 0.356406122f, 0.202043891f}, 3,
                       TOL));
    CHECK(floats_close(&out[B * T * C - 3], (float[]){1.21079969f, 1.27080011f, 0.209654763f}, 3,
                       TOL));

done:
    free(params);
    free(x);
    free(out);
}

int main(void)
{
    check_small();
    check_gpt2_size();
    return test_report("test_transformer");
}
