#include "gpt.h"

#include "dropout.h"
#include "embedding.h"
#include "linear.h"

const GPTConfig GPT_CONFIG_124M = {
    .vocab_size = 50257,
    .context_length = 1024,
    .emb_dim = 768,
    .n_heads = 12,
    .n_layers = 12,
    .drop_rate = 0.1,
    .qkv_bias = false,
};

size_t gpt_num_params(const GPTConfig *cfg)
{
    size_t d = cfg->emb_dim;
    size_t embeddings = (cfg->vocab_size + cfg->context_length) * d;
    size_t blocks = cfg->n_layers * transformer_block_num_params(d, cfg->qkv_bias);
    return embeddings + blocks + 2 * d + cfg->vocab_size * d;
}

void gpt_bind(GPTModel *model, const GPTConfig *cfg, float *params)
{
    size_t d = cfg->emb_dim;
    model->cfg = *cfg;
    model->tok_emb = params;
    model->pos_emb = model->tok_emb + cfg->vocab_size * d;
    model->blocks = model->pos_emb + cfg->context_length * d;
    float *after = model->blocks + cfg->n_layers * transformer_block_num_params(d, cfg->qkv_bias);
    model->final_norm = (LayerNorm){.dim = d, .scale = after, .shift = after + d};
    model->out_head = after + 2 * d;
}

TransformerBlock gpt_block(const GPTModel *model, size_t layer)
{
    const GPTConfig *cfg = &model->cfg;
    size_t size = transformer_block_num_params(cfg->emb_dim, cfg->qkv_bias);
    TransformerBlock tb;
    transformer_block_bind(&tb, model->blocks + layer * size, cfg->emb_dim, cfg->n_heads,
                           cfg->qkv_bias, cfg->drop_rate);
    return tb;
}

void gpt_init(const GPTModel *model, Rng *rng)
{
    const GPTConfig *cfg = &model->cfg;
    embedding_init(model->tok_emb, cfg->vocab_size, cfg->emb_dim, rng);
    embedding_init(model->pos_emb, cfg->context_length, cfg->emb_dim, rng);
    for (size_t l = 0; l < cfg->n_layers; l++) {
        TransformerBlock tb = gpt_block(model, l);
        transformer_block_init(&tb, rng);
    }
    layernorm_init(&model->final_norm);
    linear_init(model->out_head, NULL, cfg->emb_dim, cfg->vocab_size, rng);
}

size_t gpt_scratch_size(const GPTModel *model, size_t B, size_t T)
{
    /* Every block has the same shape, so the first one's need is all of
     * theirs. */
    TransformerBlock tb = gpt_block(model, 0);
    return 2 * B * T * model->cfg.emb_dim + transformer_block_scratch_size(&tb, B, T);
}

/* Embeddings and the blocks on a batch, leaving their output [B x T x
 * emb_dim] in scratch and returning where. */
static float *gpt_hidden(const int *ids, size_t B, size_t T, const GPTModel *model,
                         float *scratch, Rng *rng)
{
    const GPTConfig *cfg = &model->cfg;
    size_t d = cfg->emb_dim, n = B * T;
    float *x = scratch;
    float *y = x + n * d;
    float *rest = y + n * d;

    input_embedding_forward(x, ids, model->tok_emb, model->pos_emb, B, T, d);
    if (rng)
        dropout_forward(x, n * d, cfg->drop_rate, rng);
    /* Each block reads x and writes y, then the two trade places. */
    for (size_t l = 0; l < cfg->n_layers; l++) {
        TransformerBlock tb = gpt_block(model, l);
        transformer_block_forward(y, x, B, T, &tb, rest, rng);
        float *t = x;
        x = y;
        y = t;
    }
    return x;
}

void gpt_forward(float *logits, const int *ids, size_t B, size_t T, const GPTModel *model,
                 float *scratch, Rng *rng)
{
    const GPTConfig *cfg = &model->cfg;
    size_t n = B * T;
    float *x = gpt_hidden(ids, B, T, model, scratch, rng);
    layernorm_forward(x, x, n, &model->final_norm);
    linear_forward(logits, x, n, model->out_head, NULL, cfg->emb_dim, cfg->vocab_size);
}

size_t generate_scratch_size(const GPTModel *model, size_t context_size)
{
    return gpt_scratch_size(model, 1, context_size) + model->cfg.vocab_size;
}

size_t generate_text_simple(int *ids, size_t n, size_t max_new_tokens, size_t context_size,
                            const GPTModel *model, float *scratch)
{
    const GPTConfig *cfg = &model->cfg;
    size_t d = cfg->emb_dim, V = cfg->vocab_size;
    float *logits = scratch;
    float *rest = logits + V;
    for (size_t step = 0; step < max_new_tokens; step++) {
        /* Crop the context to its last context_size tokens. */
        size_t T = n < context_size ? n : context_size;
        float *x = gpt_hidden(ids + n - T, 1, T, model, rest, NULL);
        /* Only the last token's logits pick the next token, so the final
         * norm and out_head run on that row alone. */
        float *last = x + (T - 1) * d;
        layernorm_forward(last, last, 1, &model->final_norm);
        linear_forward(logits, last, 1, model->out_head, NULL, d, V);
        /* Greedy decoding: the first highest logit, as torch.argmax picks.
         * The book's softmax first would not change the order. */
        size_t best = 0;
        for (size_t v = 1; v < V; v++)
            if (logits[v] > logits[best])
                best = v;
        ids[n++] = (int)best;
    }
    return n;
}
