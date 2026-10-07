#include "feedforward.h"

#include "activation.h"
#include "linear.h"

void feed_forward_init(const FeedForward *ff, Rng *rng)
{
    linear_init(ff->w_in, ff->b_in, ff->emb_dim, ff->hidden_dim, rng);
    linear_init(ff->w_out, ff->b_out, ff->hidden_dim, ff->emb_dim, rng);
}

size_t feed_forward_scratch_size(const FeedForward *ff)
{
    return ff->hidden_dim;
}

void feed_forward_forward(float *out, const float *x, size_t n, const FeedForward *ff,
                          float *scratch)
{
    size_t d = ff->emb_dim, h = ff->hidden_dim;
    /* One token at a time, so the hidden layer needs room for one row. */
    for (size_t i = 0; i < n; i++) {
        linear_forward(scratch, x + i * d, 1, ff->w_in, ff->b_in, d, h);
        gelu_forward(scratch, h);
        linear_forward(out + i * d, scratch, 1, ff->w_out, ff->b_out, h, d);
    }
}
