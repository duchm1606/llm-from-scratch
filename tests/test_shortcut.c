#include <stdio.h>
#include <string.h>

#include "activation.h"
#include "linear.h"
#include "ops.h"
#include "rng.h"
#include "test_util.h"

/* Expectations come from PyTorch 2.14 on CPU, running the book's classes;
 * the book prints only gradients, which need a backward pass (chapter 5). */
#define TOL 1e-6f

/* Listing 4.5: ExampleDeepNeuralNetwork(layer_sizes=[3, 3, 3, 3, 3, 1]),
 * five nn.Sequential(nn.Linear, GELU()) layers. */
enum { LAYERS = 5, WIDTH = 3 };

static const size_t layer_sizes[LAYERS + 1] = {3, 3, 3, 3, 3, 1};

typedef struct {
    float weight[LAYERS][WIDTH * WIDTH];
    float bias[LAYERS][WIDTH];
} Network;

/* torch.manual_seed(123), then the layers in order. */
static void network_init(Network *net)
{
    Rng rng;
    rng_seed(&rng, 123);
    for (size_t l = 0; l < LAYERS; l++)
        linear_init(net->weight[l], net->bias[l], layer_sizes[l], layer_sizes[l + 1], &rng);
}

/* The forward pass on one example, leaving each layer's result in
 * steps[l]. A layer's input is added back to its output when the shapes
 * match, so not around the last layer, which maps 3 values to 1. */
static void network_forward(float steps[LAYERS][WIDTH], const Network *net, const float *x,
                            bool use_shortcut)
{
    for (size_t l = 0; l < LAYERS; l++) {
        size_t d_in = layer_sizes[l], d_out = layer_sizes[l + 1];
        float *out = steps[l];
        linear_forward(out, x, 1, net->weight[l], net->bias[l], d_in, d_out);
        gelu_forward(out, d_out);
        if (use_shortcut && d_in == d_out)
            residual_add(out, x, d_out);
        x = out;
    }
}

/* Section 4.4: sample_input = torch.tensor([[1., 0., -1.]]) through the
 * network without and with shortcut connections, both from seed 123. */
static void check_example_network(void)
{
    static const float sample_input[WIDTH] = {1, 0, -1};
    Network net;
    network_init(&net);
    CHECK(floats_close(net.weight[0],
                       (float[]){-0.235429645f, 0.0191244762f, -0.286745936f, 0.217726618f,
                                 -0.49193421f, 0.423223078f, -0.419641405f, -0.459017664f,
                                 -0.364820182f},
                       WIDTH * WIDTH, TOL));
    CHECK(floats_close(net.bias[0], (float[]){0.261478186f, -0.213326395f, 0.216052175f}, WIDTH,
                       TOL));

    static const float plain[LAYERS][WIDTH] = {
        {0.194799706f, -0.141434029f, 0.0909411684f},
        {0.0219030399f, 0.137585908f, 0.382398427f},
        {0.0894873217f, 0.222025156f, 0.0362320058f},
        {-0.0360834301f, -0.0549890809f, 0.119717427f},
        {0.0610073395f},
    };
    static const float shortcut[LAYERS][WIDTH] = {
        {1.19479966f, -0.141434029f, -0.909058809f},
        {1.09892368f, -0.244487107f, -0.910425842f},
        {1.17713535f, 0.935360849f, -0.814638674f},
        {1.04334986f, 0.993030488f, -0.399728119f},
        {0.766931534f},
    };
    float steps[LAYERS][WIDTH];
    network_forward(steps, &net, sample_input, false);
    for (size_t l = 0; l < LAYERS; l++)
        CHECK(floats_close(steps[l], plain[l], layer_sizes[l + 1], TOL));
    network_forward(steps, &net, sample_input, true);
    for (size_t l = 0; l < LAYERS; l++)
        CHECK(floats_close(steps[l], shortcut[l], layer_sizes[l + 1], TOL));
}

static void check_residual_add(void)
{
    float x[4] = {1, -2, 0.5f, 0};
    residual_add(x, (float[]){0.25f, 2, -0.5f, -3}, 4);
    CHECK(memcmp(x, (float[]){1.25f, 0, 0, -3}, sizeof x) == 0);
}

int main(void)
{
    check_residual_add();
    check_example_network();
    return test_report("test_shortcut");
}
