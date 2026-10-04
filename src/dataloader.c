#include "dataloader.h"

#include <stdlib.h>
#include <string.h>

#include "rng.h"

struct DataLoader {
    const int *ids;
    DataLoaderConfig cfg;
    size_t n_windows;
    size_t *starts;  /* window start offsets, in this epoch's order */
    size_t next;     /* index into starts of the next window to serve */
    Rng rng;
};

DataLoader *dataloader_create(const int *ids, size_t n_ids, DataLoaderConfig cfg)
{
    if (cfg.batch_size == 0 || cfg.max_length == 0 || cfg.stride == 0)
        return NULL;

    DataLoader *dl = calloc(1, sizeof *dl);
    if (!dl)
        return NULL;
    dl->ids = ids;
    dl->cfg = cfg;

    /* range(0, n_ids - max_length, stride): the last window still has a
     * target token after it. */
    if (n_ids > cfg.max_length)
        dl->n_windows = (n_ids - cfg.max_length - 1) / cfg.stride + 1;
    dl->starts = malloc((dl->n_windows > 0 ? dl->n_windows : 1) * sizeof *dl->starts);
    if (!dl->starts) {
        free(dl);
        return NULL;
    }
    for (size_t i = 0; i < dl->n_windows; i++)
        dl->starts[i] = i * cfg.stride;

    rng_seed(&dl->rng, cfg.seed);
    dataloader_reset(dl);
    return dl;
}

void dataloader_free(DataLoader *dl)
{
    if (!dl)
        return;
    free(dl->starts);
    free(dl);
}

size_t dataloader_num_windows(const DataLoader *dl)
{
    return dl->n_windows;
}

size_t dataloader_num_batches(const DataLoader *dl)
{
    size_t b = dl->cfg.batch_size;
    return dl->cfg.drop_last ? dl->n_windows / b : (dl->n_windows + b - 1) / b;
}

void dataloader_reset(DataLoader *dl)
{
    dl->next = 0;
    if (!dl->cfg.shuffle)
        return;
    /* Fisher-Yates. Shuffling the previous epoch's order is still uniform. */
    for (size_t i = dl->n_windows; i > 1; i--) {
        size_t j = (size_t)rng_below(&dl->rng, i);
        size_t tmp = dl->starts[i - 1];
        dl->starts[i - 1] = dl->starts[j];
        dl->starts[j] = tmp;
    }
}

size_t dataloader_next(DataLoader *dl, int *inputs, int *targets)
{
    size_t len = dl->cfg.max_length;
    size_t rows = dl->n_windows - dl->next;
    if (rows > dl->cfg.batch_size)
        rows = dl->cfg.batch_size;
    if (rows == 0 || (rows < dl->cfg.batch_size && dl->cfg.drop_last))
        return 0;

    for (size_t r = 0; r < rows; r++) {
        const int *window = dl->ids + dl->starts[dl->next + r];
        memcpy(inputs + r * len, window, len * sizeof *inputs);
        memcpy(targets + r * len, window + 1, len * sizeof *targets);
    }
    dl->next += rows;
    return rows;
}
