#ifndef LLM_DATALOADER_H
#define LLM_DATALOADER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * Sliding-window batches of (input, target) token rows for next-token
 * prediction: GPTDatasetV1 + create_dataloader_v1 from section 2.6.
 *
 * Windows start at 0, stride, 2*stride, ... as long as a target token is left
 * after the window. A row's target is its input shifted one token right:
 *
 *   ids:     40  367  2885  1464  1807 ...
 *   input:  [40, 367, 2885, 1464]
 *   target:     [367, 2885, 1464, 1807]
 */
typedef struct DataLoader DataLoader;

typedef struct {
    size_t batch_size;  /* rows per batch */
    size_t max_length;  /* tokens per row (the context length) */
    size_t stride;      /* tokens between the starts of consecutive windows */
    bool shuffle;       /* visit windows in a new random order every epoch */
    bool drop_last;     /* skip a final batch smaller than batch_size */
    uint64_t seed;      /* seeds the shuffle */
} DataLoaderConfig;

/*
 * The loader reads ids in place, so they must outlive it. Returns NULL if
 * batch_size, max_length or stride is 0, or if memory runs out. The first
 * epoch is ready to iterate.
 */
DataLoader *dataloader_create(const int *ids, size_t n_ids, DataLoaderConfig cfg);
void dataloader_free(DataLoader *dl);

/* Number of windows (len(dataset)) and batches per epoch (len(dataloader)). */
size_t dataloader_num_windows(const DataLoader *dl);
size_t dataloader_num_batches(const DataLoader *dl);

/* Starts a new epoch, reshuffling if cfg.shuffle is set. */
void dataloader_reset(DataLoader *dl);

/*
 * Copies the next batch into inputs and targets, each with room for
 * batch_size * max_length ids, row by row. Returns the number of rows filled,
 * which is below batch_size only for a final partial batch, or 0 once the
 * epoch is over.
 */
size_t dataloader_next(DataLoader *dl, int *inputs, int *targets);

#endif
