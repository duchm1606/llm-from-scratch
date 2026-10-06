#ifndef LLM_CH03_INPUTS_H
#define LLM_CH03_INPUTS_H

/* Section 3.3.1: "Your journey starts with one step" as 3-d embeddings, the
 * input of the chapter 3 examples. */
enum { T = 6, D_IN = 3 };

static const float inputs[T * D_IN] = {
    0.43f, 0.15f, 0.89f, /* Your    */
    0.55f, 0.87f, 0.66f, /* journey */
    0.57f, 0.85f, 0.64f, /* starts  */
    0.22f, 0.58f, 0.33f, /* with    */
    0.77f, 0.25f, 0.10f, /* one     */
    0.05f, 0.80f, 0.55f, /* step    */
};

#endif
