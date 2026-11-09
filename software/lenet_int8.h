#ifndef LENET_INT8_H
#define LENET_INT8_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int8_t first[3456];
    int8_t second[3456];
} lenet_workspace;

typedef int (*lenet_observer)(const char *name, const int8_t *values,
                              size_t count, void *context);

int lenet_max_pool(const int8_t *input, int channels, int height,
                   int width, int8_t *output);
int lenet_linear(const int8_t *input, const int8_t *weight,
                  const int32_t *bias, const int32_t *multiplier,
                  const int32_t *shift, int inputs, int outputs, int relu,
                  int32_t *accumulators, int8_t *result);
int lenet_predict(const int8_t *input, lenet_workspace *workspace,
                   int8_t scores[10], int *prediction,
                   lenet_observer observer, void *context);

#endif
