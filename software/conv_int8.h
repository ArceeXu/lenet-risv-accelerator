#ifndef LENET_CONV_INT8_H
#define LENET_CONV_INT8_H

#include <stdint.h>

int lenet_requantize(int32_t accumulator, int32_t multiplier,
                     int32_t shift, int relu, int8_t *output);

int lenet_conv_int8(const int8_t *input, const int8_t *weight,
                    const int32_t *bias, const int32_t *multiplier,
                    const int32_t *shift, int channels, int height,
                    int width, int outputs, int kernel, int relu,
                    int32_t *accumulators, int8_t *result);

#endif
