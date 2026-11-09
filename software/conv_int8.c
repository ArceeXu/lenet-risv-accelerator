#include "conv_int8.h"

#include <limits.h>
#include <stddef.h>

int lenet_requantize(int32_t accumulator, int32_t multiplier,
                     int32_t shift, int relu, int8_t *output)
{
    if (!output || multiplier < 0 || shift < 0 || shift > 62) {
        return -1;
    }
    int64_t product = (int64_t)accumulator * multiplier;
    uint64_t magnitude = (uint64_t)(product < 0 ? -product : product);
    if (shift > 0) {
        magnitude = (magnitude + (UINT64_C(1) << (shift - 1))) >> shift;
    }
    int64_t rounded = product < 0 ? -(int64_t)magnitude : (int64_t)magnitude;
    int lower = relu ? 0 : -127;
    if (rounded < lower) {
        rounded = lower;
    }
    if (rounded > 127) {
        rounded = 127;
    }
    *output = (int8_t)rounded;
    return 0;
}

int lenet_conv_int8(const int8_t *input, const int8_t *weight,
                    const int32_t *bias, const int32_t *multiplier,
                    const int32_t *shift, int channels, int height,
                    int width, int outputs, int kernel, int relu,
                    int32_t *accumulators, int8_t *result)
{
    if (!input || !weight || !bias || !multiplier || !shift || !result ||
        channels <= 0 || channels > 6 || outputs <= 0 || outputs > 16 ||
        height > 28 || width > 28 || kernel != 5 ||
        height < kernel || width < kernel) {
        return -1;
    }
    for (int channel = 0; channel < outputs; ++channel) {
        if (multiplier[channel] < 0 || shift[channel] < 0 || shift[channel] > 62) {
            return -1;
        }
    }
    int output_height = height - kernel + 1;
    int output_width = width - kernel + 1;
    for (int out_channel = 0; out_channel < outputs; ++out_channel) {
        for (int row = 0; row < output_height; ++row) {
            for (int column = 0; column < output_width; ++column) {
                int64_t accumulator = bias[out_channel];
                for (int in_channel = 0; in_channel < channels; ++in_channel) {
                    for (int kernel_row = 0; kernel_row < kernel; ++kernel_row) {
                        for (int kernel_column = 0; kernel_column < kernel; ++kernel_column) {
                            size_t input_index = ((size_t)in_channel * height + row + kernel_row)
                                                 * width + column + kernel_column;
                            size_t weight_index = (((size_t)out_channel * channels + in_channel)
                                                  * kernel + kernel_row) * kernel + kernel_column;
                            accumulator += (int32_t)input[input_index] * weight[weight_index];
                        }
                    }
                }
                if (accumulator < INT32_MIN || accumulator > INT32_MAX) {
                    return -2;
                }
                size_t output_index = ((size_t)out_channel * output_height + row)
                                      * output_width + column;
                if (accumulators) {
                    accumulators[output_index] = (int32_t)accumulator;
                }
                lenet_requantize((int32_t)accumulator, multiplier[out_channel],
                                 shift[out_channel], relu, &result[output_index]);
            }
        }
    }
    return 0;
}
