#include "lenet_int8.h"
#include "conv_int8.h"

#include <limits.h>

int lenet_linear(const int8_t *input, const int8_t *weight,
                  const int32_t *bias, const int32_t *multiplier,
                  const int32_t *shift, int inputs, int outputs, int relu,
                  int32_t *accumulators, int8_t *result)
{
    if (!input || !weight || !bias || !multiplier || !shift || !result ||
        inputs < 1 || inputs > 256 || outputs < 1 || outputs > 120) {
        return -1;
    }
    for (int neuron = 0; neuron < outputs; ++neuron) {
        if (multiplier[neuron] < 0 || shift[neuron] < 0 || shift[neuron] > 62) {
            return -1;
        }
    }
    for (int neuron = 0; neuron < outputs; ++neuron) {
        int64_t accumulator = bias[neuron];
        for (int feature = 0; feature < inputs; ++feature) {
            accumulator += (int32_t)input[feature] * weight[(size_t)neuron * inputs + feature];
        }
        if (accumulator < INT32_MIN || accumulator > INT32_MAX) {
            return -2;
        }
        if (accumulators) {
            accumulators[neuron] = (int32_t)accumulator;
        }
        int status = lenet_requantize((int32_t)accumulator, multiplier[neuron],
                                     shift[neuron], relu, &result[neuron]);
        if (status) {
            return status;
        }
    }
    return 0;
}
