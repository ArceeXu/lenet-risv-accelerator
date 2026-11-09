#include "lenet_int8.h"
#include "conv_int8.h"

#include <string.h>

#include "../data/weights/int8/conv1_weight.h"
#include "../data/weights/int8/conv1_bias.h"
#include "../data/weights/int8/conv1_multiplier.h"
#include "../data/weights/int8/conv1_shift.h"
#include "../data/weights/int8/conv2_weight.h"
#include "../data/weights/int8/conv2_bias.h"
#include "../data/weights/int8/conv2_multiplier.h"
#include "../data/weights/int8/conv2_shift.h"
#include "../data/weights/int8/fc1_weight.h"
#include "../data/weights/int8/fc1_bias.h"
#include "../data/weights/int8/fc1_multiplier.h"
#include "../data/weights/int8/fc1_shift.h"
#include "../data/weights/int8/fc2_weight.h"
#include "../data/weights/int8/fc2_bias.h"
#include "../data/weights/int8/fc2_multiplier.h"
#include "../data/weights/int8/fc2_shift.h"
#include "../data/weights/int8/fc3_weight.h"
#include "../data/weights/int8/fc3_bias.h"
#include "../data/weights/int8/fc3_multiplier.h"
#include "../data/weights/int8/fc3_shift.h"

static int observe(lenet_observer observer, const char *name,
                   const int8_t *values, size_t count, void *context)
{
    return observer && observer(name, values, count, context) ? -3 : 0;
}

int lenet_predict(const int8_t *input, lenet_workspace *workspace,
                   int8_t scores[10], int *prediction,
                   lenet_observer observer, void *context)
{
    if (!input || !workspace || !scores || !prediction) {
        return -1;
    }
    int8_t *first = workspace->first;
    int8_t *second = workspace->second;
    int status = lenet_conv_int8(input, conv1_weight, conv1_bias, conv1_multiplier,
                                 conv1_shift, 1, 28, 28, 6, 5, 1, NULL, first);
    if (status) return status;
    if (observe(observer, "relu1", first, 3456, context)) return -3;
    status = lenet_max_pool(first, 6, 24, 24, second);
    if (status) return status;
    if (observe(observer, "pool1", second, 864, context)) return -3;
    status = lenet_conv_int8(second, conv2_weight, conv2_bias, conv2_multiplier,
                             conv2_shift, 6, 12, 12, 16, 5, 1, NULL, first);
    if (status) return status;
    if (observe(observer, "relu2", first, 1024, context)) return -3;
    status = lenet_max_pool(first, 16, 8, 8, second);
    if (status) return status;
    if (observe(observer, "pool2", second, 256, context)) return -3;
    if (observe(observer, "flatten", second, 256, context)) return -3;
    status = lenet_linear(second, fc1_weight, fc1_bias, fc1_multiplier,
                          fc1_shift, 256, 120, 1, NULL, first);
    if (status) return status;
    if (observe(observer, "relu3", first, 120, context)) return -3;
    status = lenet_linear(first, fc2_weight, fc2_bias, fc2_multiplier,
                          fc2_shift, 120, 84, 1, NULL, second);
    if (status) return status;
    if (observe(observer, "relu4", second, 84, context)) return -3;
    status = lenet_linear(second, fc3_weight, fc3_bias, fc3_multiplier,
                          fc3_shift, 84, 10, 0, NULL, first);
    if (status) return status;
    if (observe(observer, "fc3", first, 10, context)) return -3;
    memcpy(scores, first, 10);
    *prediction = 0;
    for (int category = 1; category < 10; ++category) {
        if (scores[category] > scores[*prediction]) {
            *prediction = category;
        }
    }
    return 0;
}
