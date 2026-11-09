#include "conv_int8.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/weights/int8/conv1_weight.h"
#include "../data/weights/int8/conv1_bias.h"
#include "../data/weights/int8/conv1_multiplier.h"
#include "../data/weights/int8/conv1_shift.h"
#include "../data/weights/int8/conv2_weight.h"
#include "../data/weights/int8/conv2_bias.h"
#include "../data/weights/int8/conv2_multiplier.h"
#include "../data/weights/int8/conv2_shift.h"

static int read_mem(const char *path, void *destination, size_t count, int bits)
{
    FILE *stream = fopen(path, "r");
    if (!stream) {
        perror(path);
        return -1;
    }
    for (size_t index = 0; index < count; ++index) {
        char token[64];
        char *end;
        if (fscanf(stream, "%63s", token) != 1) {
            fprintf(stderr, "%s: missing element %zu\n", path, index);
            fclose(stream);
            return -1;
        }
        errno = 0;
        unsigned long long raw = strtoull(token, &end, 16);
        uint64_t limit = UINT64_C(1) << bits;
        if (errno || *end || token[0] == '-' || raw >= limit) {
            fprintf(stderr, "%s: invalid element %zu\n", path, index);
            fclose(stream);
            return -1;
        }
        int64_t value = raw >= limit / 2 ? (int64_t)raw - (int64_t)limit : (int64_t)raw;
        if (bits == 8) {
            ((int8_t *)destination)[index] = (int8_t)value;
        } else {
            ((int32_t *)destination)[index] = (int32_t)value;
        }
    }
    char extra[2];
    int trailing = fscanf(stream, "%1s", extra);
    int failed = ferror(stream) || trailing != EOF;
    fclose(stream);
    if (failed) {
        fprintf(stderr, "%s: trailing data or read error\n", path);
        return -1;
    }
    return 0;
}

static int check_layer(int sample, int second)
{
    int8_t input[864];
    int8_t result[3456];
    int8_t expected[3456];
    int32_t accumulators[3456];
    int32_t expected_accumulators[3456];
    char path[256];
    int layer = second ? 2 : 1;
    size_t count = second ? 1024 : 3456;
    snprintf(path, sizeof(path), "data/test_vectors/int8/sample_%04d/%s.mem",
             sample, second ? "pool1" : "input");
    if (read_mem(path, input, second ? 864 : 784, 8)) {
        return -1;
    }
    snprintf(path, sizeof(path), "data/test_vectors/int8/sample_%04d/conv%d_acc.mem", sample, layer);
    if (read_mem(path, expected_accumulators, count, 32)) {
        return -1;
    }
    snprintf(path, sizeof(path), "data/test_vectors/int8/sample_%04d/relu%d.mem", sample, layer);
    if (read_mem(path, expected, count, 8)) {
        return -1;
    }
    int status = lenet_conv_int8(input, second ? conv2_weight : conv1_weight,
        second ? conv2_bias : conv1_bias, second ? conv2_multiplier : conv1_multiplier,
        second ? conv2_shift : conv1_shift, second ? 6 : 1, second ? 12 : 28,
        second ? 12 : 28, second ? 16 : 6, 5, 1, accumulators, result);
    if (status) {
        fprintf(stderr, "Convolution failed: %d\n", status);
        return -1;
    }
    for (size_t index = 0; index < count; ++index) {
        if (accumulators[index] != expected_accumulators[index] || result[index] != expected[index]) {
            fprintf(stderr, "sample %d conv%d index %zu: acc %" PRId32 "/%" PRId32
                    ", output %d/%d (actual/expected)\n", sample, layer, index,
                    accumulators[index], expected_accumulators[index], result[index], expected[index]);
            return -1;
        }
    }
    printf("sample %04d conv%d: PASS (%zu accumulators + %zu outputs)\n", sample, layer, count, count);
    return 0;
}

int main(int argc, char **argv)
{
    long samples = 10;
    if (argc > 2) {
        fprintf(stderr, "Usage: test_conv [sample_count]\n");
        return EXIT_FAILURE;
    }
    if (argc == 2) {
        char *end;
        errno = 0;
        samples = strtol(argv[1], &end, 10);
        if (errno || end == argv[1] || *end || samples < 1 || samples > 10000) {
            fprintf(stderr, "sample_count must be 1..10000\n");
            return EXIT_FAILURE;
        }
    }
    for (int sample = 0; sample < samples; ++sample) {
        if (check_layer(sample, 0) || check_layer(sample, 1)) {
            return EXIT_FAILURE;
        }
    }
    puts("All convolution checks passed.");
    return EXIT_SUCCESS;
}
