#include "lenet_int8.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

static int load_mem(const char *path, int8_t *values, size_t count)
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
        unsigned long value = strtoul(token, &end, 16);
        if (errno || *end || token[0] == '-' || value > 255) {
            fprintf(stderr, "%s: invalid element %zu\n", path, index);
            fclose(stream);
            return -1;
        }
        values[index] = (int8_t)(value >= 128 ? (int)value - 256 : (int)value);
    }
    char extra[2];
    int trailing = fscanf(stream, "%1s", extra);
    int failed = ferror(stream) || trailing != EOF;
    fclose(stream);
    if (failed) fprintf(stderr, "%s: trailing data or read error\n", path);
    return failed ? -1 : 0;
}

static int compare_layer(const char *name, const int8_t *values,
                          size_t count, void *context)
{
    int sample = *(const int *)context;
    int8_t expected[3456];
    char path[256];
    if (count > sizeof(expected)) return -1;
    snprintf(path, sizeof(path), "data/test_vectors/int8/sample_%04d/%s.mem", sample, name);
    if (load_mem(path, expected, count)) return -1;
    for (size_t index = 0; index < count; ++index) {
        if (values[index] != expected[index]) {
            fprintf(stderr, "sample %04d %s[%zu]: C=%d Python=%d\n",
                    sample, name, index, values[index], expected[index]);
            return -1;
        }
    }
    printf("sample %04d %-7s: PASS (%zu values)\n", sample, name, count);
    return 0;
}

int main(int argc, char **argv)
{
    long samples = 10;
    if (argc > 2) {
        fprintf(stderr, "Usage: test_lenet [sample_count]\n");
        return EXIT_FAILURE;
    }
    if (argc == 2) {
        char *end;
        errno = 0;
        samples = strtol(argv[1], &end, 10);
        if (errno || end == argv[1] || *end || samples < 1 || samples > 10000) {
            fprintf(stderr, "sample_count must be 1..10000; vectors must exist\n");
            return EXIT_FAILURE;
        }
    }
    static lenet_workspace workspace;
    for (int sample = 0; sample < samples; ++sample) {
        char path[256];
        int8_t input[784];
        int8_t scores[10];
        int prediction;
        snprintf(path, sizeof(path), "data/test_vectors/int8/sample_%04d/input.mem", sample);
        if (load_mem(path, input, 784)) return EXIT_FAILURE;
        int status = lenet_predict(input, &workspace, scores, &prediction, compare_layer, &sample);
        if (status) {
            fprintf(stderr, "sample %04d inference failed: %d\n", sample, status);
            return EXIT_FAILURE;
        }
        printf("sample %04d prediction: %d\n", sample, prediction);
    }
    puts("All full-network INT8 layer checks passed.");
    return EXIT_SUCCESS;
}
