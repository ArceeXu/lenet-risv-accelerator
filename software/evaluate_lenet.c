#include "lenet_int8.h"

#include <stdio.h>
#include <stdlib.h>

static int read_u32(FILE *stream, uint32_t *value)
{
    unsigned char bytes[4];
    if (fread(bytes, 1, 4, stream) != 4) return -1;
    *value = ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16)
             | ((uint32_t)bytes[2] << 8) | bytes[3];
    return 0;
}

static int at_end(FILE *stream)
{
    return fgetc(stream) == EOF && !ferror(stream);
}

int main(void)
{
    FILE *images = fopen("data/c_eval/images.idx", "rb");
    FILE *labels = fopen("data/c_eval/labels.idx", "rb");
    FILE *reference = fopen("data/c_eval/python_int8_predictions.bin", "rb");
    FILE *report = NULL;
    FILE *metrics = NULL;
    int exit_status = EXIT_FAILURE;
    uint32_t image_magic, label_magic, count, label_count, height, width;
    static lenet_workspace workspace;
    if (!images || !labels || !reference) {
        fprintf(stderr, "Cannot open evaluation data; run python/export_c_eval.py first.\n");
        goto cleanup;
    }
    if (read_u32(images, &image_magic) || read_u32(images, &count) ||
        read_u32(images, &height) || read_u32(images, &width) ||
        read_u32(labels, &label_magic) || read_u32(labels, &label_count) ||
        image_magic != 2051 || label_magic != 2049 || count != 10000 ||
        label_count != count || height != 28 || width != 28) {
        fprintf(stderr, "Invalid MNIST headers; expected 10000 images and labels.\n");
        goto cleanup;
    }
    report = fopen("results/c_validation/full_test_predictions.csv", "w");
    metrics = fopen("results/c_validation/full_test_metrics.json", "w");
    if (!report || !metrics) {
        fprintf(stderr, "Cannot create reports; create results/c_validation first.\n");
        goto cleanup;
    }
    fprintf(report, "index,label,c_prediction,python_int8_prediction,c_correct,agreement\n");
    unsigned int correct = 0;
    unsigned int python_correct = 0;
    unsigned int matches = 0;
    for (uint32_t sample = 0; sample < count; ++sample) {
        unsigned char pixels[784];
        int8_t input[784];
        int8_t scores[10];
        int prediction;
        int label = fgetc(labels);
        int expected = fgetc(reference);
        if (fread(pixels, 1, sizeof(pixels), images) != sizeof(pixels) ||
            label < 0 || label > 9 || expected < 0 || expected > 9) {
            fprintf(stderr, "Truncated or invalid evaluation data at sample %u\n", (unsigned int)sample);
            goto cleanup;
        }
        for (size_t index = 0; index < sizeof(pixels); ++index) {
            input[index] = (int8_t)(((int)pixels[index] * 127 + 127) / 255);
        }
        int status = lenet_predict(input, &workspace, scores, &prediction, NULL, NULL);
        if (status) {
            fprintf(stderr, "Inference failed at sample %u: %d\n", (unsigned int)sample, status);
            goto cleanup;
        }
        correct += prediction == label;
        python_correct += expected == label;
        matches += prediction == expected;
        fprintf(report, "%u,%d,%d,%d,%d,%d\n", (unsigned int)sample, label,
                prediction, expected, prediction == label, prediction == expected);
        if ((sample + 1) % 1000 == 0) {
            printf("%u/10000: C accuracy %.2f%%, Python agreement %.2f%%\n",
                   (unsigned int)(sample + 1), 100.0 * correct / (sample + 1),
                   100.0 * matches / (sample + 1));
            fflush(stdout);
        }
    }
    if (!at_end(images) || !at_end(labels) || !at_end(reference)) {
        fprintf(stderr, "Trailing data or input read error\n");
        goto cleanup;
    }
    fprintf(metrics,
            "{\n  \"samples\": 10000,\n  \"c_correct\": %u,\n"
            "  \"python_int8_correct\": %u,\n  \"prediction_matches\": %u,\n"
            "  \"prediction_mismatches\": %u,\n  \"c_accuracy\": %.8f,\n"
            "  \"python_int8_accuracy\": %.8f,\n  \"prediction_agreement\": %.8f\n}\n",
            correct, python_correct, matches, 10000U - matches,
            correct / 10000.0, python_correct / 10000.0, matches / 10000.0);
    if (ferror(report) || ferror(metrics)) {
        fprintf(stderr, "Failed to write evaluation reports\n");
        goto cleanup;
    }
    printf("C accuracy: %.2f%% (%u/10000)\n", correct / 100.0, correct);
    printf("Python INT8 accuracy: %.2f%%\n", python_correct / 100.0);
    printf("Prediction agreement: %.2f%%; mismatches: %u\n", matches / 100.0, 10000U - matches);
    exit_status = matches == count ? EXIT_SUCCESS : 2;

cleanup:
    if (images) fclose(images);
    if (labels) fclose(labels);
    if (reference) fclose(reference);
    if (report && fclose(report)) exit_status = EXIT_FAILURE;
    if (metrics && fclose(metrics)) exit_status = EXIT_FAILURE;
    return exit_status;
}
