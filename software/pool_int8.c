#include "lenet_int8.h"

int lenet_max_pool(const int8_t *input, int channels, int height,
                   int width, int8_t *output)
{
    if (!input || !output || input == output || channels < 1 || channels > 16 ||
        height < 2 || height > 28 || width < 2 || width > 28 ||
        height % 2 || width % 2) {
        return -1;
    }
    int output_height = height / 2;
    int output_width = width / 2;
    for (int channel = 0; channel < channels; ++channel) {
        for (int row = 0; row < output_height; ++row) {
            for (int column = 0; column < output_width; ++column) {
                size_t base = ((size_t)channel * height + row * 2) * width + column * 2;
                int8_t maximum = input[base];
                for (int offset_row = 0; offset_row < 2; ++offset_row) {
                    for (int offset_column = 0; offset_column < 2; ++offset_column) {
                        int8_t value = input[base + (size_t)offset_row * width + offset_column];
                        if (value > maximum) {
                            maximum = value;
                        }
                    }
                }
                output[((size_t)channel * output_height + row) * output_width + column] = maximum;
            }
        }
    }
    return 0;
}
