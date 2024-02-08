#pragma once

#include "cuda_runtime.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>


typedef struct image_t {
    int32_t width;
    int32_t height;
    void *memory;
} image_t;

image_t OpenImage(const char *FILEPATH)
{
    FILE *file = fopen(FILEPATH, "rb");

    image_t img = { 0 };


    fseek(file, 10, SEEK_SET);
    int32_t offset_to_pixarr = 0;
    fread(&offset_to_pixarr, sizeof(offset_to_pixarr), 1, file);

    fseek(file, 18, SEEK_SET);
    fread(&img.width, sizeof(img.width), 1, file);
    fread(&img.height, sizeof(img.height), 1, file);

    fseek(file, 28, SEEK_SET);
    int16_t bits = 32;
    fread(&bits, sizeof(bits), 1, file);
    void *mem = malloc(sizeof(uint32_t) * img.width * img.height);

    fseek(file, offset_to_pixarr, SEEK_SET);
    int8_t padding = (4 - img.width * (bits / 8) % 4) % 4;
    for (int y = img.height - 1; y >= 0; --y) {
        for (int x = 0; x < img.width; ++x) {
            uint32_t color = 0x00000000;
            uint8_t red = 0x00;
            uint8_t green = 0x00;
            uint8_t blue = 0x00;
            uint8_t alpha = 0xFF;
            fread(&blue, sizeof(blue), 1, file);
            fread(&green, sizeof(green), 1, file);
            fread(&red, sizeof(red), 1, file);
            if (bits == 32) fread(&alpha, sizeof(alpha), 1, file);
            color = (alpha << 24) + (red << 16) + (green << 8) + blue;
            *((uint32_t *)mem + y * img.width + x) = color;
        }

        fseek(file, padding, SEEK_CUR);
    }

    img.memory = (void *)mem;

    fclose(file);
    return img;
}