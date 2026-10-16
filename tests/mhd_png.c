/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_mhd.h>
#include <string.h>

#define WIDTH 4
#define HEIGHT 2
#define PNGFILE "mhd_png.png"
#define MHDFILE "mhd_png.mhd"


/* the pixels of a PNG file as LIBXS writes it: one stored deflate block */
static int png_pixels(const char filename[], unsigned char pixels[])
{
  unsigned char buffer[512];
  FILE* const file = fopen(filename, "rb");
  size_t size = 0, i;
  int result = EXIT_FAILURE;
  if (NULL != file) {
    size = fread(buffer, 1, sizeof(buffer), file);
    fclose(file);
  }
  for (i = 8; i + 4 <= size && EXIT_SUCCESS != result; ++i) {
    if (0 == memcmp(buffer + i, "IDAT", 4)) {
      const size_t at = i + 4 + 2 + 5; /* chunk type, zlib header, block header */
      size_t y;
      if (at + HEIGHT * (WIDTH + 1) <= size) {
        for (y = 0; y < HEIGHT; ++y) { /* each row leads with its filter byte */
          memcpy(pixels + y * WIDTH, buffer + at + y * (WIDTH + 1) + 1, WIDTH);
        }
        result = EXIT_SUCCESS;
      }
      else break;
    }
  }
  return result;
}


static int check(const char what[], const unsigned char got[], const unsigned char want[])
{
  int result = EXIT_SUCCESS, i;
  for (i = 0; i < WIDTH * HEIGHT; ++i) {
    if (got[i] != want[i]) {
      fprintf(stderr, "%s: pixel %i is %u, expected %u\n", what, i,
        (unsigned int)got[i], (unsigned int)want[i]);
      result = EXIT_FAILURE;
    }
  }
  return result;
}


int main(void)
{
  static const double data[WIDTH * HEIGHT] = {
    0.25, 0.5, 0.75, 1.0, 1.5, 0.625, 0.875, 0.5
  };
  /* [0.5, 1] onto [0, 255], what lies outside clamped */
  static const unsigned char fixed[WIDTH * HEIGHT] = { 0, 0, 128, 255, 255, 64, 191, 0 };
  static const double sdata[WIDTH * HEIGHT] = {
    -1.0, -0.5, 0.0, 0.5, 1.0, -0.25, 0.25, 0.75
  };
  /* signed data take all of [0, 255] as well, not only their positive part */
  static const unsigned char linear[WIDTH * HEIGHT] = { 0, 64, 128, 191, 255, 96, 159, 223 };
  /* [0.25, 1.5] onto [0, 255], the extremes of the data */
  static const unsigned char found[WIDTH * HEIGHT] = { 0, 51, 102, 153, 255, 77, 128, 51 };
  const size_t size[2] = { WIDTH, HEIGHT };
  libxs_mhd_element_handler_info_t u8;
  libxs_mhd_write_info_t write_info;
  unsigned char pixels[WIDTH * HEIGHT];
  int result;
  u8.type = LIBXS_DATATYPE_U8;
  u8.hint = LIBXS_MHD_ELEMENT_CONVERSION_DEFAULT;

  memset(&write_info, 0, sizeof(write_info));
  write_info.range[0] = 0.5;
  write_info.range[1] = 1.0;
  result = libxs_png_write(PNGFILE, size, NULL, 1, LIBXS_DATATYPE_F64, data, &write_info);
  if (EXIT_SUCCESS == result) result = png_pixels(PNGFILE, pixels);
  if (EXIT_SUCCESS == result) result = check("fixed range", pixels, fixed);

  /* a conversion to U8 still scales the data, rather than taking its bytes for pixels */
  if (EXIT_SUCCESS == result) {
    memset(&write_info, 0, sizeof(write_info));
    write_info.handler_info = &u8;
    result = libxs_png_write(PNGFILE, size, NULL, 1, LIBXS_DATATYPE_F64, data, &write_info);
  }
  if (EXIT_SUCCESS == result) result = png_pixels(PNGFILE, pixels);
  if (EXIT_SUCCESS == result) result = check("found range", pixels, found);
  if (EXIT_SUCCESS == result) {
    result = libxs_png_write(PNGFILE, size, NULL, 1, LIBXS_DATATYPE_F64, sdata, NULL);
  }
  if (EXIT_SUCCESS == result) result = png_pixels(PNGFILE, pixels);
  if (EXIT_SUCCESS == result) result = check("signed data", pixels, linear);

  /* the MHD conversion reads the same range; its data are the last bytes of the file */
  if (EXIT_SUCCESS == result) {
    libxs_mhd_info_t info = { 2, 1, LIBXS_DATATYPE_F64, 0 };
    write_info.range[0] = 0.5;
    write_info.range[1] = 1.0;
    result = libxs_mhd_write(MHDFILE, NULL, size, NULL, &info, data, &write_info);
  }
  if (EXIT_SUCCESS == result) {
    FILE* const file = fopen(MHDFILE, "rb");
    result = EXIT_FAILURE;
    if (NULL != file) {
      if (0 == fseek(file, -(long)(WIDTH * HEIGHT), SEEK_END)
        && WIDTH * HEIGHT == fread(pixels, 1, WIDTH * HEIGHT, file))
      {
        result = EXIT_SUCCESS;
      }
      fclose(file);
    }
  }
  if (EXIT_SUCCESS == result) result = check("MHD range", pixels, fixed);

  remove(PNGFILE);
  remove(MHDFILE);
  if (EXIT_SUCCESS == result) fprintf(stdout, "OK\n");
  return result;
}
