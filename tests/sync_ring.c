/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_source.h>

#if !defined(_WIN32)
# include <sys/mman.h>
# include <sys/types.h>
# include <sys/wait.h>
# include <unistd.h>
#endif

#define TEST_RING_NRECORDS 100000
#define TEST_RING_MAXSIZE 200


/* Size and contents of the i-th record: varying sizes make records wrap at varying offsets. */
static size_t test_ring_size(unsigned int i)
{
  return (size_t)((i * 37U) % TEST_RING_MAXSIZE);
}


static unsigned char test_ring_byte(unsigned int i, size_t j)
{
  return (unsigned char)(i * 131U + (unsigned int)j);
}


static int test_ring_produce(libxs_ring_t* ring, unsigned int i, int timeout_ms)
{
  const size_t size = test_ring_size(i);
  unsigned char* const record = (unsigned char*)libxs_ring_reserve(ring, size, timeout_ms);
  int result = (NULL != record ? EXIT_SUCCESS : EXIT_FAILURE);
  if (EXIT_SUCCESS == result) {
    size_t j;
    for (j = 0; j < size; ++j) record[j] = test_ring_byte(i, j);
    libxs_ring_commit(ring);
  }
  return result;
}


static int test_ring_consume(libxs_ring_t* ring, unsigned int i, int timeout_ms)
{
  size_t size = 0, j;
  const unsigned char* const record = (const unsigned char*)libxs_ring_peek(ring, &size, timeout_ms);
  int result = ((NULL != record && test_ring_size(i) == size) ? EXIT_SUCCESS : EXIT_FAILURE);
  for (j = 0; j < size && EXIT_SUCCESS == result; ++j) {
    if (test_ring_byte(i, j) != record[j]) result = EXIT_FAILURE;
  }
  if (NULL != record) libxs_ring_release(ring);
  return result;
}


/**
 * One side plays both roles: every record wraps the ring once in a while, a full
 * ring and an empty one answer at once without a timeout, and closing still lets
 * the consumer drain what was committed before.
 */
static int test_single(void)
{
  int result = EXIT_SUCCESS;
  const size_t size = libxs_ring_size(1024);
  void* const memory = libxs_malloc(NULL, size, LIBXS_CACHELINE);
  libxs_ring_t* const ring = libxs_ring_init(memory, size, LIBXS_RING_PRIVATE);
  unsigned int i, n = 0;
  if (NULL == ring || NULL != libxs_ring_reserve(ring, 1024, 0) || NULL != libxs_ring_peek(ring, NULL, 0)) {
    result = EXIT_FAILURE;
  }
  for (i = 0; i < 10000 && EXIT_SUCCESS == result; ++i) {
    result = test_ring_produce(ring, i, 0);
    if (EXIT_SUCCESS == result) result = test_ring_consume(ring, i, 0);
  }
  while (EXIT_SUCCESS == result && EXIT_SUCCESS == test_ring_produce(ring, n, 0)) ++n; /* until full */
  if (EXIT_SUCCESS == result && (0 == n || EXIT_SUCCESS == test_ring_produce(ring, n, 1))) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    libxs_ring_close(ring);
    if (NULL != libxs_ring_reserve(ring, 0, 0)) result = EXIT_FAILURE;
  }
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) result = test_ring_consume(ring, i, 0);
  if (EXIT_SUCCESS == result && NULL != libxs_ring_peek(ring, NULL, -1)) result = EXIT_FAILURE;
  libxs_free(memory);
  return result;
}


#if !defined(_WIN32)
/**
 * The producer is another process, writing through its own mapping of the memory.
 * The ring is small against the stream, so both sides keep waiting for each other,
 * which is where a lost wake-up turns into a hang (bounded here by the timeout).
 */
static int test_processes(void)
{
  int result = EXIT_SUCCESS;
  const size_t size = libxs_ring_size(4096);
  void* const memory = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  libxs_ring_t* const ring = (MAP_FAILED != memory ? libxs_ring_init(memory, size, LIBXS_RING_SHARED) : NULL);
  pid_t child = -1;
  if (NULL != ring) child = fork();
  if (0 == child) {
    unsigned int i;
    for (i = 0; i < TEST_RING_NRECORDS && EXIT_SUCCESS == result; ++i) result = test_ring_produce(ring, i, 10000);
    libxs_ring_close(ring);
    _exit(result);
  }
  else if (0 < child) {
    int status = 0;
    unsigned int i;
    for (i = 0; i < TEST_RING_NRECORDS && EXIT_SUCCESS == result; ++i) result = test_ring_consume(ring, i, 10000);
    if (EXIT_SUCCESS == result && NULL != libxs_ring_peek(ring, NULL, 10000)) result = EXIT_FAILURE;
    if (EXIT_SUCCESS != result) libxs_ring_close(ring); /* the producer must not wait for room forever */
    if (child != waitpid(child, &status, 0) || !WIFEXITED(status) || 0 != WEXITSTATUS(status)) result = EXIT_FAILURE;
  }
  else result = EXIT_FAILURE;
  if (MAP_FAILED != memory) munmap(memory, size);
  return result;
}
#endif


int main(void)
{
  int result = test_single();
#if !defined(_WIN32)
  if (EXIT_SUCCESS == result) result = test_processes();
#endif
  return result;
}
