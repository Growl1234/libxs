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
# include <sys/stat.h>
# include <sys/types.h>
# include <sys/wait.h>
# include <unistd.h>
#endif


/** A key that names a path, rather than a file in the lock directory, is rejected. */
static int test_invalid(void)
{
  int result = EXIT_SUCCESS;
  if (0 <= libxs_singleton_acquire(NULL)) result = EXIT_FAILURE;
  if (0 <= libxs_singleton_acquire("../escape")) result = EXIT_FAILURE;
  libxs_singleton_release(-1);
  return result;
}


#if !defined(_WIN32)
/** Another process must not acquire what this one owns: the property is per node, not per process. */
static int test_other_process(const char key[])
{
  int result = EXIT_FAILURE;
  const pid_t child = fork();
  if (0 == child) { /* exit code 0: the key was refused */
    const int handle = libxs_singleton_acquire(key);
    _exit(0 > handle ? 0 : 1);
  }
  else if (0 < child) {
    int status = 0;
    if (child == waitpid(child, &status, 0) && WIFEXITED(status) && 0 == WEXITSTATUS(status)) {
      result = EXIT_SUCCESS;
    }
  }
  return result;
}
#endif


/**
 * The owner is refused a second acquisition as well, since an election that the
 * owner wins twice cannot tell a server already running from one to be started.
 * After release the key is free again, and its lock file is gone.
 */
static int test_ownership(void)
{
  char key[64];
  int result = EXIT_SUCCESS, handle = -1;
  if (0 >= LIBXS_SNPRINTF(key, sizeof(key), "test.singleton.%u", libxs_pid())) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    handle = libxs_singleton_acquire(key);
    if (0 > handle || 0 <= libxs_singleton_acquire(key)) result = EXIT_FAILURE;
  }
#if !defined(_WIN32)
  if (EXIT_SUCCESS == result) result = test_other_process(key);
#endif
  libxs_singleton_release(handle);
  if (EXIT_SUCCESS == result) {
    handle = libxs_singleton_acquire(key);
    if (0 > handle) result = EXIT_FAILURE;
    libxs_singleton_release(handle);
  }
#if !defined(_WIN32)
  if (EXIT_SUCCESS == result) {
    char fname[256];
    struct stat path_stat;
    if (0 >= LIBXS_SNPRINTF(fname, sizeof(fname), "/tmp/.libxs.%u.%s", (unsigned int)getuid(), key)
      || 0 == stat(fname, &path_stat))
    {
      result = EXIT_FAILURE;
    }
  }
#endif
  return result;
}


int main(void)
{
  int result = test_invalid();
  if (EXIT_SUCCESS == result) result = test_ownership();
  return result;
}
