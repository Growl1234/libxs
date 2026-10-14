/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_sync.h>
#include "libxs_main.h"

#include <stdint.h>
#if defined(_WIN32)
# include <process.h>
#else
# include <sys/file.h>
# include <sys/stat.h>
# include <fcntl.h>
# include <unistd.h>
# include <time.h>
#endif

#if !defined(INTERNAL_LIBXS_SINGLETON_MAX)
# define INTERNAL_LIBXS_SINGLETON_MAX 8
#endif
#if !defined(INTERNAL_LIBXS_SINGLETON_ATTEMPTS)
# define INTERNAL_LIBXS_SINGLETON_ATTEMPTS 4
#endif
#if !defined(_WIN32) && defined(O_CLOEXEC)
# define INTERNAL_LIBXS_SINGLETON_OFLAGS (O_RDWR | O_CREAT | O_CLOEXEC)
#elif !defined(_WIN32)
# define INTERNAL_LIBXS_SINGLETON_OFLAGS (O_RDWR | O_CREAT)
#endif


LIBXS_EXTERN_C typedef struct internal_libxs_singleton_t {
#if defined(_WIN32)
  HANDLE handle;
#else
  char fname[256];
  int handle;
#endif
  int state; /* 0: free, 1: claimed */
} internal_libxs_singleton_t;


LIBXS_APIVAR_DEFINE(internal_libxs_singleton_t internal_libxs_singletons[INTERNAL_LIBXS_SINGLETON_MAX]);


LIBXS_API unsigned int libxs_nranks(void)
{
  /* Intel MPI and MPICH, Open MPI, MVAPICH2 */
  const char* env_nranks = getenv("MPI_LOCALNRANKS");
  int nranks;
  if (NULL == env_nranks) env_nranks = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
  if (NULL == env_nranks) env_nranks = getenv("MV2_COMM_WORLD_LOCAL_SIZE");
  nranks = (NULL == env_nranks ? 1 : atoi(env_nranks));
  return (unsigned int)LIBXS_MAX(nranks, 1);
}


LIBXS_API unsigned int libxs_nrank(void)
{
  /* the global PMI_RANK is the last resort: it is local only if ranks are placed in blocks */
  const char* env_rank = getenv("MPI_LOCALRANKID");
  int nrank;
  if (NULL == env_rank) env_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (NULL == env_rank) env_rank = getenv("MV2_COMM_WORLD_LOCAL_RANK");
  if (NULL == env_rank) env_rank = getenv("SLURM_LOCALID");
  if (NULL == env_rank) env_rank = getenv("PMI_RANK");
  nrank = (NULL == env_rank ? 0 : atoi(env_rank));
  return (unsigned int)LIBXS_MAX(nrank, 0) % libxs_nranks();
}


LIBXS_API unsigned int libxs_rid(void)
{
  return 1 < libxs_nranks() ? libxs_nrank() : libxs_pid();
}


LIBXS_API unsigned int libxs_pid(void)
{
#if defined(_WIN32)
  return (unsigned int)_getpid();
#else
  return (unsigned int)getpid();
#endif
}


LIBXS_API unsigned int libxs_tid(void)
{
#if (0 != LIBXS_SYNC)
  static LIBXS_TLS unsigned int tid = 0xFFFFFFFF;
  if (0xFFFFFFFF == tid) tid = LIBXS_ATOMIC_ADD_FETCH(&libxs_thread_count, 1, LIBXS_ATOMIC_RELAXED) - 1;
  return tid;
#else
  return 0;
#endif
}


LIBXS_API void libxs_stdio_acquire(void)
{
#if !defined(_WIN32)
  if (0 < libxs_stdio_handle) {
    flock(libxs_stdio_handle - 1, LOCK_EX);
  }
  else
#endif
  {
    LIBXS_FLOCK(stdout);
    LIBXS_FLOCK(stderr);
  }
}


LIBXS_API void libxs_stdio_release(void)
{
#if !defined(_WIN32)
  if (0 < libxs_stdio_handle) {
    flock(libxs_stdio_handle - 1, LOCK_UN);
  }
  else
#endif
  {
    LIBXS_FUNLOCK(stderr);
    LIBXS_FUNLOCK(stdout);
  }
}


LIBXS_API int libxs_singleton_acquire(const char key[])
{
  int result = -1, slot = 0;
  if (NULL != key && NULL == strchr(key, '/') && NULL == strchr(key, '\\')) {
    for (; slot < INTERNAL_LIBXS_SINGLETON_MAX; ++slot) {
      int expected = 0;
      if (LIBXS_ATOMIC_CMPSWP(&internal_libxs_singletons[slot].state, expected, 1,
        LIBXS_ATOMIC_SEQ_CST))
      {
        break;
      }
    }
  }
  else slot = INTERNAL_LIBXS_SINGLETON_MAX;
  if (INTERNAL_LIBXS_SINGLETON_MAX > slot) {
    internal_libxs_singleton_t *const entry = internal_libxs_singletons + slot;
#if defined(_WIN32)
    char name[256];
    const int nchars = LIBXS_SNPRINTF(name, sizeof(name), "%s%s",
      '\0' != *key ? "LIBXS." : "GlobalLIBXS", key);
    entry->handle = ((0 < nchars && (int)sizeof(name) > nchars)
      ? CreateMutexA(NULL, TRUE, name) : NULL);
    if (NULL != entry->handle) {
      /* CreateMutex succeeds for an existing mutex as well, without taking ownership */
      if (ERROR_ALREADY_EXISTS != GetLastError()) result = slot;
      else {
        CloseHandle(entry->handle);
        entry->handle = NULL;
      }
    }
#else
    const unsigned int userid = (unsigned int)getuid();
    const int nchars = LIBXS_SNPRINTF(entry->fname, sizeof(entry->fname), "/tmp/.libxs.%u%s%s",
      /*rely on user id to avoid permission issues in case of left-over files*/userid,
      '\0' != *key ? "." : "", key);
    int attempt = 0;
    entry->handle = -1;
    if (0 < nchars && (int)sizeof(entry->fname) > nchars) {
      for (; attempt < INTERNAL_LIBXS_SINGLETON_ATTEMPTS && 0 > result; ++attempt) {
        const int fd = open(entry->fname, INTERNAL_LIBXS_SINGLETON_OFLAGS, S_IRUSR | S_IWUSR);
        if (0 > fd) break;
        /* flock belongs to the open file: a second acquisition in the same process fails as well */
        if (0 == flock(fd, LOCK_EX | LOCK_NB)) {
          struct stat file_stat, path_stat;
          /**
           * A releasing owner unlinks the file while holding the lock. A lock taken on a
           * file that is no longer at the path is stale, and two processes would own the key.
           */
          if (0 == fstat(fd, &file_stat) && 0 == stat(entry->fname, &path_stat)
            && file_stat.st_ino == path_stat.st_ino && file_stat.st_dev == path_stat.st_dev)
          {
            entry->handle = fd;
            result = slot;
          }
          else close(fd);
        }
        else {
          close(fd);
          break; /* owned by another process */
        }
      }
    }
#endif
    if (0 > result) LIBXS_ATOMIC_STORE(&entry->state, 0, LIBXS_ATOMIC_SEQ_CST);
  }
  return result;
}


LIBXS_API void libxs_singleton_release(int handle)
{
  if (0 <= handle && INTERNAL_LIBXS_SINGLETON_MAX > handle
    && 0 != LIBXS_ATOMIC_LOAD(&internal_libxs_singletons[handle].state, LIBXS_ATOMIC_SEQ_CST))
  {
    internal_libxs_singleton_t *const entry = internal_libxs_singletons + handle;
#if defined(_WIN32)
    ReleaseMutex(entry->handle);
    CloseHandle(entry->handle);
    entry->handle = NULL;
#else
    unlink(entry->fname); /* before the lock is dropped, see libxs_singleton_acquire */
    close(entry->handle);
    entry->handle = -1;
#endif
    LIBXS_ATOMIC_STORE(&entry->state, 0, LIBXS_ATOMIC_SEQ_CST);
  }
}
