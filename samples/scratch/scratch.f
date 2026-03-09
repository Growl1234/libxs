!=======================================================================!
! Copyright (c) Intel Corporation - All rights reserved.                !
! This file is part of the LIBXS library.                               !
!                                                                       !
! For information on the license, see the LICENSE file.                 !
! Further information: https://github.com/hfp/libxs/                    !
! SPDX-License-Identifier: BSD-3-Clause                                 !
!=======================================================================!

      PROGRAM scratch
        USE :: LIBXS, ONLY: LIBXS_TIMER_TICK_KIND,                      &
     &                        libxs_timer_duration,                     &
     &                        libxs_timer_tick,                         &
     &                        libxs_malloc_pool,                        &
     &                        libxs_malloc,                             &
     &                        libxs_free,                               &
     &                        libxs_free_pool,                          &
     &                        libxs_finalize,                           &
     &                        libxs_init
        USE, INTRINSIC :: ISO_C_BINDING, ONLY:                          &
     &    C_PTR, C_NULL_PTR, C_NULL_FUNPTR,                             &
     &    C_SIZE_T, C_ASSOCIATED
        IMPLICIT NONE

        INTEGER, PARAMETER :: W = 48
        INTEGER, PARAMETER :: NRPT_DEFAULT = 20
        INTEGER, PARAMETER :: MBYTES_DEFAULT = 100

        INTEGER(LIBXS_TIMER_TICK_KIND) :: start
        TYPE(C_PTR) :: pool, ptr
        DOUBLE PRECISION :: d_alloc, d_scratch, d
        INTEGER(C_SIZE_T) :: nbytes
        INTEGER :: i, nrepeat, mbytes
        INTEGER :: nerrors_alloc, nerrors_scratch

        CHARACTER(32) :: argv
        INTEGER :: argc

        argc = COMMAND_ARGUMENT_COUNT()
        IF (1 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(1, argv)
          READ(argv, "(I32)") mbytes
        ELSE
          mbytes = 0
        END IF
        IF (2 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(2, argv)
          READ(argv, "(I32)") nrepeat
        ELSE
          nrepeat = 0
        END IF

        mbytes = MERGE(mbytes, MBYTES_DEFAULT, 0 < mbytes)
        nrepeat = MERGE(nrepeat, NRPT_DEFAULT, 0 < nrepeat)
        nbytes = INT(mbytes, C_SIZE_T) * ISHFT(INT(1, C_SIZE_T), 20)

        d_alloc = 0D0
        d_scratch = 0D0
        nerrors_alloc = 0
        nerrors_scratch = 0

        CALL libxs_init()
        pool = libxs_malloc_pool(C_NULL_FUNPTR, C_NULL_FUNPTR)

        WRITE(*, "(A,I0,A,I0,A)")                                       &
     &    "Allocating ", mbytes, " MB, ", nrepeat, " repetitions"

        ! warm-up (demand-page, ramp CPU frequency)
        ptr = libxs_malloc(pool, nbytes)
        CALL libxs_free(ptr)
        BLOCK
          DOUBLE PRECISION, ALLOCATABLE :: tmp(:)
          ALLOCATE(tmp(nbytes / 8))
          DEALLOCATE(tmp)
        END BLOCK

        WRITE(*, "(A)") REPEAT("-", W)
        DO i = 1, nrepeat
          ! benchmark Fortran ALLOCATE/DEALLOCATE
          BLOCK
            DOUBLE PRECISION, ALLOCATABLE :: buf(:)
            start = libxs_timer_tick()
            ALLOCATE(buf(nbytes / 8))
            d = libxs_timer_duration(start, libxs_timer_tick())
            d_alloc = d_alloc + d
            start = libxs_timer_tick()
            DEALLOCATE(buf)
            d = d + libxs_timer_duration(start, libxs_timer_tick())
          END BLOCK
          WRITE(*, "(A,F14.3,A)") "ALLOCATE (Fortran):",                &
     &      1D3 * d, " ms"

          ! benchmark libxs scratch pool
          start = libxs_timer_tick()
          ptr = libxs_malloc(pool, nbytes)
          d = libxs_timer_duration(start, libxs_timer_tick())
          d_scratch = d_scratch + d
          IF (.NOT. C_ASSOCIATED(ptr)) THEN
            nerrors_scratch = nerrors_scratch + 1
          END IF
          start = libxs_timer_tick()
          CALL libxs_free(ptr)
          d = d + libxs_timer_duration(start, libxs_timer_tick())
          WRITE(*, "(A,F14.3,A)") "libxs_malloc+free: ",                &
     &      1D3 * d, " ms"
          WRITE(*, "(A)") REPEAT("-", W)
        END DO

        IF (0 < d_alloc .AND. 0 < d_scratch) THEN
          WRITE(*, "(A,I0,A)") "Average over ", nrepeat,                &
     &      " iterations (malloc only)"
          WRITE(*, "(A)") REPEAT("-", W)
          WRITE(*, "(A,F14.3,A)") "ALLOCATE (Fortran):",                &
     &      1D3 * d_alloc / DBLE(nrepeat), " ms"
          WRITE(*, "(A,F14.3,A)") "libxs_malloc:      ",                &
     &      1D3 * d_scratch / DBLE(nrepeat), " ms"
          WRITE(*, "(A,F14.1,A)") "Speedup:           ",                &
     &      d_alloc / d_scratch, "x"
          WRITE(*, "(A)") REPEAT("-", W)
        END IF

        CALL libxs_free_pool(pool)
        CALL libxs_finalize()

        IF (0 < nerrors_alloc .OR. 0 < nerrors_scratch) THEN
          WRITE(*, "(A,I0,A,I0,A)")                                     &
     &      "FAILED (errors: alloc=", nerrors_alloc,                    &
     &      " libxs=", nerrors_scratch, ")"
          ERROR STOP 1
        END IF
      END PROGRAM
