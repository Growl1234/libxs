!=======================================================================!
! Copyright (c) Intel Corporation - All rights reserved.                !
! This file is part of the LIBXS library.                               !
!                                                                       !
! For information on the license, see the LICENSE file.                 !
! Further information: https://github.com/hfp/libxs/                    !
! SPDX-License-Identifier: BSD-3-Clause                                 !
!=======================================================================!

      PROGRAM matcopy
        USE :: LIBXS, ONLY: LIBXS_TIMER_TICK_KIND,                      &
     &                        libxs_timer_duration,                     &
     &                        libxs_timer_tick,                         &
     &                        libxs_finalize,                           &
     &                        libxs_init,                               &
     &                        libxs_matcopy,                            &
     &                        libxs_matcopy_task,                       &
     &                        libxs_otrans,                             &
     &                        libxs_otrans_task
        USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_LOC,                   &
     &    C_NULL_PTR
!$      USE :: OMP_LIB, ONLY: omp_get_thread_num,                       &
!$   &    omp_get_num_threads
        IMPLICIT NONE

        INTEGER, PARAMETER :: T = KIND(0D0)
        INTEGER, PARAMETER :: S = T
        INTEGER, PARAMETER :: W = 50

        REAL(T), ALLOCATABLE, TARGET :: a1(:), b1(:)
        !DIR$ ATTRIBUTES ALIGN:64 :: a1, b1
        REAL(T), POINTER :: an(:,:,:), bn(:,:,:)
        INTEGER(LIBXS_TIMER_TICK_KIND) :: start
        DOUBLE PRECISION :: d, duration(9)
        INTEGER :: m, n, ldi, ldo, h, i, j, k
        INTEGER :: argc, nrepeat, nmb, nbytes, ncount
        INTEGER :: error
        CHARACTER(32) :: argv

        argc = COMMAND_ARGUMENT_COUNT()
        IF (1 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(1, argv)
          READ(argv, "(I32)") m
        ELSE
          m = 4096
        END IF
        IF (2 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(2, argv)
          READ(argv, "(I32)") n
        ELSE
          n = m
        END IF
        IF (3 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(3, argv)
          READ(argv, "(I32)") ldi
          ldi = MAX(ldi, m)
        ELSE
          ldi = m
        END IF
        IF (4 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(4, argv)
          READ(argv, "(I32)") ldo
          ldo = MAX(ldo, m)
        ELSE
          ldo = ldi
        END IF
        IF (5 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(5, argv)
          READ(argv, "(I32)") nrepeat
        ELSE
          nrepeat = 2
        END IF
        IF (6 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(6, argv)
          READ(argv, "(I32)") nmb
          IF (0 .GE. nmb) nmb = 2048
        ELSE
          nmb = 2048
        END IF

        nbytes = m * n * S
        k = INT(ISHFT(INT(nmb, 8), 20) / INT(nbytes, 8))
        IF (0 .GE. k) k = 1

        WRITE(*, "(3(A,I0),2(A,I0),A,I0)")                              &
     &    "m=", m, " n=", n, " k=", k,                                  &
     &    " ldi=", ldi, " ldo=", ldo,                                   &
     &    " size_mb=", INT(k,8)*INT(nbytes,8)/ISHFT(1,20)
        CALL libxs_init()

        ALLOCATE(a1(ldi * n * k), b1(ldo * n * k))
        an(1:ldi, 1:n, 1:k) => a1
        bn(1:ldo, 1:n, 1:k) => b1

        !$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(h, i, j)                &
        !$OMP   SHARED(n, k, ldi, ldo, an, bn)
        DO h = 1, k
          DO j = 1, n
            DO i = 1, ldi
              an(i,j,h) = REAL(INT(j-1,8)*INT(ldi,8)                    &
     &                       + INT(i-1,8), T)
            END DO
            DO i = 1, ldo
              bn(i, j, h) = REAL(-1, T)
            END DO
          END DO
        END DO

        error = 0
        duration = 0D0
        WRITE(*, "(A)") REPEAT("-", W)
        DO i = 1, nrepeat
          ! (1) LIBXS matcopy (serial)
          start = libxs_timer_tick()
          DO h = 1, k
            CALL libxs_matcopy(C_LOC(bn(1,1,h)),                        &
     &        C_LOC(an(1,1,h)), S, m, n, ldi, ldo)
          END DO
          d = libxs_timer_duration(start, libxs_timer_tick())
          IF (0 .GE. d) THEN; error = 1; EXIT; END IF
          IF (1 .LT. i) duration(1) = duration(1) + d
          WRITE(*, "(A,A10,F12.1,A,F12.1,A)")                           &
     &      "LIBXS  ", "(copy): ", 1D3 * d, " ms ",                     &
     &      REAL(2*k,8) * REAL(nbytes,8) /                              &
     &      (REAL(ISHFT(1,20),8) * d), " MB/s"

          ! (2) Fortran array section copy
          start = libxs_timer_tick()
          DO h = 1, k
            bn(1:m, :, h) = an(1:m, :, h)
          END DO
          d = libxs_timer_duration(start, libxs_timer_tick())
          IF (0 .GE. d) THEN; error = 2; EXIT; END IF
          IF (1 .LT. i) duration(2) = duration(2) + d
          WRITE(*, "(A,A10,F12.1,A,F12.1,A)")                           &
     &      "Fortran", "(copy): ", 1D3 * d, " ms ",                     &
     &      REAL(2*k,8) * REAL(nbytes,8) /                              &
     &      (REAL(ISHFT(1,20),8) * d), " MB/s"

          ! (3) LIBXS matcopy_task (threaded copy)
          start = libxs_timer_tick()
          DO h = 1, k
            !$OMP PARALLEL DEFAULT(NONE)                                &
            !$OMP   SHARED(h, m, n, ldi, ldo, an, bn)
            CALL libxs_matcopy_task(C_LOC(bn(1,1,h)),                   &
     &        C_LOC(an(1,1,h)), S, m, n, ldi, ldo,                      &
     &        omp_get_thread_num(), omp_get_num_threads())
            !$OMP END PARALLEL
          END DO
          d = libxs_timer_duration(start, libxs_timer_tick())
          IF (0 .GE. d) THEN; error = 3; EXIT; END IF
          IF (1 .LT. i) duration(3) = duration(3) + d
          WRITE(*, "(A,A10,F12.1,A,F12.1,A)")                           &
     &      "LIBXS  ", "(cpyt): ", 1D3 * d, " ms ",                     &
     &      REAL(2*k,8) * REAL(nbytes,8) /                              &
     &      (REAL(ISHFT(1,20),8) * d), " MB/s"

          ! (4) LIBXS matcopy zero (NULL source)
          start = libxs_timer_tick()
          DO h = 1, k
            CALL libxs_matcopy(C_LOC(bn(1,1,h)),                        &
     &        C_NULL_PTR, S, m, n, ldi, ldo)
          END DO
          d = libxs_timer_duration(start, libxs_timer_tick())
          IF (0 .GE. d) THEN; error = 4; EXIT; END IF
          IF (1 .LT. i) duration(4) = duration(4) + d
          WRITE(*, "(A,A10,F12.1,A,F12.1,A)")                           &
     &      "LIBXS  ", "(zero): ", 1D3 * d, " ms ",                     &
     &      REAL(1*k,8) * REAL(nbytes,8) /                              &
     &      (REAL(ISHFT(1,20),8) * d), " MB/s"

          ! (5) Fortran array section zero
          start = libxs_timer_tick()
          DO h = 1, k
            bn(1:m, :, h) = REAL(0, T)
          END DO
          d = libxs_timer_duration(start, libxs_timer_tick())
          IF (0 .GE. d) THEN; error = 5; EXIT; END IF
          IF (1 .LT. i) duration(5) = duration(5) + d
          WRITE(*, "(A,A10,F12.1,A,F12.1,A)")                           &
     &      "Fortran", "(zero): ", 1D3 * d, " ms ",                     &
     &      REAL(1*k,8) * REAL(nbytes,8) /                              &
     &      (REAL(ISHFT(1,20),8) * d), " MB/s"

          ! (6) LIBXS matcopy_task zero (threaded)
          start = libxs_timer_tick()
          DO h = 1, k
            !$OMP PARALLEL DEFAULT(NONE)                                &
            !$OMP   SHARED(h, m, n, ldi, ldo, bn)
            CALL libxs_matcopy_task(C_LOC(bn(1,1,h)),                   &
     &        C_NULL_PTR, S, m, n, ldi, ldo,                            &
     &        omp_get_thread_num(), omp_get_num_threads())
            !$OMP END PARALLEL
          END DO
          d = libxs_timer_duration(start, libxs_timer_tick())
          IF (0 .GE. d) THEN; error = 6; EXIT; END IF
          IF (1 .LT. i) duration(6) = duration(6) + d
          WRITE(*, "(A,A10,F12.1,A,F12.1,A)")                           &
     &      "LIBXS  ", "(zerot): ", 1D3 * d, " ms ",                    &
     &      REAL(1*k,8) * REAL(nbytes,8) /                              &
     &      (REAL(ISHFT(1,20),8) * d), " MB/s"

          ! (7-9) Transpose tests (only if square)
          IF (m .EQ. n .AND. ldi .EQ. ldo) THEN
            ! (7) LIBXS otrans (serial)
            start = libxs_timer_tick()
            DO h = 1, k
              CALL libxs_otrans(C_LOC(bn(1,1,h)),                       &
     &          C_LOC(an(1,1,h)), S, m, n, ldi, ldo)
            END DO
            d = libxs_timer_duration(start, libxs_timer_tick())
            IF (0 .GE. d) THEN; error = 7; EXIT; END IF
            IF (1 .LT. i) duration(7) = duration(7) + d
            WRITE(*, "(A,A10,F12.1,A,F12.1,A)")                         &
     &        "LIBXS  ", "(trans): ", 1D3 * d, " ms ",                  &
     &        REAL(2*k,8) * REAL(nbytes,8) /                            &
     &        (REAL(ISHFT(1,20),8) * d), " MB/s"

            ! (8) Fortran TRANSPOSE
            start = libxs_timer_tick()
            DO h = 1, k
              bn(1:n, 1:m, h) = TRANSPOSE(an(1:m, 1:n, h))
            END DO
            d = libxs_timer_duration(start, libxs_timer_tick())
            IF (0 .GE. d) THEN; error = 8; EXIT; END IF
            IF (1 .LT. i) duration(8) = duration(8) + d
            WRITE(*, "(A,A10,F12.1,A,F12.1,A)")                         &
     &        "Fortran", "(trans): ", 1D3 * d, " ms ",                  &
     &        REAL(2*k,8) * REAL(nbytes,8) /                            &
     &        (REAL(ISHFT(1,20),8) * d), " MB/s"

            ! (9) LIBXS otrans_task (threaded)
            start = libxs_timer_tick()
            DO h = 1, k
              !$OMP PARALLEL DEFAULT(NONE)                              &
              !$OMP   SHARED(h, m, n, ldi, ldo, an, bn)
              CALL libxs_otrans_task(C_LOC(bn(1,1,h)),                  &
     &          C_LOC(an(1,1,h)), S, m, n, ldi, ldo,                    &
     &          omp_get_thread_num(), omp_get_num_threads())
              !$OMP END PARALLEL
            END DO
            d = libxs_timer_duration(start, libxs_timer_tick())
            IF (0 .GE. d) THEN; error = 9; EXIT; END IF
            IF (1 .LT. i) duration(9) = duration(9) + d
            WRITE(*, "(A,A10,F12.1,A,F12.1,A)")                         &
     &        "LIBXS  ", "(transt): ", 1D3 * d, " ms ",                 &
     &        REAL(2*k,8) * REAL(nbytes,8) /                            &
     &        (REAL(ISHFT(1,20),8) * d), " MB/s"
          END IF

          WRITE(*, "(A)") REPEAT("-", W)
        END DO

        DEALLOCATE(a1, b1)

        IF (0 .EQ. error) THEN
          ncount = MERGE(nrepeat - 1, nrepeat, 2 .LT. nrepeat)
          IF (1 .LT. ncount) THEN
            WRITE(*, "(A,I0,A)")                                        &
     &        "Arithmetic average of ", ncount, " iterations"
            WRITE(*, "(A)") REPEAT("-", W)
            IF (0 .LT. duration(1)) THEN
              WRITE(*, "(A,A10,F12.1,A)") "LIBXS  ", "(copy): ",        &
     &          REAL(2*k*ncount,8) * REAL(nbytes,8) /                   &
     &          (REAL(ISHFT(1,20),8) * duration(1)), " MB/s"
            END IF
            IF (0 .LT. duration(2)) THEN
              WRITE(*, "(A,A10,F12.1,A)") "Fortran", "(copy): ",        &
     &          REAL(2*k*ncount,8) * REAL(nbytes,8) /                   &
     &          (REAL(ISHFT(1,20),8) * duration(2)), " MB/s"
            END IF
            IF (0 .LT. duration(3)) THEN
              WRITE(*, "(A,A10,F12.1,A)") "LIBXS  ", "(cpyt): ",        &
     &          REAL(2*k*ncount,8) * REAL(nbytes,8) /                   &
     &          (REAL(ISHFT(1,20),8) * duration(3)), " MB/s"
            END IF
            IF (0 .LT. duration(4)) THEN
              WRITE(*, "(A,A10,F12.1,A)") "LIBXS  ", "(zero): ",        &
     &          REAL(1*k*ncount,8) * REAL(nbytes,8) /                   &
     &          (REAL(ISHFT(1,20),8) * duration(4)), " MB/s"
            END IF
            IF (0 .LT. duration(5)) THEN
              WRITE(*, "(A,A10,F12.1,A)") "Fortran", "(zero): ",        &
     &          REAL(1*k*ncount,8) * REAL(nbytes,8) /                   &
     &          (REAL(ISHFT(1,20),8) * duration(5)), " MB/s"
            END IF
            IF (0 .LT. duration(6)) THEN
              WRITE(*, "(A,A10,F12.1,A)") "LIBXS  ", "(zerot): ",       &
     &          REAL(1*k*ncount,8) * REAL(nbytes,8) /                   &
     &          (REAL(ISHFT(1,20),8) * duration(6)), " MB/s"
            END IF
            IF (0 .LT. duration(7)) THEN
              WRITE(*, "(A,A10,F12.1,A)") "LIBXS  ", "(trans): ",       &
     &          REAL(2*k*ncount,8) * REAL(nbytes,8) /                   &
     &          (REAL(ISHFT(1,20),8) * duration(7)), " MB/s"
            END IF
            IF (0 .LT. duration(8)) THEN
              WRITE(*, "(A,A10,F12.1,A)") "Fortran", "(trans): ",       &
     &          REAL(2*k*ncount,8) * REAL(nbytes,8) /                   &
     &          (REAL(ISHFT(1,20),8) * duration(8)), " MB/s"
            END IF
            IF (0 .LT. duration(9)) THEN
              WRITE(*, "(A,A10,F12.1,A)") "LIBXS  ", "(transt): ",      &
     &          REAL(2*k*ncount,8) * REAL(nbytes,8) /                   &
     &          (REAL(ISHFT(1,20),8) * duration(9)), " MB/s"
            END IF
            WRITE(*, "(A)") REPEAT("-", W)
          END IF
        ELSE
          WRITE(*, "(A,I0)") "Error: test failed (code=", error
        END IF

        CALL libxs_finalize()
      END PROGRAM
