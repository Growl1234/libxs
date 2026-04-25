#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd -P)
cd "${HERE}/../samples/gemm" 2>/dev/null || exit 1

export CHECK=1

run_check() {
  PROG=$1; shift
  GOLDEN=$1; shift
  L1=$(${PROG} "$@" 2>&1 | sed -n 's/.*checksum=//p')
  if [ "${L1}" != "${GOLDEN}" ]; then
    >&2 echo "FAILED: ${PROG} $* => checksum=${L1} (expected ${GOLDEN})"
    exit 1
  fi
}

# gemm_strided: default shape
run_check ./gemm_strided.x 7.354178e+19
# gemm_strided: small shapes
run_check ./gemm_strided.x 1.201635e+05              1 1 1 100 1
run_check ./gemm_strided.x 1.964242e+11              4 4 4 1000 1
run_check ./gemm_strided.x 2.491947e+11              8 8 8 100 1
# gemm_strided: non-square
run_check ./gemm_strided.x 3.099313e+12              7 13 5 500 1
# gemm_strided: medium/large
run_check ./gemm_strided.x 1.282090e+14              16 16 16 200 1
run_check ./gemm_strided.x 4.045218e+14              23 23 23 100 1
run_check ./gemm_strided.x 4.082186e+15              32 32 32 100 1
# gemm_strided: ld-padding (pad=1)
run_check ./gemm_strided.x 2.218164e+11              8 8 8 100 1 1.0 1
# gemm_strided: ld-padding + beta=0 (pad=2, non-square)
run_check ./gemm_strided.x 1.208802e+12              7 13 5 500 1 0.0 2

# gemm_batch: same shapes (dup=0, no duplicates)
run_check ./gemm_batch.x 1.201635e+05                1 1 1 100 1 0
run_check ./gemm_batch.x 2.491947e+11                8 8 8 100 1 0
run_check ./gemm_batch.x 3.099313e+12                7 13 5 500 1 0
run_check ./gemm_batch.x 4.045218e+14                23 23 23 100 1 0
run_check ./gemm_batch.x 4.082186e+15                32 32 32 100 1 0
# gemm_batch: sorted duplicates (dup=1)
run_check ./gemm_batch.x 1.233784e+11                8 8 8 100 1 1
# gemm_batch: shuffled duplicates (dup=2)
run_check ./gemm_batch.x 1.272673e+11                8 8 8 100 1 2
# gemm_batch: ld-padding (pad=1)
run_check ./gemm_batch.x 2.218164e+11                8 8 8 100 1 0 1.0 1
# gemm_batch: ld-padding + beta=0 (pad=1)
run_check ./gemm_batch.x 1.109079e+11                8 8 8 100 1 0 0.0 1

# gemm_index: same layout as strided => identical checksums
run_check ./gemm_index.x 7.354178e+19
run_check ./gemm_index.x 1.201635e+05                1 1 1 100 1
run_check ./gemm_index.x 1.964242e+11                4 4 4 1000 1
run_check ./gemm_index.x 2.491947e+11                8 8 8 100 1
# gemm_index: non-square
run_check ./gemm_index.x 3.099313e+12                7 13 5 500 1
# gemm_index: medium/large
run_check ./gemm_index.x 4.045218e+14                23 23 23 100 1
run_check ./gemm_index.x 4.082186e+15                32 32 32 100 1
# gemm_index: ld-padding (pad=1)
run_check ./gemm_index.x 2.218164e+11                8 8 8 100 1 1.0 1
# gemm_index: ld-padding + beta=0 (pad=2, non-square)
run_check ./gemm_index.x 1.208802e+12                7 13 5 500 1 0.0 2

# gemm_groups: per-group dispatch + batch loop
run_check ./gemm_groups.x 6.750768e+06               1 100 1 8
run_check ./gemm_groups.x 1.187253e+08               2 50 1
run_check ./gemm_groups.x 1.120340e+08               3 100 1 4
run_check ./gemm_groups.x 4.189222e+10               4 50 1 16
# gemm_groups: ld-padding (pad=1)
run_check ./gemm_groups.x 1.011109e+08               2 50 1 8 1.0 1
# gemm_groups: ld-padding + beta=0 (pad=1)
run_check ./gemm_groups.x 5.054755e+07               2 50 1 8 0.0 1
