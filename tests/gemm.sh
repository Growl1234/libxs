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
  L1=$(${PROG} "$@" 2>&1 | sed -n 's/.*l1_tst=//p')
  if [ "${L1}" != "${GOLDEN}" ]; then
    >&2 echo "FAILED: ${PROG} $* => l1_tst=${L1} (expected ${GOLDEN})"
    exit 1
  fi
}

# gemm_strided: default shape
run_check ./gemm_strided.x 20801984173.500000
# gemm_strided: small shapes
run_check ./gemm_strided.x 13.500000             1 1 1 100 1
run_check ./gemm_strided.x 59532.000000           4 4 4 1000 1
run_check ./gemm_strided.x 6750768.000000         8 8 8 100 1
# gemm_strided: non-square
run_check ./gemm_strided.x 3325959.000000         7 13 5 500 1
# gemm_strided: medium/large
run_check ./gemm_strided.x 828371136.000000       16 16 16 200 1
run_check ./gemm_strided.x 10401097225.500000     23 23 23 100 1
run_check ./gemm_strided.x 104354120448.000000    32 32 32 100 1

# gemm_batch: same shapes (dup=0, no duplicates)
run_check ./gemm_batch.x 13.500000               1 1 1 100 1 0
run_check ./gemm_batch.x 6750768.000000           8 8 8 100 1 0
run_check ./gemm_batch.x 3325959.000000           7 13 5 500 1 0
run_check ./gemm_batch.x 10401097225.500000       23 23 23 100 1 0
run_check ./gemm_batch.x 104354120448.000000      32 32 32 100 1 0
# gemm_batch: sorted duplicates (dup=1)
run_check ./gemm_batch.x 13498416.000000          8 8 8 100 1 1
# gemm_batch: shuffled duplicates (dup=2)
run_check ./gemm_batch.x 43872811056.000000       8 8 8 100 1 2

# gemm_groups
run_check ./gemm_groups.x 6750768.000000          1 100 1 8
run_check ./gemm_groups.x 118725276.000000        2 50 1
run_check ./gemm_groups.x 118784808.000000        3 100 1 4
run_check ./gemm_groups.x 59813658216.000000      4 50 1 16
