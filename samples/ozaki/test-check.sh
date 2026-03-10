#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd -P)

GREP=$(command -v grep)
CAT=$(command -v cat)

if [ ! "${GREP}" ] || [ ! "${CAT}" ]; then
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi

if [ "$1" ]; then
  TESTS=$1
else
  # Discover tests from built *-wrap.x executables
  TESTS="$(ls -1 "${HERE}"/*gemm-wrap.x 2>/dev/null \
    | xargs -I{} basename {} -wrap.x | sort -u)"
  if [ ! "${TESTS}" ]; then TESTS=dgemm; fi
fi
if [ $# -gt 0 ]; then shift; fi

# BF16 Scheme 3 threshold (overridable via CHECK_BF16 env var).
# BF16 splitting is inherently approximate; the GPU's fast-relaxed-math
# and FP32 accumulation limit precision to ~1e-4 for double inputs.
CHECK_BF16=${CHECK_BF16:-1e-3}

TMPF=$(mktemp)
trap 'rm -f ${TMPF}' EXIT

for TEST in ${TESTS}; do

EXE="${HERE}/${TEST}-wrap.x"
if [ ! -e "${EXE}" ] || [ ! -e .state ] || \
   [ "$(${GREP} 'BLAS=0' .state 2>/dev/null)" ];
then
  echo "SKIPPED (${EXE} not available or BLAS=0)"
  continue
fi

RESULT=0

# Scheme 1 (mantissa slicing): exact with default nslices
echo "-----------------------------------"
echo "CHECK [${TEST}]: Scheme 1 (default)"
if [ "$*" ]; then echo "args    $*"; fi
{ CHECK=-1 OZAKI_VERBOSE=1 OZAKI=1 "${EXE}" "$@" 2>"${TMPF}"; } >/dev/null || RESULT=$?
if [ "0" != "${RESULT}" ]; then
  echo "FAILED[${RESULT}] $(${CAT} "${TMPF}")"
  exit ${RESULT}
fi
if ${GREP} -q "CHECK:" "${TMPF}"; then
  echo "OK $(${GREP} "CHECK:" "${TMPF}")"
else
  echo "FAILED (no CHECK output)"
  exit 1
fi
echo

# Scheme 2 (CRT modular arithmetic): exact with default nprimes
echo "-----------------------------------"
echo "CHECK [${TEST}]: Scheme 2 (CRT)"
if [ "$*" ]; then echo "args    $*"; fi
{ CHECK=-1 OZAKI_VERBOSE=1 OZAKI=2 "${EXE}" "$@" 2>"${TMPF}"; } >/dev/null || RESULT=$?
if [ "0" != "${RESULT}" ]; then
  echo "FAILED[${RESULT}] $(${CAT} "${TMPF}")"
  exit ${RESULT}
fi
if ${GREP} -q "CHECK:" "${TMPF}"; then
  echo "OK $(${GREP} "CHECK:" "${TMPF}")"
else
  echo "FAILED (no CHECK output)"
  exit 1
fi
echo

# Scheme 3 (BF16 Dekker split): inherently approximate (FP32 accumulation)
# Use explicit threshold — BF16 dot products are not exact unlike int8.
echo "-----------------------------------"
echo "CHECK [${TEST}]: Scheme 3 (BF16)"
if [ "$*" ]; then echo "args    $*"; fi
{ CHECK=${CHECK_BF16} OZAKI_VERBOSE=1 OZAKI=3 "${EXE}" "$@" 2>"${TMPF}"; } >/dev/null || RESULT=$?
if [ "0" != "${RESULT}" ]; then
  echo "FAILED[${RESULT}] $(${CAT} "${TMPF}")"
  exit ${RESULT}
fi
if ${GREP} -q "CHECK:" "${TMPF}"; then
  echo "OK $(${GREP} "CHECK:" "${TMPF}")"
else
  echo "FAILED (no CHECK output)"
  exit 1
fi
echo

# Scheme 4 (CRT + BF16 dot products): exact — residues are small integers
# representable in BF16, and FP32 accumulation is exact for BLOCK_K <= 64.
echo "-----------------------------------"
echo "CHECK [${TEST}]: Scheme 4 (CRT+BF16)"
if [ "$*" ]; then echo "args    $*"; fi
{ CHECK=-1 OZAKI_VERBOSE=1 OZAKI=4 "${EXE}" "$@" 2>"${TMPF}"; } >/dev/null || RESULT=$?
if [ "0" != "${RESULT}" ]; then
  echo "FAILED[${RESULT}] $(${CAT} "${TMPF}")"
  exit ${RESULT}
fi
if ${GREP} -q "CHECK:" "${TMPF}"; then
  echo "OK $(${GREP} "CHECK:" "${TMPF}")"
else
  echo "FAILED (no CHECK output)"
  exit 1
fi
echo

done # TESTS