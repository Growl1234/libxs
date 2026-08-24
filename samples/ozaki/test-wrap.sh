#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# shellcheck disable=SC2011
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd -P)
DEPDIR=${HERE}/../../..

UNAME=$(command -v uname)
GREP=$(command -v grep)
CAT=$(command -v cat)
TR=$(command -v tr)

if [ "Darwin" != "$(${UNAME})" ]; then
  LIBEXT=so
else
  LIBEXT=dylib
fi
if [ "$1" ]; then
  TESTS=$1
else
  # Discover tests from built executables (*-wrap.x and *-blas.x)
  TESTS="$({ ls -1 "${HERE}"/*-wrap.x "${HERE}"/*-blas.x 2>/dev/null || true; } \
    | xargs -I{} basename {} .x | sed 's/-wrap$//;s/-blas$//' | sort -u)"
fi
if [ $# -gt 0 ]; then shift; fi

TMPF=$(mktemp)
trap 'rm ${TMPF}' EXIT

# What tells an intercepted run from a plain one.  Not "GEMM:": the driver labels
# its own timed block "OZAKI GEMM:" whether or not anything wrapped the call, so
# that pattern matches both and cannot decide anything.  The bracketed form is the
# wrapper's own verification line, which only the wrapper prints -- measured 0 for
# a plain *-blas.x against 2 for both the static wrap and the LD_PRELOAD path.
WRAPPED="GEMM\["

# set verbosity to check for generated kernels
export OZAKI_VERBOSE=${OZAKI_VERBOSE:-1}

for TEST in ${TESTS}; do
  NAME=$(echo "${TEST}" | ${TR} [[:lower:]] [[:upper:]])

  if [ -e "${HERE}/${TEST}-blas.x" ]; then
    echo "-----------------------------------"
    echo "${NAME} (ORIGINAL BLAS)"
    if [ "$*" ]; then echo "args    $*"; fi
    RESULT=0
    { time eval "${HERE}/${TEST}-blas.x $* >${TMPF} 2>&1"; } 2>&1 \
      | ${GREP} real || RESULT=$?
    if [ "0" != "${RESULT}" ]; then
      echo "FAILED[${RESULT}] $(${CAT} "${TMPF}")"
      exit ${RESULT}
    elif ! ${GREP} -q "${WRAPPED}" "${TMPF}"; then
      echo "OK"
    else
      echo "FAILED"
      exit 1
    fi
    echo
  fi

  if [ -e "${HERE}/${TEST}-wrap.x" ] && [ -e .state ] && \
     [ ! "$(${GREP} 'BLAS=0' .state)" ];
  then
    echo "-----------------------------------"
    echo "${NAME} (STATIC WRAP)"
    if [ "$*" ]; then echo "args    $*"; fi
    RESULT=0
    { time eval "${HERE}/${TEST}-wrap.x $* >${TMPF} 2>&1"; } 2>&1 \
      | ${GREP} real || RESULT=$?
    if [ "0" != "${RESULT}" ]; then
      echo "FAILED[${RESULT}] $(${CAT} "${TMPF}")"
      exit ${RESULT}
    elif ${GREP} -q "${WRAPPED}" "${TMPF}"; then
      echo "OK"
    else
      echo "FAILED"
      exit 1
    fi
    echo
  fi

  if [ -e "${HERE}/${TEST}-blas.x" ] && \
     [ -e "${HERE}/libwrap.${LIBEXT}" ];
  then
    echo "-----------------------------------"
    echo "${NAME} (LD_PRELOAD)"
    if [ "$*" ]; then echo "args    $*"; fi
    RESULT=0
    { time eval " \
      LD_LIBRARY_PATH=${DEPDIR}/lib:${LD_LIBRARY_PATH} LD_PRELOAD=${HERE}/libwrap.${LIBEXT} \
      DYLD_LIBRARY_PATH=${DEPDIR}/lib:${DYLD_LIBRARY_PATH} DYLD_INSERT_LIBRARIES=${DEPDIR}/lib/libxs.${LIBEXT} \
      ${HERE}/${TEST}-blas.x $* >${TMPF} 2>&1"; } 2>&1 | ${GREP} real || RESULT=$?
    if [ "0" != "${RESULT}" ]; then
      echo "FAILED[${RESULT}] $(${CAT} "${TMPF}")"
      exit ${RESULT}
    elif ${GREP} -q "${WRAPPED}" "${TMPF}"; then
      echo "OK"
    else
      echo "FAILED"
      exit 1
    fi
    echo
  fi
done
