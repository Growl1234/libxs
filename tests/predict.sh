#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Absent inputs, end to end through the sample rather than through a stub.
#
# A gap is a value the corpus does not have, not a value that happens to be
# zero, and the two are only distinguishable in what the model refuses to do
# with them. A mode that reads coordinates independently answers from the ones
# it has; a rotation mixes every coordinate into every output and a tree sorts
# on one at a time, so neither can express a gap and both refuse to build.
# Refusing is the property under test: the failure it prevents is a model that
# answers confidently from a coordinate it never had.
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd -P)
SAMPLES="${HERE}/../samples/predict"

cd "${SAMPLES}" 2>/dev/null || exit 1
if [ ! -e ./predict_crystal.x ] || [ ! -e ./predict_crystal.csv ]; then exit 0; fi

# a small training split keeps the run short; the contract does not depend on it
RUN="./predict_crystal.x ./predict_crystal.csv 0.2 2 0"

for MODE in none fisher setdiff; do
  OUT=$(${RUN} "${MODE}" gaps0.15 2>&1) || true
  if ! echo "${OUT}" | grep -q "^Accuracy:"; then
    echo "${MODE} did not answer over a corpus with gaps"
    echo "${OUT}" | tail -3
    exit 1
  fi
done

for MODE in pca rf hknn; do
  # a refusal is reported through the exit code as well as the text
  OUT=$(${RUN} "${MODE}" gaps0.15 2>&1) || true
  if ! echo "${OUT}" | grep -q "^Refused:"; then
    echo "${MODE} accepted a corpus with gaps instead of refusing it"
    echo "${OUT}" | tail -3
    exit 1
  fi
done

# the sentinel resolves to a real mode and the model reports which one, so a
# caller can tell what was chosen; a mode named on the command line is reported
# as requested rather than selected
OUT=$(${RUN} 2>&1) || true
if ! echo "${OUT}" | grep -q "^Decomposition: .* (selected at build)$"; then
  echo "the default did not select a decomposition"
  echo "${OUT}" | tail -3
  exit 1
fi
if ! ${RUN} none 2>&1 | grep -q "^Decomposition: RAW (requested)$"; then
  echo "a requested decomposition was not reported as requested"
  exit 1
fi

# selecting is worth what it costs: the choice beats the mode this corpus
# punishes, which is the whole claim behind making it the default
AUTO=$(${RUN} 2>&1 | sed -n 's|^Accuracy: [0-9]*/[0-9]* = \([0-9.]*\)%$|\1|p')
PCA=$(${RUN} pca 2>&1 | sed -n 's|^Accuracy: [0-9]*/[0-9]* = \([0-9.]*\)%$|\1|p')
if [ -z "${AUTO}" ] || [ -z "${PCA}" ]; then
  echo "could not read an accuracy back from the sample"
  exit 1
fi
if [ "$(echo "${AUTO} ${PCA}" | awk '{print ($1 >= $2)}')" != "1" ]; then
  echo "the selected mode (${AUTO}%) lost to a fixed one (${PCA}%)"
  exit 1
fi

# every mode answers the same corpus without gaps, gaps being the only variable
for MODE in none fisher setdiff pca rf hknn; do
  OUT=$(${RUN} "${MODE}" 2>&1) || true
  if ! echo "${OUT}" | grep -q "^Accuracy:"; then
    echo "${MODE} failed on a corpus with no gaps at all"
    echo "${OUT}" | tail -3
    exit 1
  fi
done

echo "OK"
