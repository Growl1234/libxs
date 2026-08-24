#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd -P)

cd "${HERE}/../samples/ozaki"
./test-wrap.sh dgemm
./test-wrap.sh dgemm  16  20 350 1 0  1 0.0 350 350 1000
./test-wrap.sh dgemm  23  21  32 0 1 -1 0.5  32  32 1000
./test-wrap.sh dgemm 200 200 256 1 1  1 0.0 256 256 1000
