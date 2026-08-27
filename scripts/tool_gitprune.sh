#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################

GIT=$(command -v git)

if [ "${GIT}" ]; then
  ${GIT} gc
  ${GIT} fsck --full
  ${GIT} reflog expire --expire=now --all
  # ${GIT} gc --prune=now
  ${GIT} gc --aggressive
  ${GIT} remote update --prune
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
