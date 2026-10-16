-------------------------------------------------------------------------------
-- Copyright (c) 2009-2026 Hans Pabst                                        --
-- Copyright (c) 2009-2026 Intel Corporation                                 --
--                                                                           --
-- For information on the license, see the LICENSE file.                     --
-- SPDX-License-Identifier: BSD-3-Clause                                     --
-------------------------------------------------------------------------------
-- Pandoc's GFM reader leaves column widths unset, and LaTeX then sets each
-- column on one line that runs off the page. A table that does not fit gets
-- wrapping columns: each keeps its longest word, the rest follows content.
local LINE = 100

local function measure(rows, ncols, len, word)
  for _, row in ipairs(rows) do
    for i, cell in ipairs(row.cells) do
      if i <= ncols then
        local text = pandoc.utils.stringify(cell.contents)
        len[i] = math.max(len[i], #text)
        for w in text:gmatch("%S+") do
          word[i] = math.max(word[i], #w)
        end
      end
    end
  end
end

function Table(tbl)
  local result = nil
  local ncols = #tbl.colspecs
  local len, word = {}, {}
  local unset = true
  for i = 1, ncols do
    unset = unset and nil == tbl.colspecs[i][2]
    len[i], word[i] = 0, 0
  end
  if unset then
    measure(tbl.head.rows, ncols, len, word)
    for _, body in ipairs(tbl.bodies) do
      measure(body.head, ncols, len, word)
      measure(body.body, ncols, len, word)
    end
    measure(tbl.foot.rows, ncols, len, word)
    local natural, floor, rest = 0, 0, 0
    for i = 1, ncols do
      natural = natural + len[i] + 2
      floor = floor + word[i]
      rest = rest + len[i] - word[i]
    end
    if LINE < natural then
      local spare = math.max(LINE - 2 * ncols - floor, 0)
      local width, total = {}, 0
      for i = 1, ncols do
        width[i] = word[i]
        if 0 < rest then
          width[i] = width[i] + spare * (len[i] - word[i]) / rest
        end
        total = total + width[i]
      end
      for i = 1, ncols do
        tbl.colspecs[i] = {tbl.colspecs[i][1], width[i] / total}
      end
      result = tbl
    end
  end
  return result
end
