"""四叉树 tile 结构。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Tile:
    level: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int

    @property
    def height(self) -> int:
        return self.row_end - self.row_start

    @property
    def width(self) -> int:
        return self.col_end - self.col_start

    @property
    def area(self) -> int:
        return self.height * self.width

    def split(self) -> List["Tile"]:
        mid_row = self.row_start + self.height // 2
        mid_col = self.col_start + self.width // 2
        if mid_row <= self.row_start or mid_row >= self.row_end or mid_col <= self.col_start or mid_col >= self.col_end:
            return []
        level = self.level + 1
        return [
            Tile(level, self.row_start, mid_row, self.col_start, mid_col),
            Tile(level, self.row_start, mid_row, mid_col, self.col_end),
            Tile(level, mid_row, self.row_end, self.col_start, mid_col),
            Tile(level, mid_row, self.row_end, mid_col, self.col_end),
        ]
