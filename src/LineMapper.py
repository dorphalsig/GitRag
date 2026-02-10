import bisect
from typing import Optional


class LineMapper:
    """
    Efficiently maps byte offsets to (row, col) tuples using precomputed newline positions.
    Optimized for O(1) / O(log N) lookups after an O(N) initialization.
    """
    def __init__(self, contents: bytes):
        self.contents_len = len(contents)
        # Efficiently find all newline indices using the C-optimized find method
        self.newlines = []
        pos = contents.find(b"\n")
        while pos != -1:
            self.newlines.append(pos)
            pos = contents.find(b"\n", pos + 1)

    def find_nearest_newline(self, target: int, lo: int, hi: int) -> Optional[int]:
        """
        Return the byte index AFTER the newline closest to target in [lo, hi],
        or None if no newline.
        """
        if not self.newlines:
            return None

        # We want to find a newline index i such that lo <= i <= hi - 1.
        # Among those, we want the one that minimizes abs(i - target),
        # with a tie-break preferring the smaller index (the original behavior).
        start_idx = bisect.bisect_left(self.newlines, lo)
        end_idx = bisect.bisect_right(self.newlines, hi - 1)

        if start_idx >= end_idx:
            return None

        # Binary search for target within the valid range of newlines.
        # mid_idx will be the index in self.newlines.
        mid_idx = bisect.bisect_left(self.newlines, target, start_idx, end_idx)

        candidates = []
        if mid_idx > start_idx:
            candidates.append(self.newlines[mid_idx - 1])
        if mid_idx < end_idx:
            candidates.append(self.newlines[mid_idx])

        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0] + 1

        c1, c2 = candidates
        d1 = target - c1
        d2 = c2 - target

        if d1 <= d2:
            return c1 + 1
        return c2 + 1

    def get_newline_positions(self, lo: int, hi: int) -> list[int]:
        """Return all newline byte indices in [lo, hi)."""
        start_idx = bisect.bisect_left(self.newlines, lo)
        end_idx = bisect.bisect_left(self.newlines, hi)
        return self.newlines[start_idx:end_idx]

    def byte_to_point(self, offset: int) -> tuple[int, int]:
        """
        Convert a byte offset to a (row, column) tuple.
        Rows and Columns are 0-indexed.
        """
        # 1. Bounds Guard
        if offset < 0 or offset > self.contents_len:
            raise ValueError(f"Offset {offset} out of bounds (0-{self.contents_len})")

        # 2. Empty File / Start of File Fast-Path
        if offset == 0:
            return (0, 0)

        # 3. Binary Search for the newline preceding the offset
        # bisect_left returns the insertion point to maintain order.
        # If offset is 50 and newlines are [10, 20, 40, 60], idx will be 3 (index of 60).
        idx = bisect.bisect_left(self.newlines, offset)

        if idx == 0:
            # Case: Offset is before the very first newline (Row 0)
            # OR File has no newlines at all.
            # Col is simply the byte offset from start.
            return (0, offset)

        # Case: Offset is after at least one newline.
        # The 'row' is simply the index of the newline that started this line.
        # Example: newlines=[10], offset=15. idx=1. row=1.
        prev_newline_pos = self.newlines[idx - 1]
        
        # Col is distance from the previous newline. 
        # We subtract 1 because the newline character itself is not part of the next line's content.
        # Example: "A\nB" -> A=0, \n=1, B=2. 
        # Mapping B(2): prev_newline=1. Col = 2 - 1 - 1 = 0. Correct.
        col = offset - prev_newline_pos - 1
        
        return (idx, col)
