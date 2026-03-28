import pytest
from Chunker.LineMapper import LineMapper

def test_line_mapper_empty_file():
    lm = LineMapper(b"")
    assert lm.newlines == []
    assert lm.byte_to_point(0) == (0, 0)
    assert lm.find_nearest_newline(0, 0, 0) is None

def test_line_mapper_single_line():
    content = b"hello world"
    lm = LineMapper(content)
    assert lm.newlines == []
    assert lm.byte_to_point(0) == (0, 0)
    assert lm.byte_to_point(5) == (0, 5)
    assert lm.byte_to_point(11) == (0, 11)
    assert lm.find_nearest_newline(5, 0, 11) is None

def test_line_mapper_multi_line_unix():
    content = b"line1\nline2\nline3"
    lm = LineMapper(content)
    assert lm.newlines == [5, 11]

    # byte_to_point
    assert lm.byte_to_point(0) == (0, 0)
    assert lm.byte_to_point(5) == (0, 5)
    assert lm.byte_to_point(6) == (1, 0)
    assert lm.byte_to_point(11) == (1, 5)
    assert lm.byte_to_point(12) == (2, 0)
    assert lm.byte_to_point(17) == (2, 5)

    # find_nearest_newline
    assert lm.find_nearest_newline(3, 0, 17) == 6 # closer to 5+1
    assert lm.find_nearest_newline(9, 0, 17) == 12 # closer to 11+1
    assert lm.find_nearest_newline(5, 0, 5) is None # hi bound is exclusive or inclusive?
    # find_nearest_newline uses hi_bound = hi - 1 and bisect_right(newlines, hi_bound)
    # if hi=5, hi_bound=4. bisect_right([5, 11], 4) is 0. end_idx=0. returns None. Correct.
    assert lm.find_nearest_newline(5, 0, 6) == 6 # target 5, lo 0, hi 6. hi_bound 5. end_idx 1. mid_idx 0. c2=5. returns 6.

def test_line_mapper_multi_line_windows():
    content = b"line1\r\nline2\r\n"
    lm = LineMapper(content)
    # line1\r\n -> indices 0,1,2,3,4,5,6 (6 is \n)
    # line2\r\n -> indices 7,8,9,10,11,12,13 (13 is \n)
    assert lm.newlines == [6, 13]
    assert lm.byte_to_point(0) == (0, 0)
    assert lm.byte_to_point(6) == (0, 6)
    assert lm.byte_to_point(7) == (1, 0)
    assert lm.byte_to_point(13) == (1, 6)
    assert lm.byte_to_point(14) == (2, 0)

def test_find_nearest_newline_no_c2():
    # Test path where c2 == -1
    content = b"abc\ndef"
    lm = LineMapper(content)
    # newlines at 3
    # target 5, lo 0, hi 4. hi_bound 3. bisect_right([3], 3) is 1. end_idx 1.
    # bisect_left([3], 5, 0, 1) is 1. mid_idx 1.
    # mid_idx > start_idx (1 > 0) -> c1 = 3.
    # mid_idx < end_idx (1 < 1) is False. c2 = -1.
    assert lm.find_nearest_newline(5, 0, 4) == 4

def test_line_mapper_bounds():
    lm = LineMapper(b"abc")
    with pytest.raises(ValueError, match="out of bounds"):
        lm.byte_to_point(-1)
    with pytest.raises(ValueError, match="out of bounds"):
        lm.byte_to_point(4)

def test_find_nearest_newline_complex():
    content = b"01234\n6789\nBCDE\nFGHI"
    # newlines at 5, 10, 15
    lm = LineMapper(content)

    # target in middle, c1 vs c2
    assert lm.find_nearest_newline(7, 0, 20) == 6  # target 7 is closer to 5 (dist 2) than 10 (dist 3)
    assert lm.find_nearest_newline(8, 0, 20) == 11 # target 8 is closer to 10 (dist 2) than 5 (dist 3)
    assert lm.find_nearest_newline(7, 8, 20) == 11 # lo=8 excludes 5. only 10, 15 available.

    # target matches newline
    assert lm.find_nearest_newline(5, 0, 20) == 6
    assert lm.find_nearest_newline(10, 0, 20) == 11

def test_get_newline_positions():
    content = b"a\nb\nc\nd"
    # newlines at 1, 3, 5
    lm = LineMapper(content)
    assert lm.get_newline_positions(0, 10) == [1, 3, 5]
    assert lm.get_newline_positions(2, 5) == [3]
    assert lm.get_newline_positions(1, 2) == [1]
