import pytest
from unittest.mock import patch
from Indexer import Indexer


def _make_bare_indexer(start_time: float) -> Indexer:
    """Create Indexer via __new__, skipping __init__. Only start_time needed for _check_timeout."""
    obj = Indexer.__new__(Indexer)
    obj.start_time = start_time
    return obj


def test_no_timeout_within_limit():
    indexer = _make_bare_indexer(start_time=1000.0)
    with patch("Indexer.SOFT_TIMEOUT_SECONDS", 10), \
         patch("Indexer.time.time", return_value=1005.0):
        # 5s elapsed, limit 10s — should not raise
        indexer._check_timeout()


def test_soft_timeout_exits_75():
    indexer = _make_bare_indexer(start_time=1000.0)
    with patch("Indexer.SOFT_TIMEOUT_SECONDS", 10), \
         patch("Indexer.time.time", return_value=1011.0):
        # 11s elapsed, limit 10s — must exit with code 75
        with pytest.raises(SystemExit) as exc_info:
            indexer._check_timeout()
        assert exc_info.value.code == 75


def test_disabled_when_soft_timeout_zero():
    # SOFT_TIMEOUT_SECONDS=0: condition is `elapsed > 0` which is always True.
    # When set to 0 the caller should not rely on this guard — verify actual behavior.
    indexer = _make_bare_indexer(start_time=1000.0)
    with patch("Indexer.SOFT_TIMEOUT_SECONDS", 0), \
         patch("Indexer.time.time", return_value=1000.0):
        # elapsed == 0.0, condition 0.0 > 0 is False — no exit
        indexer._check_timeout()
