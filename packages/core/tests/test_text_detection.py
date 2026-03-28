from unittest.mock import MagicMock

from text_detection import BinaryDetector


def test_is_binary_returns_true_when_detector_raises(tmp_path):
    file_path = tmp_path / "sample.bin"
    file_path.write_bytes(b"\x00\x01\x02")

    detector = BinaryDetector.__new__(BinaryDetector)
    detector._base_dir = tmp_path
    detector._magika = MagicMock()
    detector._magika.identify_path.side_effect = RuntimeError("magika exploded")

    assert detector.is_binary(str(file_path)) is True
