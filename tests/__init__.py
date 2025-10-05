import struct
import sys
import types


def _install_numpy_stub() -> None:
    if "numpy" in sys.modules:
        return
    try:
        import numpy  # type: ignore # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("numpy")

        class _Array(list):
            def tolist(self):  # type: ignore[override]
                return list(self)

        def frombuffer(buffer: bytes, dtype=None):
            if dtype not in (None, float32):
                raise TypeError("fake numpy only supports float32")
            size = len(buffer) // 4
            if size <= 0:
                return _Array()
            fmt = "<" + "f" * size
            values = struct.unpack(fmt, buffer[: size * 4])
            return _Array(values)

        float32 = "float32"

        module.frombuffer = frombuffer  # type: ignore[attr-defined]
        module.float32 = float32  # type: ignore[attr-defined]
        sys.modules["numpy"] = module


_install_numpy_stub()


def _install_cloudflare_stub() -> None:
    if "cloudflare" in sys.modules:
        return
    try:
        import cloudflare  # type: ignore # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("cloudflare")

        class Cloudflare:  # type: ignore
            def __init__(self, *a, **k):
                pass

        module.Cloudflare = Cloudflare
        sys.modules["cloudflare"] = module


_install_cloudflare_stub()
