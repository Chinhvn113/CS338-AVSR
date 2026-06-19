"""CS338 audio-visual speech recognition pipeline."""

__all__ = [
    "AVSRCollator",
    "AVSRDataset",
    "AVSRSample",
    "AVWhisperConfig",
    "AVWhisperModel",
    "discover_samples",
]


def __getattr__(name):
    if name in {"AVSRCollator", "AVSRDataset", "AVSRSample", "discover_samples"}:
        from . import data

        return getattr(data, name)
    if name in {"AVWhisperConfig", "AVWhisperModel"}:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
