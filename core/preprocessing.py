from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from typing import Iterable

import torch
import torch.nn.functional as F
import torchaudio


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LIP_READER_MEAN = (0.421,)
LIP_READER_STD = (0.165,)


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _sample_frame_indices(num_frames: int, max_frames: int | None) -> torch.Tensor:
    if max_frames is None or num_frames <= max_frames:
        return torch.arange(num_frames)
    return torch.linspace(0, num_frames - 1, steps=max_frames).round().long()


def load_video_clip(
    video_path: str | Path,
    *,
    image_size: int = 224,
    max_frames: int | None = 96,
    mean: Iterable[float] = IMAGENET_MEAN,
    std: Iterable[float] = IMAGENET_STD,
    grayscale: bool = False,
) -> torch.Tensor:
    """Load a complete clip and return normalized frames shaped ``(T, C, H, W)``.

    ``max_frames`` samples uniformly across the full clip to control memory use.
    This is clip-level sampling, not word-level segmentation or cropping.
    """

    try:
        from torchcodec.decoders import VideoDecoder
    except Exception as exc:  # noqa: BLE001 - surface missing backend details.
        raise ImportError(
            "torchcodec is required for video decoding. Install torchcodec and ensure "
            "FFmpeg is available on your system."
        ) from exc

    path = _as_path(video_path)
    decoder = VideoDecoder(str(path), device="cpu")
    try:
        num_frames = len(decoder)
    except TypeError:
        num_frames = int(decoder.metadata.num_frames)
    if num_frames == 0:
        raise ValueError(f"No video frames decoded from {path}")

    frame_indices = _sample_frame_indices(num_frames, max_frames)
    batch = decoder.get_frames_at(indices=frame_indices.tolist())
    frames = batch.data
    if frames.numel() == 0:
        raise ValueError(f"No video frames decoded from {path}")

    frames = frames.float().div(255.0)
    if grayscale:
        red, green, blue = frames[:, 0:1], frames[:, 1:2], frames[:, 2:3]
        frames = 0.2989 * red + 0.5870 * green + 0.1140 * blue

    frames = F.interpolate(
        frames,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    if grayscale:
        if tuple(mean) == IMAGENET_MEAN and tuple(std) == IMAGENET_STD:
            mean = LIP_READER_MEAN
            std = LIP_READER_STD
        mean_values = tuple(mean)
        std_values = tuple(std)
        mean_tensor = torch.tensor(mean_values[:1], dtype=frames.dtype).view(1, 1, 1, 1)
        std_tensor = torch.tensor(std_values[:1], dtype=frames.dtype).view(1, 1, 1, 1)
    else:
        mean_tensor = torch.tensor(tuple(mean), dtype=frames.dtype).view(1, 3, 1, 1)
        std_tensor = torch.tensor(tuple(std), dtype=frames.dtype).view(1, 3, 1, 1)
    return (frames - mean_tensor) / std_tensor


def load_audio_waveform(
    audio_path: str | Path,
    *,
    target_sample_rate: int,
) -> torch.Tensor:
    """Load audio as a mono waveform at the requested sample rate."""
    path = _as_path(audio_path)
    try:
        waveform, sample_rate = torchaudio.load(str(path))
    except Exception as exc:  # noqa: BLE001 - provide a fallback for video containers.
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / f"{path.stem}.wav"
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(path),
                        "-ac",
                        "1",
                        "-ar",
                        str(target_sample_rate),
                        str(wav_path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError as ffmpeg_exc:
                raise RuntimeError(
                    "ffmpeg is required to extract audio when torchaudio cannot read the file. "
                    "Install ffmpeg or provide a separate audio file."
                ) from ffmpeg_exc
            waveform, sample_rate = torchaudio.load(str(wav_path))
    if waveform.ndim != 2:
        raise ValueError(f"Expected audio shaped (channels, samples), got {waveform.shape}")

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )

    return waveform.squeeze(0)


def load_audio_features(audio_path: str | Path, processor) -> torch.Tensor:
    """Convert audio to Whisper ``input_features`` shaped ``(80, 3000)``."""

    sample_rate = processor.feature_extractor.sampling_rate
    waveform = load_audio_waveform(audio_path, target_sample_rate=sample_rate)
    features = processor.feature_extractor(
        waveform.numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    return features.input_features.squeeze(0)


def resolve_inference_audio_path(
    video_path: str | Path,
    audio_path: str | Path | None = None,
) -> Path | None:
    """Use an explicit audio path, a same-stem WAV, or let callers try the video."""

    if audio_path is not None:
        return _as_path(audio_path)

    video = _as_path(video_path)
    same_stem_wav = video.with_suffix(".wav")
    if same_stem_wav.exists():
        return same_stem_wav

    return None


def load_inference_audio_features(
    video_path: str | Path,
    processor,
    *,
    audio_path: str | Path | None = None,
) -> torch.Tensor:
    """Load inference audio without requiring a CSV file.

    The preferred source is ``--audio`` or ``sample.wav`` next to ``sample.mp4``.
    If neither exists, this falls back to trying the video file itself in case it
    contains an audio track supported by the local torchaudio backend.
    """

    resolved_audio = resolve_inference_audio_path(video_path, audio_path)
    candidate = resolved_audio if resolved_audio is not None else _as_path(video_path)
    try:
        return load_audio_features(candidate, processor)
    except Exception as exc:  # noqa: BLE001 - keep the user-facing error actionable.
        if resolved_audio is None:
            raise RuntimeError(
                "Could not load audio for inference. Provide --audio, place a same-stem "
                ".wav next to the video, or use an mp4 readable by torchaudio."
            ) from exc
        raise
