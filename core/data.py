from __future__ import annotations

import csv
import json
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from .preprocessing import load_audio_features, load_video_clip


VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm")
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a")


@dataclass(frozen=True)
class AVSRSample:
    sample_id: str
    video_path: Path
    audio_path: Path
    csv_path: Path


def _find_same_stem_file(directory: Path, stem: str, extensions: Sequence[str]) -> Path | None:
    for extension in extensions:
        candidate = directory / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def discover_samples(
    root: str | Path,
    *,
    strict: bool = False,
    allow_video_audio: bool = True,
    video_extensions: Sequence[str] = VIDEO_EXTENSIONS,
    audio_extensions: Sequence[str] = AUDIO_EXTENSIONS,
    scan_workers: int | None = None,
    cache_path: str | Path | None = None,
    log_every: int = 0,
) -> list[AVSRSample]:
    """Find ``*.mp4``/``*.wav``/``*.csv`` triplets rooted at ``root``.

    If ``allow_video_audio`` is True, missing audio files fall back to using
    the video path so callers can try to decode audio tracks from the container.
    """

    root_path = Path(root)
    cache_file = Path(cache_path) if cache_path is not None else None
    if cache_file is not None and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if (
                cached.get("version") == 1
                and cached.get("root") == str(root_path)
                and cached.get("allow_video_audio") == allow_video_audio
                and cached.get("video_extensions") == list(video_extensions)
                and cached.get("audio_extensions") == list(audio_extensions)
            ):
                if log_every > 0:
                    print(f"[scan] Using cached sample list from {cache_file}", flush=True)
                return [
                    AVSRSample(
                        sample_id=item["sample_id"],
                        video_path=Path(item["video_path"]),
                        audio_path=Path(item["audio_path"]),
                        csv_path=Path(item["csv_path"]),
                    )
                    for item in cached.get("samples", [])
                ]
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    if log_every > 0:
        print(f"[scan] Starting scan under {root_path}", flush=True)

    samples: list[AVSRSample] = []
    missing: list[str] = []

    video_exts = {ext.lower() for ext in video_extensions}
    audio_exts = {ext.lower() for ext in audio_extensions}

    def _classify_path(path: Path) -> tuple[str | None, str, Path]:
        suffix = path.suffix.lower()
        if suffix in video_exts:
            return ("video", path.stem, path)
        if suffix in audio_exts:
            return ("audio", path.stem, path)
        if suffix == ".csv":
            return ("csv", path.stem, path)
        return (None, path.stem, path)

    def _iter_files(root: Path) -> Sequence[Path]:
        scanned = 0
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                scanned += 1
                if log_every > 0 and scanned % log_every == 0:
                    print(f"[scan] Listed {scanned} files", flush=True)
                yield Path(dirpath) / filename

    videos: dict[str, Path] = {}
    audios: dict[str, Path] = {}
    csvs: dict[str, Path] = {}

    worker_count = 1 if scan_workers is None or scan_workers < 1 else scan_workers
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for index, (kind, stem, path) in enumerate(
            executor.map(_classify_path, _iter_files(root_path)),
            start=1,
        ):
            if kind == "video":
                videos[stem] = path
            elif kind == "audio":
                audios[stem] = path
            elif kind == "csv":
                csvs[stem] = path
            if log_every > 0 and index % log_every == 0:
                print(f"[scan] Classified {index} files", flush=True)

    for stem, csv_path in csvs.items():
        video_path = videos.get(stem)
        audio_path = audios.get(stem)
        if video_path is None:
            missing.append(str(csv_path))
            continue
        if audio_path is None:
            if allow_video_audio:
                audio_path = video_path
            else:
                missing.append(str(csv_path))
                continue
        samples.append(
            AVSRSample(
                sample_id=stem,
                video_path=video_path,
                audio_path=audio_path,
                csv_path=csv_path,
            )
        )

    if strict and missing:
        preview = "\n".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f"\n...and {len(missing) - 10} more"
        raise FileNotFoundError(f"CSV files without matching media:\n{preview}{suffix}")

    if cache_file is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "root": str(root_path),
                "allow_video_audio": allow_video_audio,
                "video_extensions": list(video_extensions),
                "audio_extensions": list(audio_extensions),
                "samples": [
                    {
                        "sample_id": sample.sample_id,
                        "video_path": str(sample.video_path),
                        "audio_path": str(sample.audio_path),
                        "csv_path": str(sample.csv_path),
                    }
                    for sample in samples
                ],
            }
            cache_file.write_text(json.dumps(payload))
            if log_every > 0:
                print(f"[scan] Wrote cache to {cache_file}", flush=True)
        except OSError:
            pass

    return samples


def load_word_timestamps(csv_path: str | Path) -> list[dict[str, Any]]:
    """Load timestamp rows and preserve them for future experiments."""

    def _resolve_columns(fieldnames: list[str] | None) -> tuple[str, str, str]:
        if not fieldnames:
            raise ValueError(f"{csv_path} is missing required column(s): word, start_time, end_time")

        normalized = {name.strip().lower(): name for name in fieldnames}
        word_key = normalized.get("word")
        start_key = normalized.get("start_time") or normalized.get("start")
        end_key = normalized.get("end_time") or normalized.get("end")
        if not word_key or not start_key or not end_key:
            missing = [
                key
                for key in ("word", "start_time/start", "end_time/end")
                if (key == "word" and not word_key)
                or (key.startswith("start") and not start_key)
                or (key.startswith("end") and not end_key)
            ]
            raise ValueError(f"{csv_path} is missing required column(s): {', '.join(missing)}")
        return word_key, start_key, end_key

    rows: list[dict[str, Any]] = []
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        word_key, start_key, end_key = _resolve_columns(reader.fieldnames)

        for row in reader:
            word = (row[word_key] or "").strip()
            if not word:
                continue
            rows.append(
                {
                    "word": word,
                    "start_time": float(row[start_key]),
                    "end_time": float(row[end_key]),
                }
            )

    return rows


def transcript_from_timestamps(timestamps: Sequence[dict[str, Any]]) -> str:
    return " ".join(str(item["word"]) for item in timestamps)


def split_samples(
    samples: Sequence[AVSRSample],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[AVSRSample], list[AVSRSample]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0.0, 1.0)")

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    val_count = int(round(len(shuffled) * val_ratio))
    if val_ratio > 0.0 and val_count == 0 and len(shuffled) > 1:
        val_count = 1
    return shuffled[val_count:], shuffled[:val_count]


class AVSRDataset(Dataset):
    """Clip-level AVSR dataset for V1 training.

    Timestamps are reconstructed from the CSV and returned as metadata, but they
    are not used by the model in V1.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        samples: Sequence[AVSRSample] | None = None,
        processor,
        image_size: int = 224,
        max_video_frames: int | None = 96,
        video_grayscale: bool = False,
        strict: bool = False,
        allow_video_audio: bool = True,
        visual_only: bool = False,
        scan_workers: int | None = None,
        sample_cache_path: str | Path | None = None,
        scan_log_every: int = 0,
        feature_cache_dir: str | Path | None = None,
        cache_video: bool = True,
        cache_audio: bool = True,
        skip_errors: bool = False,
        max_skip_errors: int = 10,
        max_label_length: int = 448,
    ) -> None:
        if processor is None:
            raise ValueError("AVSRDataset requires a WhisperProcessor")
        if samples is None and root is None:
            raise ValueError("Provide either root or samples")

        self.samples = list(samples) if samples is not None else discover_samples(
            root,
            strict=strict,
            allow_video_audio=allow_video_audio,
            scan_workers=scan_workers,
            cache_path=sample_cache_path,
            log_every=scan_log_every,
        )
        if not self.samples:
            raise ValueError("No AVSR samples found")

        self.processor = processor
        self.image_size = image_size
        self.max_video_frames = max_video_frames
        self.video_grayscale = video_grayscale
        self.visual_only = visual_only
        self.cache_video = cache_video
        self.cache_audio = cache_audio
        self.cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._audio_sample_rate = processor.feature_extractor.sampling_rate
        self.skip_errors = skip_errors
        self.max_skip_errors = max_skip_errors
        self.max_label_length = max_label_length

    def _cache_key(self, sample: AVSRSample) -> str:
        payload = "|".join(
            [
                sample.sample_id,
                str(sample.video_path),
                str(sample.audio_path),
                str(self.image_size),
                str(self.max_video_frames),
                str(self.video_grayscale),
                str(self.visual_only),
                str(self._audio_sample_rate),
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _load_cached_tensor(self, cache_path: Path) -> torch.Tensor | None:
        try:
            return torch.load(cache_path, map_location="cpu")
        except (OSError, RuntimeError, ValueError):
            return None

    def _save_cached_tensor(self, cache_path: Path, tensor: torch.Tensor) -> None:
        tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.{os.getpid()}.tmp")
        try:
            torch.save(tensor, tmp_path)
            os.replace(tmp_path, cache_path)
        except OSError:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        attempts = 0
        while True:
            sample = self.samples[index]
            try:
                timestamps = load_word_timestamps(sample.csv_path)
                transcript = transcript_from_timestamps(timestamps)

                cache_key = self._cache_key(sample) if self.cache_dir is not None else None

                video = None
                if self.cache_dir is not None and self.cache_video:
                    video_cache = self.cache_dir / f"{cache_key}.video.pt"
                    video = self._load_cached_tensor(video_cache)

                if video is None:
                    video = load_video_clip(
                        sample.video_path,
                        image_size=self.image_size,
                        max_frames=self.max_video_frames,
                        grayscale=self.video_grayscale,
                    )
                    if self.cache_dir is not None and self.cache_video:
                        self._save_cached_tensor(video_cache, video)

                audio = None
                if not self.visual_only:
                    if self.cache_dir is not None and self.cache_audio:
                        audio_cache = self.cache_dir / f"{cache_key}.audio.pt"
                        audio = self._load_cached_tensor(audio_cache)
                    if audio is None:
                        audio = load_audio_features(sample.audio_path, self.processor)
                        if self.cache_dir is not None and self.cache_audio:
                            self._save_cached_tensor(audio_cache, audio)
                labels = self.processor.tokenizer(
                    transcript,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_label_length,
                ).input_ids.squeeze(0)

                return {
                    "sample_id": sample.sample_id,
                    "video": video,
                    "audio": audio,
                    "labels": labels,
                    "transcript": transcript,
                    "timestamps": timestamps,
                }
            except Exception as exc:  # noqa: BLE001 - optional skip path.
                if not self.skip_errors:
                    raise
                attempts += 1
                print(f"[warn] Skipping sample {sample.sample_id}: {exc}")
                if attempts > self.max_skip_errors:
                    raise RuntimeError("Too many consecutive data errors") from exc
                index = (index + 1) % len(self.samples)


class AVSRCollator:
    """Pad video frames and decoder labels for a batch."""

    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        videos = [item["video"] for item in features]
        max_frames = max(video.shape[0] for video in videos)
        _, channels, height, width = videos[0].shape

        video_batch = videos[0].new_zeros(len(videos), max_frames, channels, height, width)
        video_attention_mask = torch.zeros(len(videos), max_frames, dtype=torch.long)
        for index, video in enumerate(videos):
            num_frames = video.shape[0]
            video_batch[index, :num_frames] = video
            video_attention_mask[index, :num_frames] = 1

        label_inputs = {"input_ids": [item["labels"].tolist() for item in features]}
        labels = self.processor.tokenizer.pad(
            label_inputs,
            padding=True,
            return_tensors="pt",
        ).input_ids
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)

        audio_items = [item["audio"] for item in features]
        audio_features = None
        if all(audio is not None for audio in audio_items):
            audio_features = torch.stack(audio_items)  # type: ignore[arg-type]

        return {
            "sample_id": [item["sample_id"] for item in features],
            "video": video_batch,
            "video_attention_mask": video_attention_mask,
            "audio_features": audio_features,
            "labels": labels,
            "transcript": [item["transcript"] for item in features],
            "timestamps": [item["timestamps"] for item in features],
        }
