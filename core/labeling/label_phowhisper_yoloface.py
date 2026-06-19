#!/usr/bin/env python3
"""Label video datasets with PhoWhisper and crop videos around tracked faces.

The output follows this repository's clip-level AVSR dataset format:

    sample_001.mp4
    sample_001.wav
    sample_001.csv

The CSV contains ``Word,Start,End`` columns. When PhoWhisper returns word
timestamps they are used directly; otherwise the transcript words are spread
uniformly across the clip duration so the loader can reconstruct the transcript.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from tqdm import tqdm


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
DEFAULT_YOLO_FACE_MODEL = "yolov12m-face.pt"


@dataclass(frozen=True)
class LabeledSample:
    sample_id: str
    source_video: Path
    video_path: Path
    audio_path: Path
    csv_path: Path
    transcript: str
    duration: float
    face_detected: bool
    face_coverage: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PhoWhisper transcription and YOLOv12-face face-following crop over a video dataset."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing input videos.")
    parser.add_argument("--out-root", type=Path, required=True, help="Output root for labeled dataset files.")
    parser.add_argument(
        "--phowhisper-model",
        default="vinai/PhoWhisper-small",
        help="Hugging Face model id or local path for PhoWhisper.",
    )
    parser.add_argument(
        "--yolo-face-model",
        default=DEFAULT_YOLO_FACE_MODEL,
        help="YOLO face checkpoint path or URL.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--language", default="vi", help="Whisper language code.")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--chunk-length-s", type=float, default=30.0, help="ASR chunk length for long videos.")
    parser.add_argument("--batch-size", type=int, default=2, help="ASR pipeline batch size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO IoU threshold.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config.")
    parser.add_argument(
        "--track-selection",
        choices=["most-detected", "largest", "lower-face-motion"],
        default="most-detected",
        help="How to choose one face track when multiple faces are detected.",
    )
    parser.add_argument("--crop-size", type=int, default=224, help="Output cropped video frame size.")
    parser.add_argument("--face-padding", type=float, default=1.8, help="Scale factor around detected face box.")
    parser.add_argument("--smooth-window", type=int, default=9, help="Odd moving-average window for crop boxes.")
    parser.add_argument(
        "--fallback",
        choices=["copy", "center", "skip"],
        default="center",
        help="What to do when no face is detected.",
    )
    parser.add_argument(
        "--layout",
        choices=["mirror", "flat"],
        default="mirror",
        help="Mirror the input directory tree or flatten outputs by video stem.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output triplets.")
    parser.add_argument("--limit", type=int, help="Only process the first N videos.")
    parser.add_argument("--manifest-name", default="manifest.jsonl", help="JSONL manifest filename under out-root.")
    parser.add_argument("--no-word-timestamps", action="store_true", help="Disable Whisper word timestamp requests.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files for debugging.")
    return parser.parse_args()


def configure_runtime_cache(workspace: Path) -> None:
    cache_root = workspace / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(workspace / ".cache" / "ultralytics"))


def iter_videos(root: Path) -> list[Path]:
    videos: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(path)
    return sorted(videos)


def output_dir_for(video_path: Path, data_root: Path, out_root: Path, layout: str) -> Path:
    if layout == "flat":
        return out_root
    relative_parent = video_path.relative_to(data_root).parent
    return out_root / relative_parent


def run_command(command: Sequence[str], *, quiet: bool = True) -> None:
    stdout = subprocess.DEVNULL if quiet else None
    stderr = subprocess.DEVNULL if quiet else None
    subprocess.run(command, check=True, stdout=stdout, stderr=stderr)


def ffprobe_duration(video_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return float(completed.stdout.strip())


def extract_audio(video_path: Path, audio_path: Path) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ]
    )


def load_asr_pipeline(model_name: str, device: str, batch_size: int):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )
    processor = AutoProcessor.from_pretrained(model_name)

    if device.startswith("cuda"):
        model.to(device)
        pipeline_device: int | str = int(device.split(":", 1)[1]) if ":" in device else 0
    else:
        pipeline_device = -1

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=pipeline_device,
        batch_size=batch_size,
    )


def normalize_timestamp(value: Any) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        start, end = value
        return (
            None if start is None else float(start),
            None if end is None else float(end),
        )
    return None, None


def transcribe_audio(
    asr,
    audio_path: Path,
    *,
    duration: float,
    language: str,
    task: str,
    chunk_length_s: float,
    word_timestamps: bool,
) -> tuple[str, list[dict[str, float | str]]]:
    kwargs: dict[str, Any] = {
        "chunk_length_s": chunk_length_s,
        "generate_kwargs": {"task": task, "language": language},
    }
    if word_timestamps:
        kwargs["return_timestamps"] = "word"

    try:
        result = asr(str(audio_path), **kwargs)
    except TypeError:
        kwargs.pop("return_timestamps", None)
        result = asr(str(audio_path), **kwargs)

    transcript = str(result.get("text", "")).strip()
    words: list[dict[str, float | str]] = []
    for chunk in result.get("chunks", []) or []:
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        start, end = normalize_timestamp(chunk.get("timestamp"))
        if start is None:
            continue
        if end is None:
            end = duration
        words.append({"word": text, "start": max(0.0, start), "end": min(duration, end)})

    if words:
        return transcript, words
    return transcript, heuristic_word_timestamps(transcript, duration)


def heuristic_word_timestamps(transcript: str, duration: float) -> list[dict[str, float | str]]:
    tokens = transcript.split()
    if not tokens:
        return []
    step = max(duration, 0.0) / len(tokens)
    rows: list[dict[str, float | str]] = []
    for index, token in enumerate(tokens):
        start = index * step
        end = duration if index == len(tokens) - 1 else (index + 1) * step
        rows.append({"word": token, "start": start, "end": end})
    return rows


def write_timestamp_csv(csv_path: Path, rows: Iterable[dict[str, float | str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Word", "Start", "End"])
        for row in rows:
            writer.writerow([row["word"], f"{float(row['start']):.6f}", f"{float(row['end']):.6f}"])


def read_transcript_from_csv(csv_path: Path) -> str:
    words = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)  # skip header
            for row in reader:
                if len(row) > 0:
                    words.append(row[0])
    except Exception:
        pass
    return " ".join(words)


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def lower_face_crop(frame: np.ndarray, box: Sequence[float]) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    face_height = max(1.0, y2 - y1)
    left = max(0, int(round(x1)))
    right = min(width, int(round(x2)))
    top = max(0, int(round(y1 + 0.45 * face_height)))
    bottom = min(height, int(round(y2)))
    if right <= left or bottom <= top:
        return None

    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 32), interpolation=cv2.INTER_AREA)


def track_ids_for_boxes(result: Any, fallback_start: int) -> tuple[list[int], int]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], fallback_start
    if boxes.id is not None:
        return [int(item) for item in boxes.id.cpu().numpy().tolist()], fallback_start

    fallback_ids = list(range(fallback_start, fallback_start - len(boxes), -1))
    return fallback_ids, fallback_start - len(boxes)


def choose_track(results: list[Any], selection: str) -> int | None:
    area_by_track: dict[int, list[float]] = {}
    lower_motion_by_track: dict[int, list[float]] = {}
    previous_lower_crop: dict[int, np.ndarray] = {}
    fallback_index = -1
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        candidate_ids, fallback_index = track_ids_for_boxes(result, fallback_index)
        for track_id, box in zip(candidate_ids, xyxy):
            area_by_track.setdefault(track_id, []).append(box_area(box))
            if selection != "lower-face-motion":
                continue
            frame = getattr(result, "orig_img", None)
            if frame is None:
                continue
            crop = lower_face_crop(frame, box)
            if crop is None:
                continue
            previous = previous_lower_crop.get(track_id)
            if previous is not None:
                motion = float(np.mean(cv2.absdiff(previous, crop))) / 255.0
                lower_motion_by_track.setdefault(track_id, []).append(motion)
            previous_lower_crop[track_id] = crop
    if not area_by_track:
        return None

    if selection == "largest":
        return max(area_by_track, key=lambda item: (float(np.mean(area_by_track[item])), len(area_by_track[item])))
    if selection == "lower-face-motion":
        return max(
            area_by_track,
            key=lambda item: (
                float(np.mean(lower_motion_by_track.get(item, [0.0]))),
                len(area_by_track[item]),
                float(np.mean(area_by_track[item])),
            ),
        )
    return max(area_by_track, key=lambda item: (len(area_by_track[item]), float(np.mean(area_by_track[item]))))


def collect_face_boxes(
    video_path: Path,
    yolo_model: Any,
    *,
    conf: float,
    iou: float,
    tracker: str,
    track_selection: str,
    device: str,
) -> list[tuple[float, float, float, float] | None]:
    results = list(
        yolo_model.track(
            source=str(video_path),
            stream=True,
            persist=True,
            tracker=tracker,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
    )
    track_id = choose_track(results, track_selection)
    boxes_by_frame: list[tuple[float, float, float, float] | None] = []
    fallback_index = -1
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            boxes_by_frame.append(None)
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        ids_np, fallback_index = track_ids_for_boxes(result, fallback_index)
        if track_id is not None:
            if track_id in ids_np:
                boxes_by_frame.append(tuple(float(v) for v in xyxy[ids_np.index(track_id)]))
                continue
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        boxes_by_frame.append(tuple(float(v) for v in xyxy[int(np.argmax(areas))]))
    return boxes_by_frame


def interpolate_boxes(
    boxes: Sequence[tuple[float, float, float, float] | None],
) -> list[tuple[float, float, float, float] | None]:
    valid_indices = [index for index, box in enumerate(boxes) if box is not None]
    if not valid_indices:
        return [None] * len(boxes)

    filled: list[tuple[float, float, float, float] | None] = list(boxes)
    first = valid_indices[0]
    last = valid_indices[-1]
    for index in range(0, first):
        filled[index] = filled[first]
    for index in range(last + 1, len(filled)):
        filled[index] = filled[last]

    for left, right in zip(valid_indices, valid_indices[1:]):
        left_box = np.array(filled[left], dtype=np.float32)
        right_box = np.array(filled[right], dtype=np.float32)
        gap = right - left
        for offset in range(1, gap):
            alpha = offset / gap
            box = (1.0 - alpha) * left_box + alpha * right_box
            filled[left + offset] = tuple(float(v) for v in box)
    return filled


def smooth_boxes(
    boxes: Sequence[tuple[float, float, float, float] | None],
    window: int,
) -> list[tuple[float, float, float, float] | None]:
    if window <= 1 or not boxes or all(box is None for box in boxes):
        return list(boxes)
    if window % 2 == 0:
        window += 1
    radius = window // 2
    smoothed: list[tuple[float, float, float, float] | None] = []
    for index, box in enumerate(boxes):
        if box is None:
            smoothed.append(None)
            continue
        neighbors = [
            boxes[j]
            for j in range(max(0, index - radius), min(len(boxes), index + radius + 1))
            if boxes[j] is not None
        ]
        smoothed.append(tuple(float(v) for v in np.mean(np.array(neighbors, dtype=np.float32), axis=0)))
    return smoothed


def crop_from_box(
    frame: np.ndarray,
    box: tuple[float, float, float, float] | None,
    *,
    crop_size: int,
    padding: float,
) -> np.ndarray:
    height, width = frame.shape[:2]
    if box is None:
        side = min(width, height)
        cx = width / 2.0
        cy = height / 2.0
    else:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        side = max(x2 - x1, y2 - y1) * padding
    side = max(1.0, min(float(max(width, height)), side))

    half = side / 2.0
    left = int(round(cx - half))
    top = int(round(cy - half))
    right = int(round(cx + half))
    bottom = int(round(cy + half))

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - width)
    pad_bottom = max(0, bottom - height)
    if pad_left or pad_top or pad_right or pad_bottom:
        frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
        left += pad_left
        right += pad_left
        top += pad_top
        bottom += pad_top

    crop = frame[top:bottom, left:right]
    return cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)


def mux_audio(video_without_audio: Path, source_video: Path, video_with_audio: Path) -> None:
    try:
        run_command(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_without_audio),
                "-i",
                str(source_video),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(video_with_audio),
            ]
        )
    except subprocess.CalledProcessError:
        shutil.copy2(video_without_audio, video_with_audio)


def crop_face_video(
    video_path: Path,
    out_video_path: Path,
    yolo_model: Any,
    *,
    conf: float,
    iou: float,
    tracker: str,
    crop_size: int,
    face_padding: float,
    smooth_window: int,
    track_selection: str,
    fallback: str,
    device: str,
    keep_temp: bool,
) -> tuple[bool, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    boxes = collect_face_boxes(
        video_path,
        yolo_model,
        conf=conf,
        iou=iou,
        tracker=tracker,
        track_selection=track_selection,
        device=device,
    )
    if total_frames and len(boxes) < total_frames:
        boxes.extend([None] * (total_frames - len(boxes)))
    elif total_frames and len(boxes) > total_frames:
        boxes = boxes[:total_frames]

    valid_count = sum(1 for box in boxes if box is not None)
    face_coverage = valid_count / len(boxes) if boxes else 0.0
    face_detected = valid_count > 0

    if not face_detected:
        if fallback == "skip":
            raise RuntimeError(f"No face detected in {video_path}")
        if fallback == "copy":
            out_video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video_path, out_video_path)
            return False, 0.0

    boxes = smooth_boxes(interpolate_boxes(boxes), smooth_window)

    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="face_crop_"))
    silent_video = temp_dir / f"{out_video_path.stem}.silent.mp4"
    writer = cv2.VideoWriter(
        str(silent_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps if math.isfinite(fps) and fps > 0 else 25.0),
        (crop_size, crop_size),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {silent_video}")

    cap = cv2.VideoCapture(str(video_path))
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        box = boxes[frame_index] if frame_index < len(boxes) else None
        writer.write(crop_from_box(frame, box, crop_size=crop_size, padding=face_padding))
        frame_index += 1
    cap.release()
    writer.release()

    mux_audio(silent_video, video_path, out_video_path)
    if not keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return face_detected, face_coverage


def append_manifest(manifest_path: Path, sample: LabeledSample) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_id": sample.sample_id,
        "source_video": str(sample.source_video),
        "video_path": str(sample.video_path),
        "audio_path": str(sample.audio_path),
        "csv_path": str(sample.csv_path),
        "transcript": sample.transcript,
        "duration": sample.duration,
        "face_detected": sample.face_detected,
        "face_coverage": sample.face_coverage,
    }
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.out_root = args.out_root.resolve()
    configure_runtime_cache(Path.cwd())

    videos = iter_videos(args.data_root)
    if args.limit is not None:
        videos = videos[: args.limit]
    if not videos:
        raise SystemExit(f"No videos found under {args.data_root}")

    # Load existing manifest for caching metadata
    manifest_cache = {}
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / args.manifest_name
    if args.overwrite and manifest_path.exists():
        manifest_path.unlink()
    elif manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            manifest_cache[item["sample_id"]] = item
                        except Exception:
                            pass
        except Exception as exc:
            print(f"Warning: could not read manifest for cache: {exc}")

    # Check if we need YOLO model for cropping
    need_cropping = False
    for video_path in videos:
        target_dir = output_dir_for(video_path, args.data_root, args.out_root, args.layout)
        stem = video_path.stem
        out_video = target_dir / f"{stem}.mp4"
        out_audio = target_dir / f"{stem}.wav"
        out_csv = target_dir / f"{stem}.csv"
        
        all_exist = not args.overwrite and out_video.exists() and out_audio.exists() and out_csv.exists()
        if not all_exist:
            if args.overwrite or not out_video.exists():
                need_cropping = True
                break

    # ==========================================
    # PHASE 1: Face Cropping & Audio Extraction
    # ==========================================
    if need_cropping:
        from ultralytics import YOLO
        yolo_model = YOLO(args.yolo_face_model)
    else:
        yolo_model = None

    # Store video metadata for Phase 2
    video_tasks = []
    
    processed = 0
    skipped = 0
    failed: list[tuple[Path, str]] = []
    
    for video_path in tqdm(videos, desc="Phase 1: Cropping & Audio Extraction"):
        try:
            target_dir = output_dir_for(video_path, args.data_root, args.out_root, args.layout)
            target_dir.mkdir(parents=True, exist_ok=True)
            stem = video_path.stem
            out_video = target_dir / f"{stem}.mp4"
            out_audio = target_dir / f"{stem}.wav"
            out_csv = target_dir / f"{stem}.csv"

            if out_video.resolve() == video_path.resolve():
                raise ValueError("--out-root must not resolve output videos onto the input videos")

            all_exist = not args.overwrite and out_video.exists() and out_audio.exists() and out_csv.exists()
            if all_exist:
                skipped += 1
                continue

            # Crop face video if needed
            crop_run = False
            if not args.overwrite and out_video.exists():
                tqdm.write(f"[skip crop] {out_video.name} already exists.")
                cached = manifest_cache.get(stem, {})
                face_detected = cached.get("face_detected", True)
                face_coverage = cached.get("face_coverage", 1.0)
            else:
                crop_run = True
                face_detected, face_coverage = crop_face_video(
                    video_path,
                    out_video,
                    yolo_model,
                    conf=args.conf,
                    iou=args.iou,
                    tracker=args.tracker,
                    crop_size=args.crop_size,
                    face_padding=args.face_padding,
                    smooth_window=args.smooth_window,
                    track_selection=args.track_selection,
                    fallback=args.fallback,
                    device=args.device,
                    keep_temp=args.keep_temp,
                )

            # Extract audio WAV if needed
            duration = ffprobe_duration(out_video)
            audio_run = False
            if not args.overwrite and out_audio.exists():
                tqdm.write(f"[skip audio] {out_audio.name} already exists.")
            else:
                audio_run = True
                extract_audio(out_video, out_audio)

            if crop_run or audio_run:
                processed += 1

            # Decide on CSV / Transcription
            if not args.overwrite and out_csv.exists():
                tqdm.write(f"[skip transcribe] {out_csv.name} already exists.")
                if stem not in manifest_cache:
                    transcript = read_transcript_from_csv(out_csv)
                    sample = LabeledSample(
                        sample_id=stem,
                        source_video=video_path,
                        video_path=out_video,
                        audio_path=out_audio,
                        csv_path=out_csv,
                        transcript=transcript,
                        duration=duration,
                        face_detected=face_detected,
                        face_coverage=face_coverage,
                    )
                    append_manifest(manifest_path, sample)
                    manifest_cache[stem] = {
                        "sample_id": stem,
                        "source_video": str(video_path),
                        "video_path": str(out_video),
                        "audio_path": str(out_audio),
                        "csv_path": str(out_csv),
                        "transcript": transcript,
                        "duration": duration,
                        "face_detected": face_detected,
                        "face_coverage": face_coverage,
                    }
            else:
                # Keep task info for Phase 2
                video_tasks.append({
                    "stem": stem,
                    "source_video": video_path,
                    "video_path": out_video,
                    "audio_path": out_audio,
                    "csv_path": out_csv,
                    "duration": duration,
                    "face_detected": face_detected,
                    "face_coverage": face_coverage,
                })
        except Exception as exc:
            failed.append((video_path, str(exc)))
            tqdm.write(f"[failed crop/audio] {video_path}: {exc}")

    # Free YOLO memory
    if yolo_model is not None:
        del yolo_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not video_tasks:
        print(f"No new videos to transcribe. Processed: {processed}, Skipped: {skipped}, Failed: {len(failed)}")
        return

    # ==========================================
    # PHASE 2: Batched Transcription (Whisper)
    # ==========================================
    asr = load_asr_pipeline(args.phowhisper_model, args.device, args.batch_size)

    # Generator for streaming paths to HF pipeline
    def audio_generator():
        for task in video_tasks:
            yield str(task["audio_path"])

    kwargs = {
        "chunk_length_s": args.chunk_length_s,
        "generate_kwargs": {"task": args.task, "language": args.language},
        "return_timestamps": "word" if not args.no_word_timestamps else False,
    }

    # Stream transcription in batches
    transcription_iterator = asr(audio_generator(), batch_size=args.batch_size, **kwargs)

    for task, result in zip(video_tasks, tqdm(transcription_iterator, total=len(video_tasks), desc="Phase 2: Transcribing")):
        try:
            transcript = str(result.get("text", "")).strip()
            words = []
            for chunk in result.get("chunks", []) or []:
                text = str(chunk.get("text", "")).strip()
                if not text:
                    continue
                start, end = normalize_timestamp(chunk.get("timestamp"))
                if start is None:
                    continue
                if end is None:
                    end = task["duration"]
                words.append({"word": text, "start": max(0.0, start), "end": min(task["duration"], end)})

            if not words:
                words = heuristic_word_timestamps(transcript, task["duration"])
            
            write_timestamp_csv(task["csv_path"], words)

            sample = LabeledSample(
                sample_id=task["stem"],
                source_video=task["source_video"],
                video_path=task["video_path"],
                audio_path=task["audio_path"],
                csv_path=task["csv_path"],
                transcript=transcript,
                duration=task["duration"],
                face_detected=task["face_detected"],
                face_coverage=task["face_coverage"],
            )
            append_manifest(manifest_path, sample)
        except Exception as exc:
            failed.append((task["source_video"], f"Transcription error: {exc}"))
            tqdm.write(f"[failed transcribe] {task['source_video']}: {exc}")

    # Print Summary
    print(f"Processed (Cropped & Transcribed): {len(video_tasks)}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {len(failed)}")
    if failed:
        failures_path = args.out_root / "failures.jsonl"
        with failures_path.open("w", encoding="utf-8") as handle:
            for path, error in failed:
                handle.write(json.dumps({"video_path": str(path), "error": error}, ensure_ascii=False) + "\n")
        print(f"Wrote failures: {failures_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
