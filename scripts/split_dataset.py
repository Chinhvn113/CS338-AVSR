#!/usr/bin/env python3
"""Split dataset into train/val/test with paired video+csv files.

Default: 90/10 train/test, no val. Set --val-ratio to enable val split.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a"}


@dataclass(frozen=True)
class Pair:
    stem: str
    video_path: Path
    csv_path: Path
    audio_path: Path | None


def _classify_path(path: Path) -> tuple[str | None, str, Path]:
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTS:
        return ("video", path.stem, path)
    if suffix in AUDIO_EXTS:
        return ("audio", path.stem, path)
    if suffix == ".csv":
        return ("csv", path.stem, path)
    return (None, path.stem, path)


def _iter_files(root: Path, *, log_every: int) -> Iterable[Path]:
    scanned = 0
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            scanned += 1
            if log_every > 0 and scanned % log_every == 0:
                print(f"[scan] Listed {scanned} files", flush=True)
            yield Path(dirpath) / filename


def find_pairs(
    data_root: Path,
    *,
    scan_workers: int,
    log_every: int,
) -> tuple[list[Pair], dict[str, int]]:
    videos: dict[str, Path] = {}
    csvs: dict[str, Path] = {}
    audios: dict[str, Path] = {}
    missing_csv_for_video = 0
    missing_video_for_csv = 0
    missing_audio_for_csv = 0

    if scan_workers < 1:
        scan_workers = 1
    with ThreadPoolExecutor(max_workers=scan_workers) as executor:
        for index, (kind, stem, path) in enumerate(
            executor.map(_classify_path, _iter_files(data_root, log_every=log_every)),
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

    pairs: list[Pair] = []
    for stem, video_path in videos.items():
        csv_path = csvs.get(stem)
        if csv_path is None:
            print(f"[warn] Missing CSV for video: {video_path}")
            missing_csv_for_video += 1
            continue
        pairs.append(
            Pair(
                stem=stem,
                video_path=video_path,
                csv_path=csv_path,
                audio_path=audios.get(stem),
            )
        )

    for stem, csv_path in csvs.items():
        if stem not in videos:
            print(f"[warn] Missing video for CSV: {csv_path}")
            missing_video_for_csv += 1
        elif stem not in audios:
            print(f"[warn] Missing audio for CSV: {csv_path}")
            missing_audio_for_csv += 1

    stats = {
        "videos": len(videos),
        "csvs": len(csvs),
        "audios": len(audios),
        "missing_csv_for_video": missing_csv_for_video,
        "missing_video_for_csv": missing_video_for_csv,
        "missing_audio_for_csv": missing_audio_for_csv,
    }
    return pairs, stats


def split_pairs(pairs: list[Pair], train_ratio: float, val_ratio: float, seed: int) -> tuple[list[Pair], list[Pair], list[Pair]]:
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if val_ratio < 0 or val_ratio >= 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    pairs_copy = list(pairs)
    random.Random(seed).shuffle(pairs_copy)

    n_total = len(pairs_copy)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train = pairs_copy[:n_train]
    val = pairs_copy[n_train : n_train + n_val]
    test = pairs_copy[n_train + n_val :]

    assert len(test) == n_test
    return train, val, test


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    ensure_dir(dst.parent)
    if mode == "symlink":
        if dst.exists() or dst.is_symlink():
            return
        os.symlink(src.resolve(), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError("mode must be 'copy' or 'symlink'")


def extract_audio(video_path: Path, audio_dst: Path) -> None:
    ensure_dir(audio_dst.parent)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_dst),
        ],
        check=True,
    )


def export_split(
    pairs: Iterable[Pair],
    out_root: Path,
    split: str,
    mode: str,
    *,
    extract_audio_if_missing: bool,
    export_workers: int,
    log_every: int,
) -> None:
    pairs_list = list(pairs)
    if export_workers < 1:
        export_workers = 1

    def _export_one(pair: Pair) -> None:
        video_dst = out_root / split / pair.video_path.name
        csv_dst = out_root / split / pair.csv_path.name
        copy_or_link(pair.video_path, video_dst, mode)
        copy_or_link(pair.csv_path, csv_dst, mode)
        if pair.audio_path is not None:
            audio_dst = out_root / split / pair.audio_path.name
            copy_or_link(pair.audio_path, audio_dst, mode)
        elif extract_audio_if_missing:
            audio_dst = out_root / split / f"{pair.stem}.wav"
            if not audio_dst.exists():
                extract_audio(pair.video_path, audio_dst)

    print(
        f"[export:{split}] Starting {len(pairs_list)} pairs "
        f"with {export_workers} worker(s), mode={mode}, extract_audio={extract_audio_if_missing}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=export_workers) as executor:
        future_to_pair = {executor.submit(_export_one, pair): pair for pair in pairs_list}
        for index, future in enumerate(as_completed(future_to_pair), start=1):
            pair = future_to_pair[future]
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed exporting sample {pair.stem}: "
                    f"video={pair.video_path}, csv={pair.csv_path}"
                ) from exc
            if log_every > 0 and index % log_every == 0:
                print(f"[export:{split}] Processed {index}/{len(pairs_list)} pairs", flush=True)
    print(f"[export:{split}] Done {len(pairs_list)} pairs", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test with paired video+csv files.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset"), help="Root folder containing all parts")
    parser.add_argument("--out-root", type=Path, default=Path("dataset") / "splits", help="Output folder for splits")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Fraction for training set")
    parser.add_argument("--val-ratio", type=float, default=0.0, help="Fraction for validation set")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument("--mode", choices=["copy", "symlink"], default="symlink", help="How to place files in splits")
    parser.add_argument(
        "--extract-audio",
        action="store_true",
        help="Extract missing audio tracks to WAV files using ffmpeg.",
    )
    parser.add_argument(
        "--scan-workers",
        type=int,
        default= 1,
        help="Number of threads for scanning/classifying files.",
    )
    parser.add_argument(
        "--scan-log-every",
        type=int,
        default=1000,
        help="Log scanning progress every N files (0 to disable).",
    )
    parser.add_argument(
        "--export-workers",
        type=int,
        default= 1,
        help="Number of threads for exporting (copy/symlink).",
    )
    parser.add_argument(
        "--export-log-every",
        type=int,
        default=1000,
        help="Log export progress every N pairs (0 to disable).",
    )

    args = parser.parse_args()
    if args.extract_audio and shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found. Install ffmpeg or rerun without --extract-audio.")

    pairs, stats = find_pairs(
        args.data_root,
        scan_workers=args.scan_workers,
        log_every=args.scan_log_every,
    )
    if not pairs:
        raise SystemExit("No video+csv pairs found. Check data root.")

    # print("Split configuration:")
    # print(f"  data_root: {args.data_root}")
    # print(f"  out_root: {args.out_root}")
    # print(f"  train_ratio: {args.train_ratio}")
    # print(f"  val_ratio: {args.val_ratio}")
    # print(f"  seed: {args.seed}")
    # print(f"  mode: {args.mode}")
    # print(f"  extract_audio: {args.extract_audio}")
    # print(f"  scan_workers: {args.scan_workers}")
    # print("Input stats:")
    # print(f"  videos: {stats['videos']}")
    # print(f"  csvs: {stats['csvs']}")
    # print(f"  audios: {stats['audios']}")
    # print(f"  missing_csv_for_video: {stats['missing_csv_for_video']}")
    # print(f"  missing_video_for_csv: {stats['missing_video_for_csv']}")
    # print(f"  missing_audio_for_csv: {stats['missing_audio_for_csv']}")

    train, val, test = split_pairs(pairs, args.train_ratio, args.val_ratio, args.seed)

    export_split(
        train,
        args.out_root,
        "train",
        args.mode,
        extract_audio_if_missing=args.extract_audio,
        export_workers=args.export_workers,
        log_every=args.export_log_every,
    )
    if val:
        export_split(
            val,
            args.out_root,
            "val",
            args.mode,
            extract_audio_if_missing=args.extract_audio,
            export_workers=args.export_workers,
            log_every=args.export_log_every,
        )
    export_split(
        test,
        args.out_root,
        "test",
        args.mode,
        extract_audio_if_missing=args.extract_audio,
        export_workers=args.export_workers,
        log_every=args.export_log_every,
    )

    print(f"Total pairs: {len(pairs)}")
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")
    print(f"Output: {args.out_root}")


if __name__ == "__main__":
    main()
