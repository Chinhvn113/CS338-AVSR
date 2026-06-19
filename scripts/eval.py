#!/usr/bin/env python3
"""Evaluate AVSR checkpoints on a test split.

Modes:
- av: audio + visual (standard fusion)
- audio-only: ignore visual tokens via an all-zero attention mask
- visual-only: ignore audio and decode from visual features only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from core.checkpoint import load_checkpoint
from core.data import AVSRCollator, AVSRDataset, discover_samples, load_word_timestamps
from core.metrics import character_error_rate, word_error_rate
from core.model import AVWhisperConfig, AVWhisperModel
from core.preprocessing import load_audio_features, load_audio_waveform, load_video_clip
from transformers import WhisperProcessor

DEFAULT_EVAL_CONFIG: dict[str, Any] = {
    "checkpoint": None,
    "test_root": None,
    "split": "test",
    "batch_size": 2,
    "num_workers": 2,
    "scan_workers": None,
    "sample_cache_path": None,
    "scan_log_every": 0,
    "feature_cache_dir": None,
    "cache_video": True,
    "cache_audio": True,
    "skip_errors": False,
    "max_skip_errors": 10,
    "prefetch_factor": 2,
    "persistent_workers": True,
    "max_video_frames": None,
    "image_size": None,
    "video_grayscale": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "eval_max_batches": None,
    "mode": "av",
    "language": None,
    "task": None,
    "use_video_audio": True,
    "max_new_tokens": 128,
    "num_beams": 1,
    "chunk_seconds": None,
    "chunk_overlap_seconds": 0.0,
    "compare_modes": ["audio-only", "av"],
    "compare_snr_db": [0, 5, 10],
    "noise_seed": 338,
    "all_noise": False,
    "snr_db": None,
}

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "eval.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AVSR checkpoints on a test split.")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--config", type=Path, help="Path to eval YAML config.")
    parser.add_argument("--all-noise", action="store_true", default=None, help="Add all noises (Gaussian, Pink, Brown, Hum) simultaneously.")
    parser.add_argument("--snr-db", type=float, default=None, help="Target SNR for noise injection.")
    return parser.parse_args()


def load_eval_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing config at {config_path}. Create it or adjust the path."
        )
    import yaml

    loaded = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Eval config must be a YAML mapping at top-level.")
    config = DEFAULT_EVAL_CONFIG.copy()
    config.update(loaded)
    return config


def build_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    config_path = cli_args.config or CONFIG_PATH
    config = load_eval_config(config_path)
    config["data_root"] = cli_args.data_root

    if cli_args.all_noise is not None:
        config["all_noise"] = cli_args.all_noise
    if cli_args.snr_db is not None:
        config["snr_db"] = cli_args.snr_db

    path_keys = {"data_root", "test_root", "checkpoint"}
    if config.get("sample_cache_path") is not None:
        path_keys.add("sample_cache_path")
    if config.get("feature_cache_dir") is not None:
        path_keys.add("feature_cache_dir")
    for key in path_keys:
        if config.get(key) is not None:
            config[key] = Path(config[key])

    return argparse.Namespace(**config)


def infer_test_root(args: argparse.Namespace) -> Path:
    if args.test_root is not None:
        return args.test_root
    if args.data_root is None:
        raise ValueError("Provide --data-root or --test-root")
    split_path = args.data_root / args.split
    return split_path if split_path.exists() else args.data_root


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def eval_result_dir(checkpoint_path: Path) -> Path:
    checkpoint_str = str(checkpoint_path)
    if not checkpoint_path.exists():
        folder_name = checkpoint_str.replace("/", "_")
        return Path("results") / "eval" / folder_name

    resolved_checkpoint = checkpoint_path.expanduser().resolve()
    cwd = Path.cwd().resolve()
    try:
        model_path = resolved_checkpoint.relative_to(cwd)
    except ValueError:
        model_path = Path(*resolved_checkpoint.parts[1:])
    if model_path.suffix:
        model_path = model_path.with_suffix("")
    return Path("results") / "eval" / model_path


def chunk_ranges(timestamps: list[dict[str, Any]], chunk_seconds: float, overlap_seconds: float) -> list[tuple[float, float]]:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be greater than 0")
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("chunk_overlap_seconds must be >= 0 and < chunk_seconds")
    duration = max((float(item["end_time"]) for item in timestamps), default=0.0)
    if duration <= 0:
        return []
    ranges = []
    start = 0.0
    step = chunk_seconds - overlap_seconds
    while start < duration:
        end = min(start + chunk_seconds, duration)
        ranges.append((start, end))
        if end >= duration:
            break
        start += step
    return ranges


def transcript_for_range(timestamps: list[dict[str, Any]], start: float, end: float, *, is_last: bool) -> str:
    words = []
    for item in timestamps:
        midpoint = (float(item["start_time"]) + float(item["end_time"])) / 2.0
        if start <= midpoint < end or (is_last and start <= midpoint <= end):
            words.append(str(item["word"]))
    return " ".join(words)


def write_media_segment(source_path: Path, output_path: Path, start: float, end: float, *, audio: bool) -> None:
    duration = max(end - start, 0.01)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(source_path),
    ]
    if audio:
        command.extend(["-vn", "-ac", "1", str(output_path)])
    else:
        command.extend([
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ])
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for chunked eval") from exc


def add_noise_at_snr(
    waveform: torch.Tensor,
    snr_db: float | None,
    *,
    seed: int,
    all_noise: bool = False,
    sample_rate: int = 16000,
) -> torch.Tensor:
    # Use default 10 dB SNR if snr_db is None but noise is requested
    target_snr = snr_db if snr_db is not None else 10.0
    
    signal_power = waveform.pow(2).mean().clamp_min(1e-12)
    noise_power = signal_power / (10 ** (target_snr / 10.0))
    generator = torch.Generator(device=waveform.device)
    generator.manual_seed(seed)

    if all_noise:
        shape = waveform.shape
        device = waveform.device
        dtype = waveform.dtype

        # 1. Gaussian noise
        g_noise = torch.randn(shape, generator=generator, dtype=dtype, device=device)
        g_noise = g_noise / g_noise.std().clamp_min(1e-12)

        # 2. Pink noise
        seq_len = shape[-1]
        p_white = torch.randn(shape, generator=generator, dtype=dtype, device=device)
        p_fft = torch.fft.rfft(p_white, dim=-1)
        frequencies = torch.fft.rfftfreq(seq_len, device=device)
        freq_denom = frequencies.clone()
        freq_denom[0] = freq_denom[1] if len(freq_denom) > 1 else 1e-3
        p_fft = p_fft / torch.sqrt(freq_denom)
        pink_noise = torch.fft.irfft(p_fft, n=seq_len, dim=-1)
        pink_noise = pink_noise / pink_noise.std().clamp_min(1e-12)

        # 3. Brown noise
        b_white = torch.randn(shape, generator=generator, dtype=dtype, device=device)
        b_fft = torch.fft.rfft(b_white, dim=-1)
        b_fft = b_fft / freq_denom
        brown_noise = torch.fft.irfft(b_fft, n=seq_len, dim=-1)
        brown_noise = brown_noise / brown_noise.std().clamp_min(1e-12)

        # 4. Hum / background noise (harmonics)
        t = torch.arange(seq_len, device=device, dtype=dtype) / sample_rate
        freq = 50 if seed % 2 == 0 else 60
        hum = torch.sin(2 * 3.14159265 * freq * t)
        hum += 0.5 * torch.sin(2 * 3.14159265 * freq * 2 * t)
        hum += 0.25 * torch.sin(2 * 3.14159265 * freq * 3 * t)
        hum = hum / hum.std().clamp_min(1e-12)
        if len(shape) > 1:
            hum = hum.unsqueeze(0).expand(shape)

        # Combined noise
        noise = g_noise + pink_noise + brown_noise + hum
    else:
        noise = torch.randn(waveform.shape, generator=generator, dtype=waveform.dtype, device=waveform.device)

    current_noise_power = noise.pow(2).mean().clamp_min(1e-12)
    noise = noise * torch.sqrt(noise_power / current_noise_power)
    return waveform + noise


def load_noisy_audio_features(
    audio_path: Path,
    processor: WhisperProcessor,
    *,
    snr_db: float | None,
    seed: int,
    all_noise: bool = False,
) -> torch.Tensor:
    if snr_db is None and not all_noise:
        return load_audio_features(audio_path, processor)
    sample_rate = processor.feature_extractor.sampling_rate
    waveform = load_audio_waveform(audio_path, target_sample_rate=sample_rate)
    waveform = add_noise_at_snr(waveform, snr_db, seed=seed, all_noise=all_noise, sample_rate=sample_rate)
    features = processor.feature_extractor(
        waveform.numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    return features.input_features.squeeze(0)


class NoisyAVSRDataset(AVSRDataset):
    def __init__(self, *args, all_noise: bool = False, snr_db: float | None = None, noise_seed: int = 338, **kwargs):
        if all_noise or snr_db is not None:
            kwargs["cache_audio"] = False
        super().__init__(*args, **kwargs)
        self.all_noise = all_noise
        self.snr_db = snr_db
        self.noise_seed = noise_seed

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not (self.all_noise or self.snr_db is not None):
            return super().__getitem__(index)
        
        sample = self.samples[index]
        try:
            from core.data import transcript_from_timestamps
            timestamps = load_word_timestamps(sample.csv_path)
            transcript = transcript_from_timestamps(timestamps)

            video = None
            if self.cache_dir is not None and self.cache_video:
                cache_key = self._cache_key(sample)
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
                audio_seed = self.noise_seed + index
                audio = load_noisy_audio_features(
                    sample.audio_path,
                    self.processor,
                    snr_db=self.snr_db,
                    seed=audio_seed,
                    all_noise=self.all_noise,
                )

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
        except Exception as exc:
            if not self.skip_errors:
                raise
            print(f"[warn] Skipping sample {sample.sample_id}: {exc}")
            return self.__getitem__((index + 1) % len(self.samples))


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def evaluate_chunked(
    *,
    args: argparse.Namespace,
    model: AVWhisperModel,
    processor: WhisperProcessor,
    device: torch.device,
    test_root: Path,
    image_size: int,
    max_video_frames: int | None,
    video_grayscale: bool,
    eval_mode: str | None = None,
    snr_db: float | None = None,
    noise_seed_offset: int = 0,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    samples = discover_samples(
        test_root,
        strict=False,
        allow_video_audio=args.use_video_audio,
        scan_workers=args.scan_workers,
        cache_path=args.sample_cache_path,
        log_every=args.scan_log_every,
    )
    if not samples:
        raise ValueError("No AVSR samples found")

    references = []
    hypotheses = []
    predictions = []
    chunk_seconds = float(args.chunk_seconds)
    overlap_seconds = float(args.chunk_overlap_seconds or 0.0)
    skipped = 0
    sample_limit = args.eval_max_batches
    active_mode = eval_mode or args.mode

    progress = tqdm(samples, desc=f"chunk eval {active_mode}", leave=False)
    for sample_index, sample in enumerate(progress, start=1):
        if sample_limit is not None and sample_index > sample_limit:
            break
        try:
            timestamps = load_word_timestamps(sample.csv_path)
            ranges = chunk_ranges(timestamps, chunk_seconds, overlap_seconds)
            chunk_payloads = []
            sample_refs = []
            sample_hyps = []
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                for chunk_index, (start, end) in enumerate(ranges):
                    reference_end = ranges[chunk_index + 1][0] if chunk_index + 1 < len(ranges) else end
                    reference = transcript_for_range(
                        timestamps,
                        start,
                        reference_end,
                        is_last=chunk_index == len(ranges) - 1,
                    )
                    if not reference:
                        continue

                    video = None
                    video_attention_mask = None
                    audio_features = None

                    if active_mode != "audio-only":
                        video_segment = tmp_path / f"chunk_{chunk_index}.mp4"
                        write_media_segment(sample.video_path, video_segment, start, end, audio=False)
                        video = load_video_clip(
                            video_segment,
                            image_size=image_size,
                            max_frames=max_video_frames,
                            grayscale=video_grayscale,
                        ).unsqueeze(0).to(device)
                        video_attention_mask = torch.ones(
                            1,
                            video.shape[1],
                            dtype=torch.long,
                            device=device,
                        )

                    if active_mode != "visual-only":
                        audio_segment = tmp_path / f"chunk_{chunk_index}.wav"
                        write_media_segment(sample.audio_path, audio_segment, start, end, audio=True)
                        audio_seed = int(args.noise_seed) + noise_seed_offset + sample_index * 100000 + chunk_index
                        audio_features = load_noisy_audio_features(
                            audio_segment,
                            processor,
                            snr_db=snr_db,
                            seed=audio_seed,
                            all_noise=getattr(args, "all_noise", False),
                        ).unsqueeze(0).to(device)

                    generated_ids = model.generate(
                        video=video,
                        audio_features=audio_features,
                        video_attention_mask=video_attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=args.num_beams,
                    )
                    hypothesis = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    sample_refs.append(reference)
                    sample_hyps.append(hypothesis)
                    chunk_payloads.append({
                        "start": start,
                        "end": end,
                        "reference": reference,
                        "hypothesis": hypothesis,
                    })

            sample_reference = " ".join(sample_refs)
            sample_hypothesis = " ".join(sample_hyps)
            references.append(sample_reference)
            hypotheses.append(sample_hypothesis)
            predictions.append({
                "sample_id": sample.sample_id,
                "reference": sample_reference,
                "hypothesis": sample_hypothesis,
                "chunks": chunk_payloads,
            })
        except Exception as exc:
            if not args.skip_errors:
                raise
            skipped += 1
            print(f"[warn] Skipping sample {sample.sample_id}: {exc}")
            if skipped > args.max_skip_errors:
                raise RuntimeError("Too many chunked eval errors") from exc

    return references, hypotheses, predictions


def main() -> None:
    args = build_args(parse_args())
    device = torch.device(args.device)

    if args.checkpoint is None:
        raise ValueError("Missing checkpoint in configs/eval.yaml")
    
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists() and checkpoint_path.is_file():
        checkpoint = load_checkpoint(args.checkpoint, map_location=device)
        raw_model_config = checkpoint["model_config"]
        model_config = AVWhisperConfig.from_dict(raw_model_config)
        state_dict = checkpoint["model_state_dict"]
        data_config = checkpoint.get("data_config", {})
    else:
        # Treat as Hugging Face model hub ID directly
        print(f"[info] Checkpoint path '{args.checkpoint}' not found on disk. Loading base model directly from HuggingFace...", flush=True)
        model_config = AVWhisperConfig(whisper_model_name=str(args.checkpoint))
        state_dict = None
        data_config = {}

    processor_kwargs = {}
    language = args.language or data_config.get("language")
    task = args.task or data_config.get("task")
    if language:
        processor_kwargs["language"] = language
    if task:
        processor_kwargs["task"] = task

    processor = WhisperProcessor.from_pretrained(model_config.whisper_model_name, **processor_kwargs)

    if args.mode == "visual-only":
        model_config.visual_only = True
    if args.mode == "audio-only":
        model_config.visual_only = False
    model = AVWhisperModel(model_config).to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.eval()

    image_size = args.image_size or data_config.get("image_size", 224)
    max_video_frames = args.max_video_frames or data_config.get("max_video_frames", 96)
    video_grayscale = data_config.get("video_grayscale", model_config.video_channels == 1)
    if args.video_grayscale is not None:
        video_grayscale = args.video_grayscale

    test_root = infer_test_root(args)
    if args.mode == "compare":
        if args.chunk_seconds is None:
            raise ValueError("compare mode requires chunk_seconds so noisy audio is evaluated on short windows")
        compare_modes = [str(mode) for mode in as_list(args.compare_modes)]
        snr_levels = [float(snr) for snr in as_list(args.compare_snr_db)]
        if not compare_modes or not snr_levels:
            raise ValueError("compare mode requires compare_modes and compare_snr_db")

        results = []
        for mode_index, compare_mode in enumerate(compare_modes):
            if compare_mode not in {"audio-only", "av"}:
                raise ValueError(f"Unsupported compare mode: {compare_mode}")
            for snr_index, snr_db in enumerate(snr_levels):
                references, hypotheses, prediction_payload = evaluate_chunked(
                    args=args,
                    model=model,
                    processor=processor,
                    device=device,
                    test_root=test_root,
                    image_size=image_size,
                    max_video_frames=max_video_frames,
                    video_grayscale=video_grayscale,
                    eval_mode=compare_mode,
                    snr_db=snr_db,
                    noise_seed_offset=mode_index * 10000000 + snr_index * 1000000,
                )
                metrics = {}
                if references:
                    metrics["wer"] = word_error_rate(references, hypotheses)
                    metrics["cer"] = character_error_rate(references, hypotheses)
                results.append({
                    "mode": compare_mode,
                    "snr_db": snr_db,
                    "metrics": metrics,
                    "num_samples": len(references),
                    "predictions": prediction_payload,
                })
                metric_text = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
                print(f"compare {compare_mode} snr_db={snr_db:g}: {metric_text}")

        output_dir = eval_result_dir(args.checkpoint)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "compare.json"
        payload = {
            "checkpoint": str(args.checkpoint),
            "test_root": str(test_root),
            "split": args.split,
            "mode": args.mode,
            "compare_modes": compare_modes,
            "compare_snr_db": snr_levels,
            "noise_seed": args.noise_seed,
            "chunk_seconds": args.chunk_seconds,
            "chunk_overlap_seconds": args.chunk_overlap_seconds,
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "results": results,
        }
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"saved compare result: {output_path}")
        return


    if args.chunk_seconds is not None:
        references, hypotheses, prediction_payload = evaluate_chunked(
            args=args,
            model=model,
            processor=processor,
            device=device,
            test_root=test_root,
            image_size=image_size,
            max_video_frames=max_video_frames,
            video_grayscale=video_grayscale,
        )
        metrics = {}
        if references:
            metrics["wer"] = word_error_rate(references, hypotheses)
            metrics["cer"] = character_error_rate(references, hypotheses)
        output_dir = eval_result_dir(args.checkpoint)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"eval_{args.mode}.json"
        payload = {
            "checkpoint": str(args.checkpoint),
            "test_root": str(test_root),
            "split": args.split,
            "mode": args.mode,
            "chunk_seconds": args.chunk_seconds,
            "chunk_overlap_seconds": args.chunk_overlap_seconds,
            "metrics": metrics,
            "num_samples": len(references),
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "predictions": prediction_payload,
        }
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        metric_text = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        print(metric_text)
        print(f"saved eval result: {output_path}")
        return


    dataset = NoisyAVSRDataset(
        test_root,
        processor=processor,
        image_size=image_size,
        max_video_frames=max_video_frames,
        video_grayscale=video_grayscale,
        allow_video_audio=args.use_video_audio,
        visual_only=args.mode == "visual-only",
        scan_workers=args.scan_workers,
        sample_cache_path=args.sample_cache_path,
        scan_log_every=args.scan_log_every,
        feature_cache_dir=args.feature_cache_dir,
        cache_video=args.cache_video,
        cache_audio=args.cache_audio,
        skip_errors=args.skip_errors,
        max_skip_errors=args.max_skip_errors,
        all_noise=getattr(args, "all_noise", False),
        snr_db=getattr(args, "snr_db", None),
        noise_seed=args.noise_seed,
    )
    collator = AVSRCollator(processor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )

    references: list[str] = []
    hypotheses: list[str] = []

    progress = tqdm(loader, desc=f"eval {args.mode}", leave=False)
    for batch_index, batch in enumerate(progress, start=1):
        batch = move_batch_to_device(batch, device)

        video = batch["video"]
        video_attention_mask = batch["video_attention_mask"]
        audio_features = batch.get("audio_features")

        if args.mode == "audio-only":
            video = None
            video_attention_mask = None

        generated_ids = model.generate(
            video=video,
            audio_features=audio_features,
            video_attention_mask=video_attention_mask,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        hypotheses.extend(text.strip() for text in processor.batch_decode(generated_ids, skip_special_tokens=True))
        references.extend(batch["transcript"])

        if args.eval_max_batches is not None and batch_index >= args.eval_max_batches:
            break

    metrics = {}
    if references:
        metrics["wer"] = word_error_rate(references, hypotheses)
        metrics["cer"] = character_error_rate(references, hypotheses)
    output_dir = eval_result_dir(args.checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"eval_{args.mode}.json"
    payload = {
        "checkpoint": str(args.checkpoint),
        "test_root": str(test_root),
        "split": args.split,
        "mode": args.mode,
        "metrics": metrics,
        "num_samples": len(references),
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "predictions": [
            {"reference": reference, "hypothesis": hypothesis}
            for reference, hypothesis in zip(references, hypotheses)
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    metric_text = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
    print(metric_text)
    print(f"saved eval result: {output_path}")


if __name__ == "__main__":
    main()
