from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .hf_cache import configure_huggingface_cache

configure_huggingface_cache()

from transformers import WhisperProcessor
from huggingface_hub import hf_hub_download

from .checkpoint import load_checkpoint, save_checkpoint
from .data import AVSRCollator, AVSRDataset, discover_samples, split_samples
from .metrics import character_error_rate, word_error_rate
from .model import AVWhisperConfig, AVWhisperModel

DEFAULT_TRAIN_CONFIG: dict[str, Any] = {
    "train_root": None,
    "val_root": None,
    "output_dir": "checkpoints",
    "whisper_model": "openai/whisper-small",
    "model_type": "flamingo",
    "visual_encoder": "resnet3d",
    "avhubert_checkpoint": None,
    "avhubert_user_dir": None,
    "avhubert_output_dim": 1024,
    "avhubert_finetuned": False,
    "language": None,
    "task": "transcribe",
    "epochs": 5,
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
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 1,
    "max_video_frames": 96,
    "image_size": 88,
    "video_grayscale": True,
    "visual_hidden_size": 512,
    "video_dropout": 0.1,
    "val_ratio": 0.1,
    "seed": 338,
    "max_new_tokens": 128,
    "num_beams": 1,
    "eval_max_batches": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "amp": False,
    "freeze_whisper": True,
    "freeze_whisper_encoder": False,
    "train_visual_encoder": True,
    "strict_data": False,
    "use_video_audio": True,
    "visual_only": False,
    "visual_pretrained_checkpoint": None,
    "max_target_positions": 448,
}

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "train.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the clip-level AVSR pipeline.")
    parser.add_argument("--data-root", type=Path, help="Dataset root. Uses train/ and val/ if present.")
    parser.add_argument("--config", type=Path, help="Path to train YAML config.")
    return parser.parse_args()


def load_train_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing config at {config_path}. Create it or adjust the path."
        )
    import yaml

    loaded = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Train config must be a YAML mapping at top-level.")
    config = DEFAULT_TRAIN_CONFIG.copy()
    config.update(loaded)
    return config


def build_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    config_path = cli_args.config or CONFIG_PATH
    config = load_train_config(config_path)
    config["data_root"] = cli_args.data_root
    config["config_name"] = config_path.stem

    path_keys = {
        "data_root",
        "train_root",
        "val_root",
        "output_dir",
        "avhubert_checkpoint",
        "avhubert_user_dir",
        "visual_pretrained_checkpoint",
        "sample_cache_path",
        "feature_cache_dir",
    }
    for key in path_keys:
        if config.get(key) is not None:
            config[key] = Path(config[key])

    return argparse.Namespace(**config)


def load_processor(args: argparse.Namespace) -> WhisperProcessor:
    kwargs: dict[str, str] = {}
    if args.language:
        kwargs["language"] = args.language
    if args.task:
        kwargs["task"] = args.task
    return WhisperProcessor.from_pretrained(args.whisper_model, **kwargs)


def resolve_avhubert_checkpoint(args: argparse.Namespace) -> str | None:
    if args.visual_encoder != "avhubert":
        return str(args.avhubert_checkpoint) if args.avhubert_checkpoint else None

    if args.avhubert_checkpoint is not None:
        return str(args.avhubert_checkpoint)

    return hf_hub_download(
        repo_id="vumichien/AV-HuBERT",
        filename="model.pt",
    )


def infer_split_roots(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.train_root is not None:
        train_root = args.train_root
    elif args.data_root is not None and (args.data_root / "train").exists():
        train_root = args.data_root / "train"
    elif args.data_root is not None:
        train_root = args.data_root
    else:
        raise ValueError("Provide --data-root or --train-root")

    if args.val_root is not None:
        val_root = args.val_root
    elif args.data_root is not None and (args.data_root / "val").exists():
        val_root = args.data_root / "val"
    else:
        val_root = None

    return train_root, val_root


def infer_test_root(args: argparse.Namespace) -> Path | None:
    if args.data_root is None:
        return None
    test_root = args.data_root / "test"
    return test_root if test_root.exists() else None


def next_run_dir(base_dir: Path, prefix: str = "run") -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith(prefix)]
    used = set()
    for path in existing:
        suffix = path.name[len(prefix) :]
        if suffix.isdigit():
            used.add(int(suffix))
    run_id = 1
    while run_id in used:
        run_id += 1
    return base_dir / f"{prefix}{run_id}"


def build_datasets(args: argparse.Namespace, processor: WhisperProcessor) -> tuple[AVSRDataset, AVSRDataset | None]:
    train_root, val_root = infer_split_roots(args)
    common_kwargs = {
        "processor": processor,
        "image_size": args.image_size,
        "max_video_frames": args.max_video_frames,
        "video_grayscale": args.video_grayscale,
        "strict": args.strict_data,
        "allow_video_audio": args.use_video_audio,
        "visual_only": args.visual_only,
        "scan_workers": args.scan_workers,
        "sample_cache_path": args.sample_cache_path,
        "scan_log_every": args.scan_log_every,
        "feature_cache_dir": args.feature_cache_dir,
        "cache_video": args.cache_video,
        "cache_audio": args.cache_audio,
        "skip_errors": args.skip_errors,
        "max_skip_errors": args.max_skip_errors,
        "max_label_length": args.max_target_positions,
    }

    if val_root is not None:
        return AVSRDataset(train_root, **common_kwargs), AVSRDataset(val_root, **common_kwargs)

    samples = discover_samples(
        train_root,
        strict=args.strict_data,
        allow_video_audio=args.use_video_audio,
        scan_workers=args.scan_workers,
        cache_path=args.sample_cache_path,
        log_every=args.scan_log_every,
    )
    train_samples, val_samples = split_samples(samples, val_ratio=args.val_ratio, seed=args.seed)
    train_dataset = AVSRDataset(samples=train_samples, **common_kwargs)
    val_dataset = AVSRDataset(samples=val_samples, **common_kwargs) if val_samples else None
    return train_dataset, val_dataset


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    payload = vars(args).copy()
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def load_visual_pretrained(model: AVWhisperModel, checkpoint_path: Path) -> None:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    prefixes = ("visual_encoder.", "video_projection.", "fusion_norm.")
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key.startswith(prefixes) or key == "video_projection_scalar"
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if unexpected:
        print("[warn] Unexpected keys in visual checkpoint:")
        for name in unexpected:
            print(f"  {name}")
    if missing:
        print("[warn] Missing visual keys when loading checkpoint:")
        for name in missing:
            print(f"  {name}")


def save_training_history(
    history: list[dict[str, float | int | None]],
    results_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))


def _history_series(
    history: list[dict[str, float | int | None]],
    key: str,
) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    values: list[float] = []
    for row in history:
        value = row.get(key)
        if value is None:
            continue
        epochs.append(int(row["epoch"]))
        values.append(float(value))
    return epochs, values


def save_training_curves(
    history: list[dict[str, float | int | None]],
    results_dir: Path,
) -> None:
    """Render train/validation metrics for the current run.

    Matplotlib is imported lazily so training can still run in minimal
    environments; the JSON history is always saved by ``save_training_history``.
    """

    if not history:
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = results_dir / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib is not installed; skipping training curve figure.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    loss_ax = axes[0]
    for key, label in (("train_loss", "train loss"), ("val_loss", "val loss")):
        epochs, values = _history_series(history, key)
        if values:
            loss_ax.plot(epochs, values, marker="o", label=label)
    loss_ax.set_title("Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(True, alpha=0.3)
    loss_ax.legend()

    error_ax = axes[1]
    for key, label in (("wer", "WER"), ("cer", "CER")):
        epochs, values = _history_series(history, key)
        if values:
            error_ax.plot(epochs, values, marker="o", label=label)
    error_ax.set_title("Error Rates")
    error_ax.set_xlabel("Epoch")
    error_ax.set_ylabel("Rate")
    error_ax.grid(True, alpha=0.3)
    if error_ax.lines:
        error_ax.legend()
    else:
        error_ax.text(
            0.5,
            0.5,
            "No validation WER/CER yet",
            ha="center",
            va="center",
            transform=error_ax.transAxes,
        )

    fig.suptitle("Training Metrics")
    fig.tight_layout()
    fig.savefig(results_dir / "training_metrics.png", dpi=160)
    plt.close(fig)


def train_one_epoch(
    *,
    model: AVWhisperModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> float:
    model.train()
    total_loss = 0.0
    total_steps = 0
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for step, batch in enumerate(progress, start=1):
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            outputs = model(
                video=batch["video"],
                audio_features=batch["audio_features"],
                video_attention_mask=batch["video_attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss / args.gradient_accumulation_steps

        scaler.scale(loss).backward()
        if step % args.gradient_accumulation_steps == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_loss = loss.item() * args.gradient_accumulation_steps
        total_loss += batch_loss
        total_steps += 1
        progress.set_postfix(loss=f"{batch_loss:.4f}")

    return total_loss / max(total_steps, 1)


@torch.no_grad()
def evaluate(
    *,
    model: AVWhisperModel,
    loader: DataLoader,
    processor: WhisperProcessor,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    references: list[str] = []
    hypotheses: list[str] = []

    progress = tqdm(loader, desc="validate", leave=False)
    for batch_index, batch in enumerate(progress, start=1):
        batch = move_batch_to_device(batch, device)
        outputs = model(
            video=batch["video"],
            audio_features=batch["audio_features"],
            video_attention_mask=batch["video_attention_mask"],
            labels=batch["labels"],
        )
        total_loss += outputs.loss.item()
        total_steps += 1

        generated_ids = model.generate(
            video=batch["video"],
            audio_features=batch["audio_features"],
            video_attention_mask=batch["video_attention_mask"],
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        hypotheses.extend(text.strip() for text in processor.batch_decode(generated_ids, skip_special_tokens=True))
        references.extend(batch["transcript"])

        if args.eval_max_batches is not None and batch_index >= args.eval_max_batches:
            break

    metrics = {"val_loss": total_loss / max(total_steps, 1)}
    if references:
        metrics["wer"] = word_error_rate(references, hypotheses)
        metrics["cer"] = character_error_rate(references, hypotheses)
    return metrics


def main() -> None:
    args = build_args(parse_args())
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    base_output_dir = Path(args.output_dir)
    prefix = f"{args.config_name}_run"
    run_dir = next_run_dir(base_output_dir, prefix=prefix)
    results_dir = Path("results") / run_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = run_dir
    print(f"run dir: {run_dir}")
    print(f"results dir: {results_dir}")

    processor = load_processor(args)
    data_load_start = time.perf_counter()
    train_dataset, val_dataset = build_datasets(args, processor)
    collator = AVSRCollator(processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
            collate_fn=collator,
            pin_memory=device.type == "cuda",
        )
        if val_dataset is not None
        else None
    )
    test_root = infer_test_root(args)
    test_loader = None
    if test_root is not None:
        test_dataset = AVSRDataset(
            test_root,
            processor=processor,
            image_size=args.image_size,
            max_video_frames=args.max_video_frames,
            video_grayscale=args.video_grayscale,
            strict=args.strict_data,
            allow_video_audio=args.use_video_audio,
            visual_only=args.visual_only,
            scan_workers=args.scan_workers,
            sample_cache_path=args.sample_cache_path,
            scan_log_every=args.scan_log_every,
            feature_cache_dir=args.feature_cache_dir,
            cache_video=args.cache_video,
            cache_audio=args.cache_audio,
            max_label_length=args.max_target_positions,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
            collate_fn=collator,
            pin_memory=device.type == "cuda",
        )
    data_load_elapsed = time.perf_counter() - data_load_start
    print(f"data loading time: {data_load_elapsed:.2f}s")

    model_config = AVWhisperConfig(
        whisper_model_name=args.whisper_model,
        fusion_type=args.model_type,
        visual_encoder_type=args.visual_encoder,
        video_channels=1 if args.video_grayscale else 3,
        visual_hidden_size=args.visual_hidden_size,
        avhubert_checkpoint=resolve_avhubert_checkpoint(args),
        avhubert_user_dir=str(args.avhubert_user_dir) if args.avhubert_user_dir else None,
        avhubert_output_size=args.avhubert_output_dim,
        avhubert_finetuned=args.avhubert_finetuned,
        video_dropout=args.video_dropout,
        freeze_whisper=args.freeze_whisper,
        freeze_whisper_encoder=args.freeze_whisper_encoder,
        train_visual_encoder=args.train_visual_encoder,
        visual_only=args.visual_only,
        max_target_positions=args.max_target_positions,
    )
    model = AVWhisperModel(model_config).to(device)
    if args.visual_pretrained_checkpoint:
        load_visual_pretrained(model, args.visual_pretrained_checkpoint)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters. Check freeze/train flags.")
    print("Training parameters:")
    for name in model.trainable_parameter_names():
        print(f"  {name}")

    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_loss = math.inf
    best_val_wer = math.inf
    training_history: list[dict[str, float | int | None]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
        )
        print(f"epoch {epoch}: train_loss={train_loss:.4f}")

        metrics: dict[str, float] = {}
        if val_loader is not None:
            metrics = evaluate(
                model=model,
                loader=val_loader,
                processor=processor,
                device=device,
                args=args,
            )
            metric_text = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            print(f"epoch {epoch}: {metric_text}")

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics.get("val_loss"),
            "wer": metrics.get("wer"),
            "cer": metrics.get("cer"),
        }
        training_history.append(history_row)
        save_training_history(training_history, results_dir)
        save_training_curves(training_history, results_dir)

        metadata = {
            "epoch": epoch,
            "train_loss": train_loss,
            "metrics": metrics,
            "training_history": training_history,
            "training_args": serializable_args(args),
            "data_config": {
                "image_size": args.image_size,
                "max_video_frames": args.max_video_frames,
                "video_grayscale": args.video_grayscale,
                "language": args.language,
                "task": args.task,
                "visual_encoder": args.visual_encoder,
                "visual_only": args.visual_only,
            },
        }
        save_checkpoint(
            args.output_dir / "last.pt",
            model=model,
            model_config=model_config,
            optimizer=optimizer,
            **metadata,
        )

        val_loss = metrics.get("val_loss", train_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                args.output_dir / "best.pt",
                model=model,
                model_config=model_config,
                optimizer=optimizer,
                best_val_loss=best_val_loss,
                **metadata,
            )
            print(f"saved best checkpoint: {args.output_dir / 'best.pt'}")

        val_wer = metrics.get("wer")
        if val_wer is not None and val_wer < best_val_wer:
            best_val_wer = val_wer
            save_checkpoint(
                args.output_dir / "best_wer.pt",
                model=model,
                model_config=model_config,
                optimizer=optimizer,
                best_val_wer=best_val_wer,
                **metadata,
            )
            print(f"saved best WER checkpoint: {args.output_dir / 'best_wer.pt'} (WER: {best_val_wer:.4f})")

    if test_loader is not None:
        def _eval_checkpoint(tag: str, checkpoint_path: Path) -> None:
            checkpoint = load_checkpoint(checkpoint_path, map_location=device)
            checkpoint_config = AVWhisperConfig.from_dict(checkpoint["model_config"])
            eval_model = AVWhisperModel(checkpoint_config).to(device)
            eval_model.load_state_dict(checkpoint["model_state_dict"])
            metrics = evaluate(
                model=eval_model,
                loader=test_loader,
                processor=processor,
                device=device,
                args=args,
            )
            payload = {
                "checkpoint": str(checkpoint_path),
                "metrics": metrics,
                "run_dir": str(run_dir),
            }
            out_path = results_dir / f"test_{tag}.json"
            out_path.write_text(json.dumps(payload, indent=2))
            metric_text = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            print(f"test {tag}: {metric_text}")

        _eval_checkpoint("last", args.output_dir / "last.pt")
        if (args.output_dir / "best.pt").exists():
            _eval_checkpoint("best", args.output_dir / "best.pt")
        if (args.output_dir / "best_wer.pt").exists():
            _eval_checkpoint("best_wer", args.output_dir / "best_wer.pt")


if __name__ == "__main__":
    main()
