from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cs338 import data as data_module  # noqa: E402
from cs338 import train as train_module  # noqa: E402


class DummyTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, return_tensors: str | None = None):
        token_ids = [index + 1 for index, _ in enumerate(text.split())]
        if not token_ids:
            token_ids = [1]
        return SimpleNamespace(input_ids=torch.tensor([token_ids], dtype=torch.long))

    def pad(self, inputs, padding: bool = True, return_tensors: str | None = None):
        sequences = inputs["input_ids"]
        max_length = max(len(sequence) for sequence in sequences)
        padded = [
            sequence + [self.pad_token_id] * (max_length - len(sequence))
            for sequence in sequences
        ]
        return SimpleNamespace(input_ids=torch.tensor(padded, dtype=torch.long))


class DummyProcessor:
    def __init__(self) -> None:
        self.tokenizer = DummyTokenizer()

    def batch_decode(self, generated_ids, skip_special_tokens: bool = True):
        return ["hello world" for _ in range(generated_ids.shape[0])]


class DummyAVWhisperModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.tensor(1.0))

    def trainable_parameter_names(self) -> list[str]:
        return ["weight"]

    def forward(self, *, video, audio_features, labels=None, video_attention_mask=None):
        loss = self.weight.square()
        if labels is not None:
            loss = loss + labels[labels != -100].float().mean() * 0.0
        return SimpleNamespace(loss=loss)

    @torch.no_grad()
    def generate(self, *, video, audio_features, video_attention_mask=None, **kwargs):
        return torch.ones(video.shape[0], 2, dtype=torch.long, device=video.device)


def write_mock_sample(split_dir: Path, sample_id: str) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / f"{sample_id}.mp4").write_bytes(b"mock video placeholder")
    (split_dir / f"{sample_id}.wav").write_bytes(b"mock audio placeholder")
    (split_dir / f"{sample_id}.csv").write_text(
        "word,start_time,end_time\n"
        "hello,0.00,0.20\n"
        "world,0.20,0.45\n",
        encoding="utf-8",
    )


def run_training_smoke_test() -> None:
    with tempfile.TemporaryDirectory(prefix="cs338-train-smoke-") as temp_dir:
        temp_path = Path(temp_dir)
        data_root = temp_path / "data"
        output_dir = temp_path / "checkpoints"

        write_mock_sample(data_root / "train", "sample_001")
        write_mock_sample(data_root / "train", "sample_002")
        write_mock_sample(data_root / "val", "sample_101")

        argv = [
            "scripts/train.py",
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--eval-max-batches",
            "1",
            "--max-new-tokens",
            "4",
        ]

        with patch.object(sys, "argv", argv), \
            patch.object(train_module, "load_processor", return_value=DummyProcessor()), \
            patch.object(train_module, "AVWhisperModel", DummyAVWhisperModel), \
            patch.object(data_module, "load_video_clip", return_value=torch.zeros(4, 1, 8, 8)), \
            patch.object(data_module, "load_audio_features", return_value=torch.zeros(80, 3000)):
            train_module.main()

        last_checkpoint = output_dir / "last.pt"
        best_checkpoint = output_dir / "best.pt"
        assert last_checkpoint.exists(), f"Missing checkpoint: {last_checkpoint}"
        assert best_checkpoint.exists(), f"Missing checkpoint: {best_checkpoint}"

        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        assert checkpoint["epoch"] == 1
        assert "model_state_dict" in checkpoint
        assert checkpoint["data_config"]["image_size"] == 88


if __name__ == "__main__":
    run_training_smoke_test()
    print("training smoke test passed")
