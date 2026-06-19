# CS338 AVSR Pipeline

Audio-visual speech recognition pipeline for training and evaluating Whisper-based
models on paired video, audio, and transcript timestamp files.

The project is configured with YAML files under `configs/`. Training and
evaluation scripts load defaults from the code, then override them with the
selected config file.

## Requirements

- Python 3.12 or newer
- CUDA-capable GPU recommended for training/evaluation
- `ffmpeg` for chunked evaluation
- Dataset samples containing same-stem media/transcript files

Each dataset sample should look like:

```text
sample_001.mp4
sample_001.wav
sample_001.csv
```

CSV files should contain word timestamps:

```csv
word,start_time,end_time
hello,0.32,0.67
world,0.75,1.12
```

## Install Dependencies

### Option 1: Using uv

Create a virtual environment and install from the lock/project metadata:

```bash
uv venv
source .venv/bin/activate
uv sync
```

If you only want to install from `requirements.txt`:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Option 2: Using pip

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install the project metadata directly:

```bash
pip install -e .
```

## Train With a Config

Edit one of the training configs in `configs/`, for example:

- `configs/train.yaml`
- `configs/train_aug.yaml`
- `configs/train_p3.yaml`
- `configs/train_pho.yaml`
- `configs/train_pho_p3.yaml`

Run training from the repository root:

```bash
python scripts/train.py --config configs/train.yaml --data-root /path/to/dataset
```

If `--data-root` contains `train/`, `val/`, and optionally `test/`, the script
uses those splits automatically. Otherwise it uses the configured `train_root`,
`val_root`, or `val_ratio`.

Training outputs are written under `checkpoints/` using the config name:

```text
checkpoints/train_run1/
checkpoints/train_run1/last.pt
checkpoints/train_run1/best.pt
checkpoints/train_run1/best_wer.pt
```

Training curves and history are written under `results/`.

You can also put paths directly in the YAML config:

```yaml
train_root: /path/to/dataset/train
val_root: /path/to/dataset/val
output_dir: checkpoints
epochs: 10
batch_size: 2
device: cuda
amp: true
```

Then run:

```bash
python scripts/train.py --config configs/train.yaml
```

## Evaluate With a Config

Edit an evaluation config, for example:

- `configs/eval.yaml`
- `configs/eval_2.yaml`
- `configs/eval_base.yaml`
- `configs/eval_base_pho.yaml`
- `configs/eval_noise.yaml`

Set at least:

```yaml
checkpoint: checkpoints/train_run1/best_wer.pt
test_root: /path/to/dataset/test
mode: av
device: cuda
```

Run evaluation:

```bash
python scripts/eval.py --config configs/eval.yaml
```

You can override the dataset root from the command line:

```bash
python scripts/eval.py --config configs/eval.yaml --data-root /path/to/dataset
```

Supported evaluation modes include:

- `av`: audio + visual evaluation
- `audio-only`: ignore visual input
- `visual-only`: visual-only decoding
- `compare`: compare `audio-only` and/or `av` across configured SNR levels

For noisy comparison, configure fields like:

```yaml
mode: compare
chunk_seconds: 30.0
compare_modes:
  - audio-only
  - av
compare_snr_db:
  - -10
  - 10
all_noise: true
```

Evaluation results are saved under:

```text
results/eval/<checkpoint-path>/
```

For example:

```text
results/eval/checkpoints/train_run1/best_wer/eval_av.json
results/eval/checkpoints/train_run1/best_wer/compare.json
```

## Inference

Run inference on a single video:

```bash
python scripts/inference.py \
  --checkpoint checkpoints/train_run1/best_wer.pt \
  --video /path/to/sample.mp4
```

If `--audio` is not provided, the script looks for a same-stem `.wav` file next
to the video, then falls back to reading audio from the video file when
supported locally.

## Common Workflow

```bash
# install
uv venv
source .venv/bin/activate
uv sync

# train
python scripts/train.py --config configs/train.yaml --data-root /path/to/dataset

# evaluate the selected checkpoint
python scripts/eval.py --config configs/eval.yaml
```
