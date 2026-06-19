# Additional Requirements: Dataset Format and Initial Training Pipeline

Add the following requirements to the architecture specification.

---

# Dataset Format

Each video sample contains:

```text
sample_001.mp4
sample_001.wav
sample_001.csv
```

CSV format:

```csv
word,start_time,end_time
hello,0.32,0.67
world,0.75,1.12
```

Where:

* `word` = spoken word
* `start_time` = word start timestamp in seconds
* `end_time` = word end timestamp in seconds

---

# Initial Training Strategy (V1)

The first implementation should focus on a simple and stable pipeline.

Do NOT implement frame-level alignment, contrastive learning, CTC, or auxiliary losses in V1.

Do NOT use word timestamps during model training.

The timestamps should only be loaded and preserved by the dataset loader for future experiments.

The first version should train at the sentence/clip level.

---

# V1 Pipeline

Each training sample consists of:

```python
{
    "video": video_clip,
    "audio": audio_clip,
    "transcript": full_transcript,
    "word_timestamps": csv_annotations
}
```

Example:

```text
Video Duration: 4.8 seconds

Transcript:
"hello world how are you"

CSV:
hello,0.32,0.67
world,0.75,1.12
how,1.20,1.40
are,1.45,1.60
you,1.65,1.95
```

The model should receive the entire clip:

```text
Video -> Visual Encoder
Audio -> Whisper Encoder
Fusion -> Whisper Decoder
Transcript -> Target Text
```

No segmentation should be performed.

No word-level cropping should be performed.

No forced alignment should be performed.

---

# Transcript Construction

The dataset loader must automatically reconstruct the transcript from the CSV file.

Example:

```csv
word,start_time,end_time
hello,0.32,0.67
world,0.75,1.12
```

becomes:

```text
hello world
```

Implementation:

```python
transcript = " ".join(df["word"].tolist())
```

The reconstructed transcript becomes the decoder target text.

---

# Dataset Loader

Implement:

```python
class AVSRDataset(Dataset):
```

Expected output:

```python
{
    "video": Tensor,
    "audio": Tensor,
    "labels": Tensor,
    "transcript": str,
    "timestamps": List[Dict]
}
```

Where:

```python
timestamps = [
    {
        "word": "hello",
        "start_time": 0.32,
        "end_time": 0.67,
    }
]
```

The timestamps are stored but not used by the model in V1.

---

# Audio Processing

Use Whisper processor:

```python
WhisperProcessor.from_pretrained(...)
```

Convert waveform into:

```python
input_features
```

for the Whisper encoder.

---

# Video Processing

Load the entire clip.

Recommended preprocessing:

```text
Decode video
↓
Extract frames
↓
Resize
↓
Normalize
↓
Visual Encoder
```

Target output:

```python
(B, T_video, D_visual)
```

---

# Model Input

Forward signature:

```python
outputs = model(
    video=video_tensor,
    audio_features=audio_features,
    labels=labels,
)
```

The model should not receive timestamps during training.

---

# Training Objective

Use only Whisper seq2seq loss.

```python
loss = outputs.loss
```

Do not add:

* contrastive loss
* alignment loss
* CTC loss
* auxiliary losses

for the first implementation.

---

# Evaluation

Compute:

* Validation loss
* WER
* CER

using the full transcript reconstructed from the CSV.

---

# Inference

Input:

```bash
python scripts/inference.py \
    --checkpoint checkpoints/best.pt \
    --video sample.mp4
```

Output:

```text
Predicted Transcript:
hello world how are you
```

No CSV file should be required during inference.

---

# Future Extensions (Not V1)

Keep the code modular so future versions can support:

1. Word-level clip training
2. Audio-visual contrastive learning
3. Timestamp-guided attention
4. Forced alignment losses
5. AV-HuBERT style pretraining
6. Whisper-Flamingo style fusion

These features should not be implemented in the initial version.

---

# Whisper-Flamingo Direct Training Path

The codebase also includes a Whisper-Flamingo-style path adapted from the local
`whisper-flamingo` reference implementation.

This path still uses the same custom dataset structure:

```text
sample_001.mp4
sample_001.wav
sample_001.csv
```

The transcript is still reconstructed from CSV words, and word timestamps remain
metadata only.

Model flow:

```text
Video -> 3D ResNet visual encoder -> visual projection
Audio -> Whisper encoder
Visual projection -> gated cross-attention adapters inside Whisper decoder
Whisper decoder -> transcript target
```

Training strategy for this project:

* Do not run an audio-only fine-tuning stage.
* Load a pretrained Whisper checkpoint directly.
* Freeze the base Whisper encoder, decoder, embeddings, and LM head by default.
* Train:
  * visual encoder
  * video projection
  * video projection scalar
  * gated decoder cross-attention layers
  * gated decoder feed-forward adapters
* Use only Whisper seq2seq loss.

Default command:

```bash
python scripts/train.py \
    --data-root data \
    --model-type flamingo \
    --visual-encoder resnet3d \
    --whisper-model openai/whisper-small
```

Use `--train-whisper` only if full Whisper fine-tuning is desired.

Visual encoder backends:

* `resnet3d`: local 3D ResNet visual encoder adapted from Whisper-Flamingo's
  lip-reader encoder shape.
* `cnn2d`: small per-frame CNN baseline.
* `avhubert`: official Fairseq AV-HuBERT checkpoint wrapper.

AV-HuBERT command:

```bash
python scripts/train.py \
    --data-root data \
    --model-type flamingo \
    --visual-encoder avhubert \
    --avhubert-checkpoint /path/to/avhubert.pt \
    --avhubert-user-dir /path/to/av_hubert \
    --avhubert-output-dim 1024
```

If the AV-HuBERT checkpoint is fine-tuned and the encoder is nested under
`model.encoder`, add:

```bash
--avhubert-finetuned
```

The `avhubert` backend requires a working Fairseq + AV-HuBERT user module
environment. It is optional so the local `resnet3d` backend remains usable
without Fairseq.
