#!/usr/bin/env python3
import argparse
import random
import shutil
import sys
from pathlib import Path
import cv2
import torch
import torchaudio
from tqdm import tqdm

# Add project root to python path to import cs338 modules if needed
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cs338.data import discover_samples
from cs338.preprocessing import load_audio_waveform


class VideoAugmentor:
    """Applies consistent random spatial and color augmentations to all frames of a video."""

    def __init__(self):
        pass

    def sample_params(self, width: int, height: int) -> dict:
        # 50% chance to flip horizontally
        flip = random.random() < 0.5
        
        # Brightness & Contrast adjustment
        contrast = random.uniform(0.85, 1.15)
        brightness = random.uniform(-15.0, 15.0)
        
        # Minor rotation: angle in degrees
        angle = random.uniform(-8.0, 8.0)
        
        # Crop scaling: crop between 90% and 100% of the image size
        crop_scale = random.uniform(0.9, 1.0)
        new_w = int(width * crop_scale)
        new_h = int(height * crop_scale)
        x = random.randint(0, width - new_w)
        y = random.randint(0, height - new_h)
        
        return {
            "flip": flip,
            "contrast": contrast,
            "brightness": brightness,
            "angle": angle,
            "crop": (x, y, new_w, new_h)
        }

    def apply(self, frame, params: dict):
        h, w = frame.shape[:2]
        
        # 1. Horizontal Flip
        if params["flip"]:
            frame = cv2.flip(frame, 1)
            
        # 2. Brightness & Contrast
        frame = cv2.convertScaleAbs(frame, alpha=params["contrast"], beta=params["brightness"])
        
        # 3. Rotation
        if abs(params["angle"]) > 0.1:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), params["angle"], 1.0)
            frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            
        # 4. Crop & Resize back
        x, y, nw, nh = params["crop"]
        cropped = frame[y:y+nh, x:x+nw]
        frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return frame


class AudioAugmentor:
    """Applies random noise (Gaussian, Pink, Brown, Hum, Mixed) with target SNR, and amplitude scaling."""

    def __init__(self):
        pass

    def generate_pink_noise(self, num_samples: int) -> torch.Tensor:
        white_noise = torch.randn(num_samples)
        fft = torch.fft.rfft(white_noise)
        frequencies = torch.fft.rfftfreq(num_samples)
        frequencies[0] = frequencies[1]  # avoid division by zero
        fft = fft / torch.sqrt(frequencies)
        pink = torch.fft.irfft(fft, n=num_samples)
        pink = pink / (pink.std() + 1e-8)
        return pink

    def generate_brown_noise(self, num_samples: int) -> torch.Tensor:
        white_noise = torch.randn(num_samples)
        fft = torch.fft.rfft(white_noise)
        frequencies = torch.fft.rfftfreq(num_samples)
        frequencies[0] = frequencies[1]  # avoid division by zero
        fft = fft / frequencies
        brown = torch.fft.irfft(fft, n=num_samples)
        brown = brown / (brown.std() + 1e-8)
        return brown

    def generate_hum(self, num_samples: int, sample_rate: int = 16000) -> torch.Tensor:
        t = torch.arange(num_samples) / sample_rate
        freq = random.choice([50, 60])
        # Main hum
        hum = torch.sin(2 * 3.14159265 * freq * t)
        # Add some harmonics
        hum += 0.5 * torch.sin(2 * 3.14159265 * freq * 2 * t)
        hum += 0.25 * torch.sin(2 * 3.14159265 * freq * 3 * t)
        hum = hum / (hum.std() + 1e-8)
        return hum

    def apply(self, waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        channels, seq_len = waveform.shape
        
        # Select noise type: Gaussian, Pink, Brown, Hum, or Mixed
        noise_type = random.choice(["gaussian", "pink", "brown", "hum", "mixed"])
        
        # Generate raw noise
        if noise_type == "gaussian":
            noise = torch.randn_like(waveform)
        elif noise_type == "pink":
            noise = torch.stack([self.generate_pink_noise(seq_len) for _ in range(channels)])
        elif noise_type == "brown":
            noise = torch.stack([self.generate_brown_noise(seq_len) for _ in range(channels)])
        elif noise_type == "hum":
            noise = torch.stack([self.generate_hum(seq_len, sample_rate) for _ in range(channels)])
        else:  # mixed
            noise_parts = []
            for _ in range(channels):
                p = self.generate_pink_noise(seq_len) * random.uniform(0.3, 0.7)
                b = self.generate_brown_noise(seq_len) * random.uniform(0.3, 0.7)
                h = self.generate_hum(seq_len, sample_rate) * random.uniform(0.1, 0.4)
                g = torch.randn(seq_len) * random.uniform(0.1, 0.3)
                noise_parts.append(p + b + h + g)
            noise = torch.stack(noise_parts)

        # Choose SNR in dB (lower = noisier, higher = cleaner)
        # Choosing 3.0 to 15.0 dB for more prominent audio noise
        snr_db = random.uniform(3.0, 15.0)
        
        # Compute signal power and noise power
        sig_power = torch.mean(waveform ** 2)
        noise_power = torch.mean(noise ** 2)
        
        if sig_power > 1e-8 and noise_power > 1e-8:
            target_noise_power = sig_power / (10 ** (snr_db / 10.0))
            scale = torch.sqrt(target_noise_power / noise_power)
            waveform = waveform + scale * noise
        
        # 2. Volume Scale: random factor
        volume_factor = random.uniform(0.8, 1.2)
        waveform = waveform * volume_factor
        
        # Clamp to avoid clipping/distortion
        return torch.clamp(waveform, -1.0, 1.0)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Augment AVSR dataset with random audio/video noise.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to the dataset directory containing wav, csv, mp4 files.",
    )
    parser.add_argument(
        "--augment",
        type=str2bool,
        default=True,
        help="Whether to perform augmentation (True/False).",
    )
    parser.add_argument(
        "--augment-ratio",
        type=float,
        default=0.5,
        help="Fraction of samples in the dataset to augment (0.0 to 1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=338,
        help="Random seed for sample selection and augmentation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing augmented files if present.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.dataset_dir.exists():
        print(f"Error: Dataset directory {args.dataset_dir} does not exist.")
        sys.exit(1)

    if not args.augment:
        print("Augment is set to False. Exiting without augmenting.")
        sys.exit(0)

    print(f"Scanning dataset directory: {args.dataset_dir}")
    # Discover triplets/samples using project's discover_samples
    samples = discover_samples(args.dataset_dir, allow_video_audio=True)
    
    if not samples:
        print("No valid samples (triplets/pairs of wav/mp4/csv) found in the directory.")
        sys.exit(0)

    print(f"Found {len(samples)} total samples.")
    
    # Exclude already augmented samples to avoid double augmentation
    original_samples = [s for s in samples if not s.sample_id.endswith("_augmented")]
    print(f"Found {len(original_samples)} original (non-augmented) samples.")

    if not original_samples:
        print("No original samples available to augment.")
        sys.exit(0)

    num_to_augment = int(len(original_samples) * args.augment_ratio)
    print(f"Targeting {num_to_augment} samples for augmentation (ratio: {args.augment_ratio}).")

    selected_samples = random.sample(original_samples, num_to_augment)

    video_augmentor = VideoAugmentor()
    audio_augmentor = AudioAugmentor()

    augmented_count = 0
    for sample in tqdm(selected_samples, desc="Augmenting samples"):
        # Target paths with _augmented postfix
        aug_video_path = sample.video_path.parent / f"{sample.video_path.stem}_augmented.mp4"
        aug_audio_path = sample.audio_path.parent / f"{sample.audio_path.stem}_augmented.wav"
        aug_csv_path = sample.csv_path.parent / f"{sample.csv_path.stem}_augmented.csv"

        if not args.overwrite and aug_video_path.exists() and aug_audio_path.exists() and aug_csv_path.exists():
            continue

        try:
            # Bias augmentation significantly towards audio (more noise on audio than visual)
            # 0: Augment both video and audio
            # 1: Augment video only (copy original audio)
            # 2: Augment audio only (copy original video)
            aug_mode = random.choices([0, 1, 2], weights=[0.25, 0.05, 0.70], k=1)[0]

            # 1. Video Processing (Augment or Copy)
            if aug_mode in (0, 1):
                cap = cv2.VideoCapture(str(sample.video_path))
                if not cap.isOpened():
                    print(f"\n[warn] Could not open video: {sample.video_path}")
                    continue
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                
                out = cv2.VideoWriter(str(aug_video_path), fourcc, fps, (width, height))
                if not out.isOpened():
                    print(f"\n[warn] Could not open video writer for: {aug_video_path}")
                    cap.release()
                    continue

                video_params = video_augmentor.sample_params(width, height)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    aug_frame = video_augmentor.apply(frame, video_params)
                    out.write(aug_frame)
                
                cap.release()
                out.release()
            else:
                shutil.copy2(sample.video_path, aug_video_path)

            # 2. Audio Processing (Augment or Copy/Extract)
            if aug_mode in (0, 2):
                if sample.audio_path.suffix.lower() == ".wav" and sample.audio_path.exists() and sample.audio_path != sample.video_path:
                    # Normal WAV file
                    waveform, sample_rate = torchaudio.load(str(sample.audio_path))
                    aug_waveform = audio_augmentor.apply(waveform, sample_rate)
                    torchaudio.save(str(aug_audio_path), aug_waveform, sample_rate)
                else:
                    # Fell back to video file container for audio, extract at 16kHz
                    sample_rate = 16000
                    waveform = load_audio_waveform(sample.audio_path, target_sample_rate=sample_rate)
                    waveform = waveform.unsqueeze(0)
                    aug_waveform = audio_augmentor.apply(waveform, sample_rate)
                    torchaudio.save(str(aug_audio_path), aug_waveform, sample_rate)
            else:
                if sample.audio_path.suffix.lower() == ".wav" and sample.audio_path.exists() and sample.audio_path != sample.video_path:
                    shutil.copy2(sample.audio_path, aug_audio_path)
                else:
                    sample_rate = 16000
                    waveform = load_audio_waveform(sample.audio_path, target_sample_rate=sample_rate)
                    waveform = waveform.unsqueeze(0)
                    torchaudio.save(str(aug_audio_path), waveform, sample_rate)

            # 3. Copy CSV
            shutil.copy2(sample.csv_path, aug_csv_path)
            augmented_count += 1

        except Exception as e:
            print(f"\n[error] Failed to augment sample {sample.sample_id}: {e}")
            # Clean up partial files if any
            for path in (aug_video_path, aug_audio_path, aug_csv_path):
                if path.exists():
                    try:
                        path.unlink()
                    except OSError:
                        pass

    print(f"Successfully generated {augmented_count} augmented samples.")


if __name__ == "__main__":
    main()
