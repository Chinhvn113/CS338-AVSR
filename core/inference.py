from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .hf_cache import configure_huggingface_cache

configure_huggingface_cache()

from transformers import WhisperProcessor
from huggingface_hub import hf_hub_download

from .checkpoint import load_checkpoint
from .model import AVWhisperConfig, AVWhisperModel
from .preprocessing import load_audio_features, load_inference_audio_features, load_video_clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AVSR inference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--audio", type=Path, help="Optional audio file. Defaults to same-stem .wav.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-video-frames", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--rgb-video", action="store_false", dest="video_grayscale")
    parser.add_argument("--visual-encoder", choices=["resnet3d", "r3d18", "cnn2d", "avhubert"])
    parser.add_argument("--avhubert-checkpoint", type=Path)
    parser.add_argument("--avhubert-user-dir", type=Path)
    parser.add_argument("--avhubert-output-dim", type=int)
    parser.add_argument("--avhubert-finetuned", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--audio-only", action="store_true")
    parser.add_argument("--no-audio-only", action="store_false", dest="audio_only")
    parser.add_argument("--visual-only", action="store_true")
    parser.add_argument("--no-visual-only", action="store_false", dest="visual_only")
    parser.set_defaults(video_grayscale=None, audio_only=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)

    raw_model_config = checkpoint["model_config"]
    model_config = AVWhisperConfig.from_dict(raw_model_config)
    checkpoint_visual_encoder = raw_model_config.get("visual_encoder_type")
    if args.visual_encoder and checkpoint_visual_encoder and args.visual_encoder != checkpoint_visual_encoder:
        raise ValueError(
            f"This checkpoint was trained with visual_encoder={checkpoint_visual_encoder!r}, "
            f"but inference requested {args.visual_encoder!r}. Train a separate checkpoint "
            "for that visual encoder instead of switching encoders at inference time."
        )
    if args.visual_encoder:
        model_config.visual_encoder_type = args.visual_encoder
    if args.avhubert_checkpoint:
        model_config.avhubert_checkpoint = str(args.avhubert_checkpoint)
    elif model_config.visual_encoder_type == "avhubert" and not model_config.avhubert_checkpoint:
        model_config.avhubert_checkpoint = hf_hub_download(
            repo_id="vumichien/AV-HuBERT",
            filename="model.pt",
        )
    if args.avhubert_user_dir:
        model_config.avhubert_user_dir = str(args.avhubert_user_dir)
    if args.avhubert_output_dim:
        model_config.avhubert_output_size = args.avhubert_output_dim
    if args.avhubert_finetuned:
        model_config.avhubert_finetuned = True

    data_config = checkpoint.get("data_config", {})
    processor_kwargs = {}
    if data_config.get("language"):
        processor_kwargs["language"] = data_config["language"]
    if data_config.get("task"):
        processor_kwargs["task"] = data_config["task"]

    processor = WhisperProcessor.from_pretrained(model_config.whisper_model_name, **processor_kwargs)
    model = AVWhisperModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.audio_only and args.visual_only:
        raise ValueError("Choose only one of --audio-only or --visual-only")

    visual_only = args.visual_only or data_config.get("visual_only", False) or model_config.visual_only
    audio_only = args.audio_only
    if audio_only:
        visual_only = False
    if audio_only and visual_only:
        raise ValueError("audio_only and visual_only cannot both be enabled")
    model_config.visual_only = visual_only

    image_size = args.image_size or data_config.get("image_size", 224)
    max_video_frames = args.max_video_frames or data_config.get("max_video_frames", 96)
    video_grayscale = data_config.get("video_grayscale", model_config.video_channels == 1)
    if args.video_grayscale is not None:
        video_grayscale = args.video_grayscale

    video = None
    if not audio_only:
        if args.video is None:
            raise ValueError("--video is required unless --audio-only is set")
        video = load_video_clip(
            args.video,
            image_size=image_size,
            max_frames=max_video_frames,
            grayscale=video_grayscale,
        ).unsqueeze(0)

    audio_features = None
    if not visual_only:
        if args.audio is not None:
            audio_features = load_audio_features(args.audio, processor).unsqueeze(0)
        elif args.video is not None:
            audio_features = load_inference_audio_features(
                args.video,
                processor,
                audio_path=args.audio,
            ).unsqueeze(0)
        else:
            raise ValueError("Provide --audio or --video for audio features")

    if video is not None:
        video = video.to(device)
    if audio_features is not None:
        audio_features = audio_features.to(device)
    video_attention_mask = None
    if video is not None:
        video_attention_mask = torch.ones(video.shape[:2], dtype=torch.long, device=device)

    generated_ids = model.generate(
        video=video,
        audio_features=audio_features,
        video_attention_mask=video_attention_mask,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )
    transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print("Predicted Transcript:")
    print(transcript)


if __name__ == "__main__":
    main()
