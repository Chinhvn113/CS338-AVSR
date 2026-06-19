from __future__ import annotations

import math
import copy
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .hf_cache import configure_huggingface_cache

configure_huggingface_cache()

from transformers import WhisperForConditionalGeneration
from transformers.models.whisper.modeling_whisper import WhisperAttention
from transformers.modeling_outputs import BaseModelOutput


FLAMINGO_TRAINABLE_NAMES = (
    "gated_x_attn",
    "gated_x_attn_ln",
    "gated_ff",
    "gated_ff_ln",
    "attn_gate",
    "ff_gate",
)


@dataclass
class AVWhisperConfig:
    whisper_model_name: str = "openai/whisper-small"
    fusion_type: str = "flamingo"
    visual_encoder_type: str = "resnet3d"
    video_channels: int = 1
    visual_hidden_size: int = 512
    avhubert_checkpoint: str | None = None
    avhubert_user_dir: str | None = None
    avhubert_output_size: int = 1024
    avhubert_finetuned: bool = False
    video_dropout: float = 0.1
    freeze_whisper: bool = True
    freeze_whisper_encoder: bool = False
    train_visual_encoder: bool = True
    visual_only: bool = False
    max_target_positions: int = 448

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AVWhisperConfig":
        known_keys = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in payload.items() if key in known_keys})


class FusionType(str, Enum):
    CONCAT = "concat"
    FLAMINGO = "flamingo"


class VisualEncoderType(str, Enum):
    CNN2D = "cnn2d"
    RESNET3D = "resnet3d"
    R3D18 = "r3d18"
    AVHUBERT = "avhubert"


class FrameVisualEncoder(nn.Module):
    """A compact per-frame CNN that emits one visual token per sampled frame."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_size: int,
        hidden_size: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"Expected video shaped (B, T, C, H, W), got {video.shape}")

        batch_size, frames, channels, height, width = video.shape
        encoded = self.features(video.reshape(batch_size * frames, channels, height, width))
        projected = self.projection(encoded)
        return projected.reshape(batch_size, frames, -1)


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.PReLU(num_parameters=planes)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.PReLU(num_parameters=planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu2(x + residual)


class ResNet2DTrunk(nn.Module):
    """ResNet-18 trunk used by the Whisper-Flamingo lip encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers: list[nn.Module] = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers.extend(BasicBlock(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.flatten(start_dim=1)


class ResNet3DVisualEncoder(nn.Module):
    """Whisper-Flamingo-style visual encoder for lip-frame sequences.

    This mirrors the upstream ``ResEncoder`` shape contract but does not require
    Fairseq or an AV-HuBERT checkpoint. Input is ``(B, T, C, H, W)`` and output
    is ``(B, T, 512)``.
    """

    output_size = 512

    def __init__(
        self,
        *,
        input_channels: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        frontend_channels = 64
        self.frontend3d = nn.Sequential(
            nn.Conv3d(
                1,
                frontend_channels,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(frontend_channels),
            nn.PReLU(num_parameters=frontend_channels),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.trunk = ResNet2DTrunk()
        self.dropout = nn.Dropout(dropout)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"Expected video shaped (B, T, C, H, W), got {video.shape}")

        if video.shape[2] != 1:
            red, green, blue = video[:, :, 0:1], video[:, :, 1:2], video[:, :, 2:3]
            video = 0.2989 * red + 0.5870 * green + 0.1140 * blue

        video = video.permute(0, 2, 1, 3, 4).contiguous()
        batch_size = video.shape[0]
        video = self.frontend3d(video)
        frames = video.shape[2]
        video = video.transpose(1, 2).contiguous()
        video = video.reshape(batch_size * frames, video.shape[2], video.shape[3], video.shape[4])
        video = self.trunk(video)
        video = self.dropout(video)
        return video.reshape(batch_size, frames, -1)


class R3D18VisualEncoder(nn.Module):
    """Torchvision r3d_18 visual encoder with pretrained weights."""

    def __init__(self, *, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models import video as video_models
        except Exception as exc:  # noqa: BLE001 - surface missing torchvision details.
            raise ImportError("torchvision is required for the r3d50 visual encoder.") from exc

        try:
            weights_enum = getattr(video_models, "R3D_18_Weights", None)
            if weights_enum is not None:
                weights = weights_enum.DEFAULT if pretrained else None
                model = video_models.r3d_18(weights=weights)
            else:
                model = video_models.r3d_18(pretrained=pretrained)
        except TypeError:
            model = video_models.r3d_18(pretrained=pretrained)
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    "torchvision does not provide pretrained r3d_18 weights in this version. "
                    "Upgrade torchvision or disable pretrained weights."
                ) from exc
            raise

        self.stem = model.stem
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.output_size = model.fc.in_features

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"Expected video shaped (B, T, C, H, W), got {video.shape}")

        if video.shape[2] == 1:
            video = video.repeat(1, 1, 3, 1, 1)
        elif video.shape[2] != 3:
            raise ValueError("r3d18 expects 1 or 3 input channels")

        video = video.permute(0, 2, 1, 3, 4).contiguous()
        x = self.stem(video)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.mean(dim=(3, 4))
        return x.transpose(1, 2).contiguous()


class AVHubertVisualEncoder(nn.Module):
    """Official Fairseq AV-HuBERT visual encoder wrapper.

    This follows the loading/inference pattern used by the local
    ``whisper-flamingo`` reference code, but keeps Fairseq optional until this
    backend is selected.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str,
        user_dir: str | None = None,
        output_size: int = 1024,
        finetuned: bool = False,
    ) -> None:
        super().__init__()
        if not checkpoint_path:
            raise ValueError("--avhubert-checkpoint is required when --visual-encoder avhubert")

        try:
            from argparse import Namespace

            from fairseq import checkpoint_utils, utils
        except ImportError as exc:
            raise ImportError(
                "AV-HuBERT backend requires fairseq and the AV-HuBERT user module. "
                "Install the AV-HuBERT/Fairseq environment, then pass "
                "--avhubert-checkpoint and --avhubert-user-dir."
            ) from exc

        if user_dir:
            utils.import_user_module(Namespace(user_dir=str(user_dir)))

        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(checkpoint_path)])
        if not models:
            raise ValueError(f"No AV-HuBERT model loaded from {checkpoint_path}")

        self.output_size = output_size
        self.finetuned = finetuned
        self.model = models[0].encoder if finetuned and hasattr(models[0], "encoder") else models[0]

    def _to_avhubert_video(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"Expected video shaped (B, T, C, H, W), got {video.shape}")

        if video.shape[2] != 1:
            red, green, blue = video[:, :, 0:1], video[:, :, 1:2], video[:, :, 2:3]
            video = 0.2989 * red + 0.5870 * green + 0.1140 * blue
        return video.permute(0, 2, 1, 3, 4).contiguous()

    @staticmethod
    def _extract_features(outputs) -> torch.Tensor:
        if isinstance(outputs, dict):
            if "x" in outputs:
                return outputs["x"]
            if "encoder_out" in outputs:
                encoder_out = outputs["encoder_out"]
                if isinstance(encoder_out, (list, tuple)):
                    encoder_out = encoder_out[0]
                if encoder_out.ndim == 3:
                    return encoder_out.transpose(0, 1).contiguous()
                return encoder_out
            if "features" in outputs:
                return outputs["features"]

        if isinstance(outputs, (list, tuple)):
            return outputs[0]

        return outputs

    def forward(
        self,
        video: torch.Tensor,
        video_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source_video = self._to_avhubert_video(video)
        padding_mask = None
        if video_attention_mask is not None:
            padding_mask = ~video_attention_mask.to(device=video.device, dtype=torch.bool)

        source = {"video": source_video, "audio": None}
        if self.finetuned:
            outputs = self.model(source=source, padding_mask=padding_mask)
        else:
            try:
                outputs = self.model(
                    source=source,
                    padding_mask=padding_mask,
                    mask=False,
                    features_only=True,
                )
            except TypeError:
                outputs = self.model(source=source, padding_mask=padding_mask)

        features = self._extract_features(outputs)
        if features.ndim != 3:
            raise ValueError(f"Expected AV-HuBERT features shaped (B, T, D), got {features.shape}")
        if features.shape[-1] != self.output_size:
            raise ValueError(
                f"AV-HuBERT produced D={features.shape[-1]}, but model was configured "
                f"with avhubert_output_size={self.output_size}. Set --avhubert-output-dim "
                "to match the checkpoint."
            )
        return features


class GatedCrossAttentionDecoderLayer(nn.Module):
    """Wrap a Whisper decoder layer with Flamingo-style gated visual adapters."""

    def __init__(self, base_layer: nn.Module, *, d_model: int, num_heads: int, dropout: float, whisper_config=None) -> None:
        super().__init__()
        self.base_layer = base_layer
        layer_idx = getattr(base_layer.self_attn, "layer_idx", None)
        if whisper_config is None:
            whisper_config = getattr(base_layer.self_attn, "config", None)
        self.gated_x_attn_ln = nn.LayerNorm(d_model)
        self.gated_x_attn = WhisperAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            is_decoder=True,
            layer_idx=layer_idx,
            config=whisper_config,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))
        self.gated_ff_ln = nn.LayerNorm(d_model)
        self.gated_ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))
        self.dropout = dropout
        self.visual_hidden_states: torch.Tensor | None = None
        self.visual_attention_mask: torch.Tensor | None = None

    def set_visual_context(
        self,
        visual_hidden_states: torch.Tensor | None,
        visual_attention_mask: torch.Tensor | None,
    ) -> None:
        self.visual_hidden_states = visual_hidden_states
        self.visual_attention_mask = visual_attention_mask

    def _visual_mask(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if self.visual_attention_mask is None:
            return None

        mask = self.visual_attention_mask.to(device=hidden_states.device)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask.to(dtype=hidden_states.dtype)) * torch.finfo(hidden_states.dtype).min
        return mask.expand(-1, 1, hidden_states.shape[1], -1)

    def _apply_gated_visual_attention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.visual_hidden_states is None:
            return hidden_states

        residual = hidden_states
        attn_outputs = self.gated_x_attn(
            hidden_states=self.gated_x_attn_ln(hidden_states),
            key_value_states=self.visual_hidden_states,
            attention_mask=self._visual_mask(hidden_states),
            output_attentions=False,
        )
        attn_output = attn_outputs[0]
        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        hidden_states = residual + attn_output * self.attn_gate.tanh()

        residual = hidden_states
        ff_output = self.gated_ff(self.gated_ff_ln(hidden_states))
        ff_output = F.dropout(ff_output, p=self.dropout, training=self.training)
        return residual + ff_output * self.ff_gate.tanh()

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        hidden_states = self._apply_gated_visual_attention(hidden_states)
        return self.base_layer(hidden_states, *args, **kwargs)


class AVWhisperModel(nn.Module):
    """Clip-level AVSR model with concat or Whisper-Flamingo fusion."""

    def __init__(self, config: AVWhisperConfig) -> None:
        super().__init__()
        self.config = config
        self.whisper = WhisperForConditionalGeneration.from_pretrained(config.whisper_model_name)

        # Check if we need to resize decoder position embeddings
        if hasattr(config, "max_target_positions") and config.max_target_positions is not None:
            old_max = self.whisper.config.max_target_positions
            new_max = config.max_target_positions
            if new_max != old_max:
                print(f"[info] Resizing Whisper decoder max_target_positions from {old_max} to {new_max}", flush=True)
                old_embed = self.whisper.model.decoder.embed_positions
                from transformers.models.whisper.modeling_whisper import WhisperPositionalEmbedding
                new_embed = WhisperPositionalEmbedding(new_max, old_embed.embedding_dim)
                with torch.no_grad():
                    copy_len = min(old_max, new_max)
                    new_embed.weight[:copy_len].copy_(old_embed.weight[:copy_len])
                    if new_max > old_max:
                        new_embed.weight[copy_len:].normal_(mean=0, std=old_embed.embedding_dim ** -0.5)
                self.whisper.model.decoder.embed_positions = new_embed
                self.whisper.config.max_target_positions = new_max
                self.whisper.model.decoder.max_target_positions = new_max
                self.whisper.max_target_positions = new_max
                if hasattr(self.whisper, "generation_config") and self.whisper.generation_config is not None:
                    if self.whisper.generation_config.max_length == old_max:
                        self.whisper.generation_config.max_length = new_max

        d_model = self.whisper.config.d_model
        self.fusion_type = FusionType(config.fusion_type)
        self.visual_encoder_type = VisualEncoderType(config.visual_encoder_type)
        self.visual_encoder, visual_output_size = self._build_visual_encoder(config)
        self.video_projection = nn.Linear(visual_output_size, d_model)
        self.video_projection_scalar = nn.Parameter(torch.tensor(1.0))
        self.fusion_norm = nn.LayerNorm(d_model)

        if self.fusion_type == FusionType.FLAMINGO:
            self._install_flamingo_decoder_layers()

        if config.freeze_whisper:
            self._freeze_base_whisper()
        elif config.freeze_whisper_encoder:
            for parameter in self.whisper.model.encoder.parameters():
                parameter.requires_grad = False

        if not config.train_visual_encoder:
            for parameter in self.visual_encoder.parameters():
                parameter.requires_grad = False

    def _build_visual_encoder(self, config: AVWhisperConfig) -> tuple[nn.Module, int]:
        visual_encoder_type = VisualEncoderType(config.visual_encoder_type)
        if visual_encoder_type == VisualEncoderType.AVHUBERT:
            return (
                AVHubertVisualEncoder(
                    checkpoint_path=config.avhubert_checkpoint or "",
                    user_dir=config.avhubert_user_dir,
                    output_size=config.avhubert_output_size,
                    finetuned=config.avhubert_finetuned,
                ),
                config.avhubert_output_size,
            )

        if visual_encoder_type == VisualEncoderType.RESNET3D:
            return (
                ResNet3DVisualEncoder(
                    input_channels=config.video_channels,
                    dropout=config.video_dropout,
                ),
                ResNet3DVisualEncoder.output_size,
            )

        if visual_encoder_type == VisualEncoderType.R3D18:
            encoder = R3D18VisualEncoder(pretrained=True)
            return encoder, encoder.output_size

        return (
            FrameVisualEncoder(
                input_channels=config.video_channels,
                output_size=config.visual_hidden_size,
                hidden_size=config.visual_hidden_size,
                dropout=config.video_dropout,
            ),
            config.visual_hidden_size,
        )

    def _install_flamingo_decoder_layers(self) -> None:
        decoder = self.whisper.model.decoder
        decoder.layers = nn.ModuleList(
            GatedCrossAttentionDecoderLayer(
                layer,
                d_model=self.whisper.config.d_model,
                num_heads=self.whisper.config.decoder_attention_heads,
                dropout=self.whisper.config.dropout,
                whisper_config=self.whisper.config,
            )
            for layer in decoder.layers
        )

    def _freeze_base_whisper(self) -> None:
        for name, parameter in self.whisper.named_parameters():
            parameter.requires_grad = any(token in name for token in FLAMINGO_TRAINABLE_NAMES)

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def encode_video(
        self,
        video: torch.Tensor,
        video_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.visual_encoder_type == VisualEncoderType.AVHUBERT:
            visual_hidden = self.visual_encoder(video, video_attention_mask)
        else:
            visual_hidden = self.visual_encoder(video)
        visual_hidden = self.video_projection_scalar * self.video_projection(visual_hidden)
        visual_hidden = self.fusion_norm(visual_hidden)

        if video_attention_mask is None:
            return visual_hidden, None

        video_attention_mask = video_attention_mask.to(device=visual_hidden.device, dtype=torch.long)
        if video_attention_mask.shape[1] != visual_hidden.shape[1]:
            video_attention_mask = F.interpolate(
                video_attention_mask[:, None].float(),
                size=visual_hidden.shape[1],
                mode="nearest",
            ).squeeze(1).long()
        return visual_hidden, video_attention_mask

    def _set_flamingo_context(
        self,
        visual_hidden: torch.Tensor | None,
        visual_attention_mask: torch.Tensor | None,
    ) -> None:
        if self.fusion_type != FusionType.FLAMINGO:
            return

        for layer in self.whisper.model.decoder.layers:
            if isinstance(layer, GatedCrossAttentionDecoderLayer):
                layer.set_visual_context(visual_hidden, visual_attention_mask)

    def clear_flamingo_context(self) -> None:
        self._set_flamingo_context(None, None)

    def encode(
        self,
        *,
        video: torch.Tensor | None,
        audio_features: torch.Tensor | None,
        video_attention_mask: torch.Tensor | None = None,
    ) -> tuple[BaseModelOutput, torch.Tensor]:
        if audio_features is None and video is None:
            raise ValueError("Provide video and/or audio features")

        if self.config.visual_only or audio_features is None:
            if video is None:
                raise ValueError("visual_only requires video input")
            video_hidden, projected_video_attention_mask = self.encode_video(video, video_attention_mask)
            if projected_video_attention_mask is None:
                projected_video_attention_mask = torch.ones(
                    video_hidden.shape[:2],
                    dtype=torch.long,
                    device=video_hidden.device,
                )
            return BaseModelOutput(last_hidden_state=video_hidden), projected_video_attention_mask

        if video is None:
            audio_outputs = self.whisper.model.encoder(
                input_features=audio_features,
                return_dict=True,
            )
            audio_hidden = audio_outputs.last_hidden_state
            audio_attention_mask = torch.ones(
                audio_hidden.shape[:2],
                dtype=torch.long,
                device=audio_hidden.device,
            )
            return BaseModelOutput(last_hidden_state=audio_hidden), audio_attention_mask

        audio_outputs = self.whisper.model.encoder(
            input_features=audio_features,
            return_dict=True,
        )
        audio_hidden = audio_outputs.last_hidden_state

        video_hidden, projected_video_attention_mask = self.encode_video(video, video_attention_mask)

        if self.fusion_type == FusionType.FLAMINGO:
            self._set_flamingo_context(video_hidden, projected_video_attention_mask)
            audio_attention_mask = torch.ones(
                audio_hidden.shape[:2],
                dtype=torch.long,
                device=audio_hidden.device,
            )
            return BaseModelOutput(last_hidden_state=audio_hidden), audio_attention_mask

        fused_hidden = torch.cat([video_hidden, audio_hidden], dim=1)

        if projected_video_attention_mask is None:
            projected_video_attention_mask = torch.ones(
                video_hidden.shape[:2],
                dtype=torch.long,
                device=video_hidden.device,
            )

        audio_attention_mask = torch.ones(
            audio_hidden.shape[:2],
            dtype=torch.long,
            device=audio_hidden.device,
        )
        fused_attention_mask = torch.cat([projected_video_attention_mask, audio_attention_mask], dim=1)
        return BaseModelOutput(last_hidden_state=fused_hidden), fused_attention_mask

    def forward(
        self,
        *,
        video: torch.Tensor | None,
        audio_features: torch.Tensor | None,
        labels: torch.Tensor | None = None,
        video_attention_mask: torch.Tensor | None = None,
    ):
        encoder_outputs, attention_mask = self.encode(
            video=video,
            audio_features=audio_features,
            video_attention_mask=video_attention_mask,
        )
        try:
            return self.whisper(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
        finally:
            self.clear_flamingo_context()

    @torch.no_grad()
    def generate(
        self,
        *,
        video: torch.Tensor | None,
        audio_features: torch.Tensor | None,
        video_attention_mask: torch.Tensor | None = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        encoder_outputs, attention_mask = self.encode(
            video=video,
            audio_features=audio_features,
            video_attention_mask=video_attention_mask,
        )
        try:
            if "max_new_tokens" in generate_kwargs and "generation_config" not in generate_kwargs:
                generation_config = copy.deepcopy(self.whisper.generation_config)
                generation_config.max_length = None
                generate_kwargs["generation_config"] = generation_config

            return self.whisper.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                **generate_kwargs,
            )
        finally:
            self.clear_flamingo_context()
