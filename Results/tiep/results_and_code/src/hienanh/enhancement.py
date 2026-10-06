"""Official-checkpoint-compatible Zero-DCE models and the project's DIP baseline.

Architecture reference: https://github.com/Li-Chongyi/Zero-DCE
Zero-DCE++ reference: https://github.com/Li-Chongyi/Zero-DCE_extension
The official implementations/checkpoints use CC BY-NC 4.0 for academic research.
"""

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SeparableConv(nn.Module):
    """Keep parameter names compatible with the official Zero-DCE++ checkpoint."""

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            input_channels, input_channels, 3, padding=1, groups=input_channels
        )
        self.point_conv = nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.point_conv(self.depth_conv(image))


class ZeroDCE(nn.Module):
    """Seven-layer DCE-Net, including the official skip order and curve sign."""

    def __init__(self, variant: str = "zerodce", scale_factor: int = 1):
        super().__init__()
        if variant not in {"zerodce", "zerodcepp"}:
            raise ValueError(f"Unknown model variant: {variant}")
        if scale_factor < 1 or (variant == "zerodce" and scale_factor != 1):
            raise ValueError("Only Zero-DCE++ supports a scale factor greater than one.")
        self.variant = variant
        self.scale_factor = scale_factor
        channels = [3, 32, 32, 32, 32, 64, 64, 64]
        outputs = [32, 32, 32, 32, 32, 32, 24 if variant == "zerodce" else 3]
        for index, output_channels in enumerate(outputs, start=1):
            input_channels = channels[index - 1]
            # The fifth layer concatenates two 32-channel feature maps.
            if index >= 5:
                input_channels = 64
            layer = (
                nn.Conv2d(input_channels, output_channels, 3, padding=1)
                if variant == "zerodce"
                else SeparableConv(input_channels, output_channels)
            )
            setattr(self, f"e_conv{index}", layer)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = image
        if self.scale_factor != 1:
            features = F.interpolate(
                image, scale_factor=1 / self.scale_factor,
                mode="bilinear", align_corners=False,
            )
        first = F.relu(self.e_conv1(features))
        second = F.relu(self.e_conv2(first))
        third = F.relu(self.e_conv3(second))
        fourth = F.relu(self.e_conv4(third))
        fifth = F.relu(self.e_conv5(torch.cat((third, fourth), dim=1)))
        sixth = F.relu(self.e_conv6(torch.cat((second, fifth), dim=1)))
        curves = torch.tanh(self.e_conv7(torch.cat((first, sixth), dim=1)))
        if self.scale_factor != 1:
            # Explicit target size preserves every input pixel; no border crop.
            curves = F.interpolate(
                curves, size=image.shape[-2:], mode="bilinear", align_corners=True
            )
        enhanced = image
        for index in range(8):
            curve = curves[:, index * 3:(index + 1) * 3] if self.variant == "zerodce" else curves
            # Official checkpoints predict r = -A in the README's LE equation.
            enhanced = enhanced + curve * (enhanced.square() - enhanced)
        return enhanced


def enhance_clahe_bilateral(image_bgr: np.ndarray) -> np.ndarray:
    """Apply the exact default parameters from the linked project's DIP module."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)
    luminance = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(luminance)
    enhanced = cv2.cvtColor(
        cv2.merge((luminance, channel_a, channel_b)), cv2.COLOR_LAB2BGR
    )
    return cv2.bilateralFilter(enhanced, d=7, sigmaColor=50.0, sigmaSpace=50.0)


def load_model(checkpoint, variant="zerodce", scale_factor=1, device="cpu"):
    """Require a complete, matching pretrained state dictionary."""
    model = ZeroDCE(variant, scale_factor)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def enhance_neural(image_bgr, model, device="cpu"):
    """Enhance at the source image dimensions and return an 8-bit BGR image."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32) / 255.0
    with torch.inference_mode():
        result = model(tensor)
    if not torch.isfinite(result).all():
        raise ValueError("The model produced non-finite pixels.")
    rgb = (result[0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
