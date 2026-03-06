"""Backbone registry for vision feature extractors."""
import torch
import torch.nn as nn
from torchvision.models.efficientnet import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
)
from torchvision.models.convnext import (
    convnext_tiny, ConvNeXt_Tiny_Weights,
    convnext_small, ConvNeXt_Small_Weights,
)
from torchvision.models.resnet import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
)
from torchvision.models.swin_transformer import swin_v2_t, Swin_V2_T_Weights
from torchvision.models.maxvit import maxvit_t, MaxVit_T_Weights


BACKBONES = {
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.DEFAULT, 1280),
    "efficientnet_v2_s": (efficientnet_v2_s, EfficientNet_V2_S_Weights.DEFAULT, 1280),
    "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT, 768),
    "convnext_small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT, 768),
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT, 512),
    "resnet34": (resnet34, ResNet34_Weights.DEFAULT, 512),
    "swin_v2_t": (swin_v2_t, Swin_V2_T_Weights.DEFAULT, 768),
    "maxvit_t": (maxvit_t, MaxVit_T_Weights.DEFAULT, 512),
}


class SwinFeatures(nn.Module):
    """Wrapper that runs Swin features -> norm -> permute to output [B, C, H, W]."""

    def __init__(self, model):
        super().__init__()
        self.features = model.features
        self.norm = model.norm
        self.permute = model.permute

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        return x


class MaxVitFeatures(nn.Module):
    """Wrapper that runs MaxViT stem + blocks to output [B, C, H, W]."""

    def __init__(self, model):
        super().__init__()
        self.stem = model.stem
        self.blocks = model.blocks

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x


def get_backbone(name: str, pretrained: bool = True):
    """Return (feature_extractor, output_dim) for a given backbone name.

    The feature extractor is the backbone without the classification head,
    outputting a 4D tensor [B, C, H, W].

    Args:
        name: Backbone name (must be a key in BACKBONES).
        pretrained: If True, load DEFAULT pretrained weights. If False, random init.
    """
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone: {name}. Available: {list(BACKBONES.keys())}")

    factory, default_weights, out_dim = BACKBONES[name]
    weights = default_weights if pretrained else None
    model = factory(weights=weights)

    # Extract just the feature layers (no classifier head)
    if name.startswith("efficientnet"):
        features = model.features
    elif name.startswith("convnext"):
        features = model.features
    elif name.startswith("resnet"):
        # ResNet: everything except avgpool and fc
        features = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
        )
    elif name == "swin_v2_t":
        features = SwinFeatures(model)
    elif name == "maxvit_t":
        features = MaxVitFeatures(model)

    return features, out_dim


def adapt_first_conv(features, name: str):
    """Replace first conv layer from 3-channel to 1-channel input.
    Sums pretrained 3-channel weights into a single channel."""
    if name.startswith("resnet"):
        old_conv = features[0]  # conv1
    elif name.startswith("efficientnet"):
        old_conv = features[0][0]  # features[0] is first ConvBNActivation block
    elif name.startswith("convnext"):
        old_conv = features[0][0]  # features[0] is the stem
    elif name == "swin_v2_t":
        old_conv = features.features[0][0]  # features -> Sequential[0] -> Conv2d
    elif name == "maxvit_t":
        old_conv = features.stem[0][0]  # stem -> Conv2dNormActivation[0] -> Conv2d

    new_conv = nn.Conv2d(
        1, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    if name.startswith("resnet"):
        features[0] = new_conv
    elif name.startswith("efficientnet"):
        features[0][0] = new_conv
    elif name.startswith("convnext"):
        features[0][0] = new_conv
    elif name == "swin_v2_t":
        features.features[0][0] = new_conv
    elif name == "maxvit_t":
        features.stem[0][0] = new_conv

    return features
