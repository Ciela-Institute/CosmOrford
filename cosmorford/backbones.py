"""Backbone registry for vision feature extractors."""
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


BACKBONES = {
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.DEFAULT, 1280),
    "efficientnet_v2_s": (efficientnet_v2_s, EfficientNet_V2_S_Weights.DEFAULT, 1280),
    "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT, 768),
    "convnext_small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT, 768),
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT, 512),
    "resnet34": (resnet34, ResNet34_Weights.DEFAULT, 512),
}


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
        import torch.nn as nn
        # ResNet: everything except avgpool and fc
        features = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
        )

    return features, out_dim
