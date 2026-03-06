"""Backbone registry for vision feature extractors."""
from torchvision.models.efficientnet import efficientnet_b0, efficientnet_v2_s
from torchvision.models.convnext import convnext_tiny, convnext_small
from torchvision.models.resnet import resnet18, resnet34


BACKBONES = {
    "efficientnet_b0": (efficientnet_b0, 1280),
    "efficientnet_v2_s": (efficientnet_v2_s, 1280),
    "convnext_tiny": (convnext_tiny, 768),
    "convnext_small": (convnext_small, 768),
    "resnet18": (resnet18, 512),
    "resnet34": (resnet34, 512),
}


def get_backbone(name: str):
    """Return (feature_extractor, output_dim) for a given backbone name.

    The feature extractor is the backbone without the classification head,
    outputting a 4D tensor [B, C, H, W].
    """
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone: {name}. Available: {list(BACKBONES.keys())}")

    factory, out_dim = BACKBONES[name]
    model = factory()

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
