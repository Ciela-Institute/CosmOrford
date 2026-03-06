import torch
import pytest
from cosmorford.backbones import get_backbone, BACKBONES


def test_get_backbone_pretrained():
    features, dim = get_backbone("resnet18", pretrained=True)
    x = torch.randn(1, 3, 224, 224)
    out = features(x)
    assert out.shape[1] == dim


def test_get_backbone_random_init():
    features, dim = get_backbone("resnet18", pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    out = features(x)
    assert out.shape[1] == dim


def test_get_backbone_unknown_raises():
    with pytest.raises(ValueError, match="Unknown backbone"):
        get_backbone("nonexistent")
