# tests/test_backbones.py
import torch
import pytest


def test_get_backbone_efficientnet_b0():
    from cosmorford.backbones import get_backbone
    backbone, out_dim = get_backbone("efficientnet_b0")
    x = torch.randn(2, 3, 224, 224)
    out = backbone(x)
    assert out.shape[0] == 2
    assert out_dim == 1280


def test_get_backbone_convnext_tiny():
    from cosmorford.backbones import get_backbone
    backbone, out_dim = get_backbone("convnext_tiny")
    x = torch.randn(2, 3, 224, 224)
    out = backbone(x)
    assert out.shape[0] == 2
    assert out_dim == 768


def test_get_backbone_resnet18():
    from cosmorford.backbones import get_backbone
    backbone, out_dim = get_backbone("resnet18")
    x = torch.randn(2, 3, 224, 224)
    out = backbone(x)
    assert out.shape[0] == 2
    assert out_dim == 512


def test_get_backbone_unknown_raises():
    from cosmorford.backbones import get_backbone
    with pytest.raises(ValueError, match="Unknown backbone"):
        get_backbone("vgg99")
