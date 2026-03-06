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


def test_adapt_first_conv_single_channel():
    from cosmorford.backbones import adapt_first_conv
    features, dim = get_backbone("resnet18", pretrained=True)
    features = adapt_first_conv(features, "resnet18")
    x = torch.randn(1, 1, 224, 224)
    out = features(x)
    assert out.shape[1] == dim


def test_adapt_first_conv_efficientnet():
    from cosmorford.backbones import adapt_first_conv
    features, dim = get_backbone("efficientnet_b0", pretrained=True)
    features = adapt_first_conv(features, "efficientnet_b0")
    x = torch.randn(1, 1, 224, 224)
    out = features(x)
    assert out.shape[1] == dim


def test_swin_v2_t_backbone():
    features, dim = get_backbone("swin_v2_t", pretrained=True)
    x = torch.randn(1, 3, 256, 256)  # Swin needs sizes divisible by window
    out = features(x)
    assert out.shape[1] == dim


def test_maxvit_t_backbone():
    features, dim = get_backbone("maxvit_t", pretrained=True)
    x = torch.randn(1, 3, 224, 224)
    out = features(x)
    assert out.shape[1] == dim


def test_adapt_first_conv_swin():
    from cosmorford.backbones import adapt_first_conv
    features, dim = get_backbone("swin_v2_t", pretrained=True)
    features = adapt_first_conv(features, "swin_v2_t")
    x = torch.randn(1, 1, 256, 256)
    out = features(x)
    assert out.shape[1] == dim


def test_adapt_first_conv_maxvit():
    from cosmorford.backbones import adapt_first_conv
    features, dim = get_backbone("maxvit_t", pretrained=True)
    features = adapt_first_conv(features, "maxvit_t")
    x = torch.randn(1, 1, 224, 224)
    out = features(x)
    assert out.shape[1] == dim
