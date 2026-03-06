# Experiment Log

## Phase 1: Foundational Improvements
| ID | Run | val_loss | val_mse | Notes |
|----|-----|----------|---------|-------|
| P1-baseline | | | | Random init, StepLR |
| P1-pretrained | | | | Pretrained, cosine LR, 1-ch adapt |

## Phase 2: Backbone Sweep
| ID | Backbone | Run | val_loss | val_mse | Notes |
|----|----------|-----|----------|---------|-------|
| P2-effnet-b0 | EfficientNet-B0 | | | | |
| P2-effnet-v2s | EfficientNet V2-S | | | | |
| P2-convnext-t | ConvNeXt Tiny | | | | |
| P2-resnet18 | ResNet-18 | | | | |
| P2-swin-v2-t | Swin-V2-T | | | | |
| P2-maxvit-t | MaxViT-T | | | | May need padding |

## Phase 3: Augmentation & Regularization
(Fill in after Phase 2 winner selected)

## Phase 4: Best Combination
(Fill in after Phase 3)
