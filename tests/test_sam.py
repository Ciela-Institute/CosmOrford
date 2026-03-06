import torch
from torch import nn


def test_sam_optimizer_step():
    from cosmorford.sam import SAM

    model = nn.Linear(10, 2)
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.01, rho=0.05)

    x = torch.randn(4, 10)
    loss = model(x).sum()
    loss.backward()
    optimizer.first_step(zero_grad=True)

    loss = model(x).sum()
    loss.backward()
    optimizer.second_step(zero_grad=True)
