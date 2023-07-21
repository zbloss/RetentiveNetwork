import torch
import torch.nn as nn
import torch.nn.functional as F


class SwishGate(nn.Module):
    def __init__(self):
        super(SwishGate, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the Swish Gate as described in the
        Retentive Network: A Successor to Transformer
        for Large Language Models paper.

        :ref: https://arxiv.org/pdf/1606.08415.pdf
        :ref: https://arxiv.org/pdf/1710.05941v1.pdf

        """

        x *= F.sigmoid(x)
        return x


if __name__ == "__main__":
    x: torch.Tensor = torch.randn((4, 20, 10))
    layer: nn.Module = SwishGate()
    output: torch.Tensor = layer(x)
