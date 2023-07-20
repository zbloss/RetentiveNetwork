import torch
import torch.nn as nn

from retentive_network.exceptions import InvalidRetentionParametersException
from retentive_network.layers.retention import Retention
from retentive_network.layers.swish_gate import SwishGate


class MultiScaleRetention(nn.Module):
    def __init__(
        self, hidden_size: int, number_of_heads: int, half_point_precision: bool = False
    ):
        super(MultiScaleRetention, self).__init__()

        if hidden_size % number_of_heads != 0:
            raise InvalidRetentionParametersException(hidden_size, number_of_heads)

        self.hidden_size: int = hidden_size
        self.number_of_heads: int = number_of_heads
        self.half_point_precision: bool = half_point_precision
        self.head_size = self.hidden_size / self.number_of_heads

        self.torch_dtype: torch.dtype = (
            torch.float16 if self.half_point_precision else torch.float32
        )
        self.complex_torch_dtype: torch.dtype = (
            torch.complex32 if self.half_point_precision else torch.complex64
        )

        # values pulled from Section 3.1 Setup - Parameter Allocation
        # https://arxiv.org/pdf/2307.08621.pdf
        gamma_minimum: torch.Tensor = torch.log(torch.tensor(1 / 512)).detach().cpu()
        gamma_maximum: torch.Tensor = torch.log(torch.tensor(1 / 32)).detach().cpu()
        linspace: torch.Tensor = torch.linspace(
            gamma_minimum, gamma_maximum, self.number_of_heads
        )
        self.gammas: list = (1 - torch.exp(linspace)).detach().cpu().tolist()

        self.swish_gate: nn.Module = SwishGate()

        self.weight1: nn.Parameter = nn.Parameter(
            torch.randn(hidden_size, hidden_size, dtype=self.complex_torch_dtype)
            / hidden_size
        )
        self.weight2: nn.Parameter = nn.Parameter(
            torch.randn(hidden_size, hidden_size, dtype=self.complex_torch_dtype)
            / hidden_size
        )
        self.group_norm = nn.GroupNorm(heads, hidden_size)

        self.retention_layers = nn.ModuleList(
            [Retention(self.head_size, gamma) for gamma in self.gammas]
        )


if __name__ == "__main__":
    hidden_size = 10
    number_of_heads = 5
    MultiScaleRetention(hidden_size, number_of_heads)
