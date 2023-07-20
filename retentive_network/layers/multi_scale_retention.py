import itertools

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
        self.head_size = self.hidden_size // self.number_of_heads

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
            torch.randn(self.hidden_size, self.hidden_size).to(self.complex_torch_dtype)
            / self.hidden_size
        )
        self.weight2: nn.Parameter = nn.Parameter(
            torch.randn(self.hidden_size, self.hidden_size).to(self.complex_torch_dtype)
            / self.hidden_size
        )
        self.group_norm: nn.Module = nn.GroupNorm(
            self.number_of_heads, self.hidden_size, dtype=self.torch_dtype
        )
        self.retention_layers: nn.ModuleList = nn.ModuleList(
            [Retention(self.head_size, gamma) for gamma in self.gammas]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass of the parallel form of MSR

        Arguments:
            x (torch.Tensor): A Tensor of shape [batch_size, sequence_length].

        Returns:
            torch.Tensor: A Tensor of shape [batch_size, sequence_length, self.hidden_size]
        """

        x = x.to(self.complex_torch_dtype)

        # Apply Retention to iterations of `x` by model head.
        retention_slices = []
        for head, retention_layer in zip(
            [_ for _ in range(self.number_of_heads)], self.retention_layers
        ):
            head_index_start = head * self.head_size
            head_index_end = (head + 1) * self.head_size
            x_slice = x[:, :, head_index_start:head_index_end]
            retention_slices.append(retention_layer(x_slice))

        # concatenate the computed retention slices into a
        # single matrix.
        retention_slices = torch.cat(retention_slices, dim=2)
        retention_slices = retention_slices.reshape(-1, self.hidden_size)
        retention_slices = retention_slices.real

        # Apply GroupNorm per the original paper per Page 4
        # `2.2 Gated Multi-Scale Retention`
        retention_slices = self.group_norm(retention_slices)
        retention_slices = retention_slices.reshape(x.shape)

        # Apply a SwishGate per the original paper Page 4
        # `2.2 Gated Multi-Scale Retention`
        out = self.swish_gate(torch.matmul(x, self.weight1))
        out += retention_slices
        out = torch.matmul(out, self.weight2)
        return out

    def forward_recurrent(self, x: torch.Tensor, previous_Ses: list, n: int) -> tuple:
        """
        Implements the forward pass of the recurrent form of MSR

        Arguments:
            x (torch.Tensor): A Tensor of shape [batch_size, sequence_length].
            previous_Ses (list): List of floats containing previous S values.
            n (int): The current nth iteration.

        Returns:
            torch.Tensor: A Tensor of shape [batch_size, sequence_length, self.hidden_size]
            torch.Tensor: s Tensor value to be used in the next
                          recurrent retention forward pass.
        """

        batch_size, sequence_length, hidden_size = x.shape[:3]

        x = x.to(self.complex_torch_dtype)
        n: torch.Tensor = torch.tensor(n, dtype=self.complex_torch_dtype)

        retention_slices = []
        ses = []
        for head, retention_layer, previous_S in zip(
            [_ for _ in range(self.number_of_heads)], 
            self.retention_layers,
            previous_Ses
        ):

            head_index_start = head * self.head_size
            head_index_end = (head + 1) * self.head_size
            x_slice = x[:, :, head_index_start:head_index_end]       

            retention_slice, s = retention_layer.forward_recurrent(
                x_slice, previous_S, n
            )
            print(f'retention_slice.shape: {retention_slice.shape}')

            retention_slices.append(retention_slice)
            ses.append(s)

        
        retention_slices = torch.cat(retention_slices, dim=1)
        # retention_slices = retention_slices.reshape(-1, self.hidden_size)

        print(f'\n\n\nbefore retention_slices.shape: {retention_slices.shape}')
        retention_slices = self.group_norm(retention_slices.real)
        print(f'\n\n\nafter retention_slices.shape: {retention_slices.shape}')

        out = self.swish_gate(torch.matmul(x, self.weight1))
        print(f'out.shape: {out.shape} | retention_slices.shape: {retention_slices.shape}')
        out += retention_slices.reshape(batch_size, hidden_size, -1)
        out = torch.matmul(out, self.weight2)

        return out, ses
        

if __name__ == "__main__":
    number_of_heads = 5
    batch_size, sequence_length, hidden_size = (4, 20, 100)

    MultiScaleRetention(hidden_size, number_of_heads)

    input_: torch.Tensor = torch.randn((batch_size, sequence_length, hidden_size))
    layer: nn.Module = MultiScaleRetention(hidden_size, number_of_heads)
    output: torch.Tensor = layer(input_)

    (output, S) = layer.forward_recurrent(input_, [0, 1, 2, 3, 4], 4)
    