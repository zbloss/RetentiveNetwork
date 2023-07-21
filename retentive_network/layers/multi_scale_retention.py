import torch
import torch.nn as nn

from retentive_network.exceptions import InvalidRetentionParametersException
from retentive_network.layers.retention import Retention
from retentive_network.layers.swish_gate import SwishGate


class MultiScaleRetention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        number_of_heads: int,
        half_point_precision: bool = False,
        use_complex_numbers: bool = False,
    ):
        super(MultiScaleRetention, self).__init__()

        if hidden_size % number_of_heads != 0:
            raise InvalidRetentionParametersException(hidden_size, number_of_heads)

        self.hidden_size: int = hidden_size
        self.number_of_heads: int = number_of_heads
        self.half_point_precision: bool = half_point_precision
        self.use_complex_numbers: bool = use_complex_numbers

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

        if self.use_complex_numbers:
            self.weight1: nn.Parameter = nn.Parameter(
                torch.randn(self.hidden_size, self.hidden_size).to(
                    self.complex_torch_dtype
                )
                / self.hidden_size
            )
            self.weight2: nn.Parameter = nn.Parameter(
                torch.randn(self.hidden_size, self.hidden_size).to(
                    self.complex_torch_dtype
                )
                / self.hidden_size
            )

        else:
            self.weight1: nn.Parameter = nn.Parameter(
                torch.randn(self.hidden_size, self.hidden_size) / self.hidden_size
            )
            self.weight2: nn.Parameter = nn.Parameter(
                torch.randn(self.hidden_size, self.hidden_size) / self.hidden_size
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

        if self.use_complex_numbers:
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

        if self.use_complex_numbers:
            x = x.to(self.complex_torch_dtype)

        n: torch.Tensor = torch.tensor(
            n,
            dtype=self.complex_torch_dtype
            if self.use_complex_numbers
            else self.torch_dtype,
            requires_grad=False,
        )

        retention_slices = []
        ses = []
        for head, retention_layer, previous_S in zip(
            [_ for _ in range(self.number_of_heads)],
            self.retention_layers,
            previous_Ses,
        ):
            head_index_start = head * self.head_size
            head_index_end = (head + 1) * self.head_size
            x_slice = x[:, head_index_start:head_index_end]

            retention_slice, s = retention_layer.forward_recurrent(
                x_slice, previous_S, n
            )
            retention_slices.append(retention_slice)
            ses.append(s)

        retention_slices = torch.cat(retention_slices, dim=1)

        retention_slices = self.group_norm(retention_slices.real)
        out = self.swish_gate(torch.matmul(x, self.weight1))

        out += retention_slices
        out = torch.matmul(out, self.weight2)
        return out, ses

    @property
    def head_size(self):
        return self.hidden_size // self.number_of_heads


if __name__ == "__main__":
    batch_size, sequence_length, hidden_size, number_of_heads = (4, 5, 32, 4)

    input_: torch.Tensor = torch.randn((batch_size, sequence_length, hidden_size))
    layer: nn.Module = MultiScaleRetention(hidden_size, number_of_heads)
    output: torch.Tensor = layer(input_)

    previous_S = [
        torch.randn(
            (
                batch_size,
                layer.head_size,
                layer.head_size,
            )
        )
        for _ in range(number_of_heads)
    ]

    retention_outputs = []
    for idx in range(sequence_length):
        out, s = layer.forward_recurrent(input_[:, idx, :], previous_S, idx)
        retention_outputs.append(out)
        previous_S = s

    retention_outputs = torch.stack(retention_outputs, dim=1)
    assert retention_outputs.shape == (batch_size, sequence_length, hidden_size)
