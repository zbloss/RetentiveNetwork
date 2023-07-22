import torch
import torch.nn as nn

from retentive_network.exceptions import (
    InvalidHiddenSizeException,
    InvalidRetentionParametersException,
)
from retentive_network.layers.group_norm import GroupNorm
from retentive_network.layers.projection import Projection
from retentive_network.layers.retention import Retention
from retentive_network.layers.swish_gate import SwishGate


class MultiScaleRetention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        number_of_heads: int,
        chunk_size: int,
        dtype: torch.dtype = torch.float32,
    ):
        super(MultiScaleRetention, self).__init__()

        if hidden_size % number_of_heads != 0:
            raise InvalidRetentionParametersException(hidden_size, number_of_heads)

        self.hidden_size: int = hidden_size
        self.number_of_heads: int = number_of_heads
        self.chunk_size: int = chunk_size
        self.dtype: torch.dtype = dtype

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
            torch.randn(self.hidden_size, self.hidden_size).to(self.dtype)
            / self.hidden_size
        )
        self.weight2: nn.Parameter = nn.Parameter(
            torch.randn(self.hidden_size, self.hidden_size).to(self.dtype)
            / self.hidden_size
        )

        self.group_norm: nn.Module = GroupNorm(
            number_of_groups=self.number_of_heads,
            number_of_channels=self.hidden_size,
            dtype=self.dtype,
        )
        self.retention_layers: nn.ModuleList = nn.ModuleList(
            # [Retention(self.head_size, gamma) for gamma in self.gammas]
            [
                Retention(
                    hidden_size=self.head_size,
                    gamma=gamma,
                    chunk_size=self.chunk_size,
                    dtype=self.dtype,
                )
                for gamma in self.gammas
            ]
        )

        self.weight_q_projection: nn.Module = Projection(
            hidden_size=self.hidden_size, bias=True, dtype=self.dtype
        )
        self.weight_k_projection: nn.Module = Projection(
            hidden_size=self.hidden_size, bias=True, dtype=self.dtype
        )
        self.weight_v_projection: nn.Module = Projection(
            hidden_size=self.hidden_size, bias=False, dtype=self.dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass of the parallel form of MSR

        Arguments:
            x (torch.Tensor): A Tensor of shape [batch_size, sequence_length].

        Returns:
            torch.Tensor: A Tensor of shape [batch_size, sequence_length, self.hidden_size]
        """

        if x.dtype != self.dtype:
            x = x.to(self.dtype)

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

        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        n: torch.Tensor = torch.tensor(
            n,
            dtype=self.dtype,
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

    def forward_chunkwise(
        self, x: torch.Tensor, previous_kv: torch.Tensor
    ) -> torch.Tensor:
        """
        Implements the forward pass of the chunkwise form of MSR.

        Arguments:
            x (torch.Tensor): A Tensor of shape [batch_size, sequence_length].

        Returns:
            torch.Tensor: A Tensor of shape [batch_size, sequence_length, hidden_size]
            torch.Tensor: A Tensor of shape
                          [batch_size, sequence_length, kv_dim, kv_dim] where kv_dim
                          is hidden_size // number_of_heads
        """

        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        batch_size, sequence_length, hidden_size = x.shape

        if hidden_size != self.hidden_size:
            raise InvalidHiddenSizeException(hidden_size, self.hidden_size)

        total_chunks = sequence_length // self.chunk_size
        if sequence_length % self.chunk_size != 0:
            total_chunks += 1

        print(f"MSR total_chunks: {total_chunks}")

        # Initialize state
        state = torch.zeros(batch_size, hidden_size, hidden_size)

        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = (1 + chunk_idx) * self.chunk_size
            if end_idx > sequence_length:
                end_idx = sequence_length

            chunk = x[:, start_idx:end_idx]

        q, k, v = self._project_qkv(x)
        retention = torch.matmul(q, k.transpose(-1, -2))

        inner_retention = torch.matmul(retention, v)
        cross_retention = torch.matmul(q, previous_kv)
        retention = inner_retention + cross_retention

        output = self.group_norm(retention)
        current_kv = previous_kv + torch.matmul(k.transpose(-1, -2), v)

        output = output.reshape(batch_size, sequence_length, hidden_size)
        return output, current_kv

    def _project_qkv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Helper method to project Q, K, and V values
        from x.

        Arguments:
            x (torch.Tensor): Torch tensor of shape [
                batch_size, sequence_length,
                hidden_size, chunk_size
            ]

        Returns:
            torch.Tensor: (Q) Torch tensor of shape [
                batch_size, sequence_length,
                hidden_size, chunk_size
            ]
            torch.Tensor: (K) Torch tensor of shape [
                batch_size, sequence_length,
                hidden_size, chunk_size
            ]
            torch.Tensor: (V) Torch tensor of shape [
                batch_size, sequence_length,
                hidden_size, chunk_size
            ]
        """

        (batch_size, sequence_length, hidden_size) = x.shape

        if hidden_size != self.hidden_size:
            raise InvalidHiddenSizeException(hidden_size, self.hidden_size)

        q = self.weight_q_projection(x)
        k = self.weight_k_projection(x)
        v = self.weight_v_projection(x)

        q = q.reshape(batch_size, sequence_length, self.number_of_heads, -1)
        k = k.reshape(batch_size, sequence_length, self.number_of_heads, -1)
        v = v.reshape(batch_size, sequence_length, self.number_of_heads, -1)

        return q, k, v

    @property
    def head_size(self):
        return self.hidden_size // self.number_of_heads


if __name__ == "__main__":
    batch_size, sequence_length, hidden_size, number_of_heads, chunk_size = (
        4,
        20,
        100,
        4,
        2,
    )

    # batch_size, sequence_length, hidden_size = (4, 20, 100)

    input_: torch.Tensor = torch.randn((batch_size, sequence_length, hidden_size))

    layer: nn.Module = MultiScaleRetention(
        hidden_size=hidden_size, number_of_heads=number_of_heads, chunk_size=chunk_size
    )
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

    q, k, v = layer._project_qkv(input_)
    previous_kv = torch.matmul(k.transpose(-1, -2), v)
    out, previous_kv = layer.forward_chunkwise(input_, previous_kv)
    assert out.shape == (batch_size, sequence_length, hidden_size)
    kv_dim = hidden_size // layer.number_of_heads
    assert previous_kv.shape == (batch_size, sequence_length, kv_dim, kv_dim)
