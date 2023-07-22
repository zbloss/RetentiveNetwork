import torch
import torch.nn as nn

from retentive_network.layers.projection import Projection


class Retention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        gamma: float,
        chunk_size: int,
        dtype: torch.dtype = torch.float32,
    ):
        super(Retention, self).__init__()

        self.hidden_size: int = hidden_size
        self.gamma: float = gamma
        self.chunk_size: int = chunk_size
        self.dtype: torch.dtype = dtype

        self.project_q = Projection(
            hidden_size=self.hidden_size, bias=True, dtype=self.dtype
        )
        self.project_k = Projection(
            hidden_size=self.hidden_size, bias=True, dtype=self.dtype
        )
        self.project_v = Projection(
            hidden_size=self.hidden_size, bias=False, dtype=self.dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        `The Parallel Representation of Retention`

        Arguments:
            x (torch.Tensor): Tensor of shape [batch_size, sequence_length, hidden_size]

        Returns:
            torch.Tensor: Tensor value after applying parallel retention.
        """

        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        batch_size, sequence_length, hidden_size = x.shape[:3]
        diagonal_matrix: torch.Tensor = self.diagonal_matrix(sequence_length)

        q, k, v = self._project_qkv(x)

        attention_mask: torch.Tensor = torch.matmul(
            q, k.permute(0, 2, 1)
        ) * diagonal_matrix.unsqueeze(0)

        x: torch.Tensor = torch.matmul(attention_mask, v)
        return x

    def forward_recurrent(self, x: torch.Tensor, previous_S: torch.Tensor, n: int):
        """
        `The Recurrent Representation of Retention`.

        Shoutout to https://github.com/Jamie-Stirling/RetNet implementation

        Arguments:
            x (torch.Tensor): Tensor of shape [batch_size, hidden_size]
            previous_S (torch.Tensor): Tensor of shape [batch_size,
                                       hidden_size] that typically comes
                                       from the `s` value returned from the
                                       last time this method was called.

        Returns:
            torch.Tensor: x Tensor value after applying recurrent retention.
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

        q, k, v = self._project_qkv(x)

        matmulled = torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2))

        s: torch.Tensor = self.gamma * previous_S + matmulled
        x: torch.Tensor = torch.matmul(q.unsqueeze(1), s).squeeze(1)
        return x, s

    def diagonal_matrix(self, sequence_length: int) -> torch.Tensor:
        """
        Calculates a diagonal matrix with `1` on the diagonal
        and `gamma ** row` in the lower triangle diagonal row,
        and returns the matrix as a dtype.

        Arguments:
            sequence_length (int): Sequence size.

        Returns:
            torch.Tensor: Diagonal Matrix.

        """
        x: torch.Tensor = torch.diag(
            torch.tensor([1.0 for _ in range(sequence_length)], dtype=self.dtype),
            0,
        )
        for row in range(sequence_length - 1, 0, -1):
            eye: torch.Tensor = torch.tensor(
                [self.gamma ** (sequence_length - row) for _ in range(sequence_length)],
                dtype=self.dtype,
            )
            diagonal: torch.Tensor = torch.diag(eye, sequence_length - row)[
                :sequence_length, :sequence_length
            ].T
            x += diagonal

        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        return x

    def forward_chunkwise(self, x: torch.Tensor, state: torch.Tensor = None):
        """
        Implements a forward pass on a chunk `x` with
        hidden state `state` and bias `gamma`.

        Arguments:
            x (torch.Tensor): A Tensor of shape [batch_size, sequence_length, hidden_size].
            state (torch.Tensor): Torch Tensor of shape [batch_size, hidden_size, hidden_size].
                                  If None, a zero tensor is created.

        Returns:
            torch.Tensor: A Tensor of shape [batch_size, sequence_length, hidden_size]
            torch.Tensor: A Tensor of shape
                          [batch_size, sequence_length, kv_dim, kv_dim] where kv_dim
                          is hidden_size // number_of_heads

        """

        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        batch_size, sequence_length, hidden_size = x.shape
        if state is None:
            state = torch.zeros(
                (batch_size, hidden_size, hidden_size), dtype=self.dtype
            )

        q, k, v = self._project_qkv(x)

        retention = torch.matmul(q, k.transpose(-1, -2))
        retention_inner = torch.matmul(retention, v)
        retention_cross = torch.matmul(q, state)

        out = retention_inner + retention_cross

        state *= self.gamma
        kv = torch.matmul(k, v.transpose(-1, -2))
        kv_dim = hidden_size // sequence_length
        kv = kv.repeat([1, kv_dim, kv_dim])
        state += kv
        return out, state

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

        q = self.project_q(x)
        k = self.project_k(x)
        v = self.project_v(x)

        return q, k, v


if __name__ == "__main__":
    batch_size, sequence_length, hidden_size, chunk_size = (4, 20, 100, 4)

    input_: torch.Tensor = torch.randn((batch_size, sequence_length, hidden_size))
    layer: nn.Module = Retention(
        hidden_size=hidden_size,
        gamma=0.1,
        chunk_size=chunk_size,
    )
    parallel_out: torch.Tensor = layer(input_)

    recurrent_out, S = layer.forward_recurrent(input_, 0.1234, 2)
    chunkwise_out, state = layer.forward_chunkwise(x=input_, state=None)
    print(input_.shape, parallel_out.shape, recurrent_out.shape, chunkwise_out.shape)
