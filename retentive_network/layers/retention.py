import torch
import torch.nn as nn


class Retention(nn.Module):
    def __init__(
        self, hidden_size: int, gamma: float, half_point_precision: bool = False
    ):
        super(Retention, self).__init__()

        self.hidden_size: int = hidden_size
        self.gamma: float = gamma
        self.half_point_precision: bool = half_point_precision

        self.torch_dtype: torch.dtype = (
            torch.float16 if self.half_point_precision else torch.float32
        )
        self.complex_torch_dtype: torch.dtype = (
            torch.complex32 if self.half_point_precision else torch.complex64
        )

        self.complex_tensor: torch.Tensor = torch.complex(
            torch.tensor(0.0), torch.tensor(1.0)
        ).to(self.complex_torch_dtype)

        self.weight_q: nn.Parameter = nn.Parameter(
            torch.randn(self.hidden_size, self.hidden_size) / self.hidden_size
        )
        self.weight_k: nn.Parameter = nn.Parameter(
            torch.randn(self.hidden_size, self.hidden_size) / self.hidden_size
        )
        self.weight_v: nn.Parameter = nn.Parameter(
            torch.randn(self.hidden_size, self.hidden_size) / self.hidden_size
        )
        self.theta: nn.Parameter = nn.Parameter(
            torch.randn(self.hidden_size) / self.hidden_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        `The Parallel Representation of Retention`

        Arguments:
            x (torch.Tensor): Tensor of shape [batch_size, sequence_length, hidden_size]

        Returns:
            torch.Tensor: Tensor value after applying parallel retention.
        """

        batch_size, sequence_length, hidden_size = x.shape[:3]
        diagonal_matrix: torch.Tensor = self.diagonal_matrix(sequence_length)

        x = x.to(self.complex_torch_dtype)

        thetas = []
        for seq_dim in range(1, sequence_length + 1):
            thetas.append(
                torch.exp(
                    self.complex_tensor
                    * torch.tensor(seq_dim, dtype=self.complex_torch_dtype)
                    * self.theta
                )
            )

        thetas: torch.Tensor = torch.stack(thetas, dim=0)
        theta_: torch.Tensor = thetas.conj()

        q: torch.Tensor = torch.matmul(
            x, self.weight_q.to(self.complex_torch_dtype)
        ) * thetas.unsqueeze(0)
        k: torch.Tensor = torch.matmul(
            x, self.weight_k.to(self.complex_torch_dtype)
        ) * theta_.unsqueeze(0)
        v: torch.Tensor = torch.matmul(x, self.weight_v.to(self.complex_torch_dtype))

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

        x = x.to(self.complex_torch_dtype)
        n: torch.Tensor = torch.tensor(n, dtype=self.complex_torch_dtype)

        theta: torch.Tensor = torch.exp(self.complex_tensor * n * self.theta)
        theta_: torch.Tensor = theta.conj()

        q: torch.Tensor = (
            torch.matmul(x, self.weight_q.to(self.complex_torch_dtype)) * theta
        )

        k: torch.Tensor = (
            torch.matmul(x, self.weight_k.to(self.complex_torch_dtype)) * theta_
        )
        v: torch.Tensor = torch.matmul(x, self.weight_v.to(self.complex_torch_dtype))
        matmulled = torch.matmul(
            k.unsqueeze(-1), v.unsqueeze(-2)
        )
       
        s: torch.Tensor = self.gamma * previous_S + matmulled
        x: torch.Tensor = torch.matmul(q.unsqueeze(1), s).squeeze(1)
        return x, s

    def diagonal_matrix(self, sequence_length: int) -> torch.Tensor:
        """
        Calculates a diagonal matrix with `1` on the diagonal
        and `gamma ** row` in the lower triangle diagonal row,
        and returns the matrix as a dtype
        self.complex_torch_dtype.

        Arguments:
            sequence_length (int): Sequence size.

        Returns:
            torch.Tensor: Diagonal Matrix.

        """
        x: torch.Tensor = torch.diag(
            torch.tensor([1.0 for _ in range(sequence_length)], dtype=self.torch_dtype),
            0,
        )
        for row in range(sequence_length - 1, 0, -1):
            eye: torch.Tensor = torch.tensor(
                [self.gamma ** (sequence_length - row) for _ in range(sequence_length)],
                dtype=self.torch_dtype,
            )
            diagonal: torch.Tensor = torch.diag(eye, sequence_length - row)[
                :sequence_length, :sequence_length
            ].T
            x += diagonal
        return x.to(self.complex_torch_dtype)


if __name__ == "__main__":
    batch_size, sequence_length, hidden_size = (4, 20, 100)

    input_: torch.Tensor = torch.randn((batch_size, sequence_length, hidden_size))
    layer: nn.Module = Retention(hidden_size, 0.1, False)
    output: torch.Tensor = layer(input_)

    out, S = layer.forward_recurrent(input_, 0.1234, 2)
