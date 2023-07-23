import torch
import torch.nn as nn


class Projection(nn.Module):
    def __init__(
        self, hidden_size: int, bias: bool = True, dtype: torch.dtype = torch.float32
    ):
        super(Projection, self).__init__()
        self.hidden_size: int = hidden_size
        self.bias = bias
        self.dtype = dtype

        self.model: nn.Module = nn.Linear(
            self.hidden_size, self.hidden_size, bias=self.bias, dtype=self.dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a Linear Projection on `x`
        of shape [batch_size, sequence_length, hidden_size].

        Arguments:
            x (torch.Tensor): Torch tensor of shape
                              [batch_size, sequence_length, hidden_size].

        Returns:
            x (torch.Tensor): Torch tensor of shape
                              [batch_size, sequence_length, hidden_size].

        """
        if not x.dtype == self.dtype:
            x = x.to(self.dtype)

        x = self.model(x)

        return x


if __name__ == "__main__":
    batch_size, sequence_length, hidden_size = (4, 20, 100)
    x: torch.Tensor = torch.randn((batch_size, sequence_length, hidden_size))
    model: nn.Module = Projection(
        hidden_size=hidden_size, bias=True, dtype=torch.float32
    )

    out = model(x)
