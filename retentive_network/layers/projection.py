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
        return self.model(x)
