import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, feed_forward_size: int):
        super(FeedForward, self).__init__()
        self.hidden_size: int = hidden_size
        self.feed_forward_size: int = feed_forward_size

        sqrt_hidden_size: torch.Tensor = torch.sqrt(torch.tensor(self.hidden_size))
        sqrt_feed_forward_size: torch.Tensor = torch.sqrt(
            torch.tensor(self.feed_forward_size)
        )

        self.weight1: nn.Parameter = nn.Parameter(
            torch.randn(self.hidden_size, self.feed_forward_size) / sqrt_hidden_size
        )

        self.weight2: nn.Parameter = nn.Parameter(
            torch.randn(self.feed_forward_size, self.hidden_size)
            / sqrt_feed_forward_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a 2-layer feed forward pass without bias.
        The input tensor `x` is multiplied with the first
        weight `self.weight1` then GELU is applied. Lastly,
        the resulting tensor is multiplied by the second
        and final weight parameter `self.weight2`.

        Arguments:
            x (torch.Tensor): Tensor of shape [batch_size, sequence_length, hidden_size]

        Returns:
            torch.Tensor: Tensor after applying feed forward operations.
        """

        x: torch.Tensor = torch.matmul(x.real, self.weight1)
        x: torch.Tensor = F.gelu(x)
        x: torch.Tensor = torch.matmul(x, self.weight2)
        return x


if __name__ == "__main__":
    input_: torch.Tensor = torch.randn((4, 20, 10))
    layer: nn.Module = FeedForward(10, 5)
    output: torch.Tensor = layer(input_)
