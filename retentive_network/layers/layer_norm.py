import torch
import torch.nn as nn

from retentive_network.exceptions import HalfPointPrecisionException


class LayerNorm(nn.Module):
    def __init__(
        self,
        number_of_channels: int,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
    ):
        super(LayerNorm, self).__init__()
        self.number_of_channels = number_of_channels
        self.eps = eps
        self.dtype = dtype

        self.gamma = nn.Parameter(torch.ones(number_of_channels, dtype=self.dtype))
        self.beta = nn.Parameter(torch.zeros(number_of_channels, dtype=self.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LayerNorm according to the original paper.

        Arguments:
            x (torch.Tensor): Torch tensor of generic size.

        Returns:
            torch.Tensor: Torch tensor of shape x.shape.
        """

        original_shape = x.shape

        x = x.reshape(-1, original_shape[-1])
        mean = torch.mean(x, dim=1, keepdim=True)
        variance = torch.var(x, dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        x *= self.gamma
        x += self.beta
        x = x.reshape(original_shape)

        if x.dtype != self.dtype:
            if torch.is_complex(x):
                raise HalfPointPrecisionException(x)
            else:
                x = x.to(self.dtype)

        return x


if __name__ == "__main__":
    batch_size, sequence_length, hidden_size = (4, 8, 32)
    x: torch.Tensor = torch.randn((batch_size, sequence_length, hidden_size))
    layer: nn.Module = LayerNorm(hidden_size)
    out: torch.Tensor = layer(x)
