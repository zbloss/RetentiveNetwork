import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(
        self, num_channels: int, eps: float = 1e-5, half_point_precision: bool = False
    ):
        super(LayerNorm, self).__init__()
        self.num_channels = num_channels
        self.eps = eps

        dtype = torch.float16 if half_point_precision else torch.float32

        self.gamma = nn.Parameter(torch.ones(num_channels, dtype=dtype))
        self.beta = nn.Parameter(torch.zeros(num_channels, dtype=dtype))

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
        return x


if __name__ == "__main__":
    batch_size, sequence_length, hidden_dim = (4, 8, 32)
    x: torch.Tensor = torch.randn((batch_size, sequence_length, hidden_dim))
    layer: nn.Module = LayerNorm(hidden_dim)
    out: torch.Tensor = layer(x)
