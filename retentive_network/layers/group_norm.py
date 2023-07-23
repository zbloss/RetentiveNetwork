import torch
import torch.nn as nn

from retentive_network.exceptions import ComplexTensorException


class GroupNorm(nn.Module):
    def __init__(
        self,
        number_of_groups: int,
        number_of_channels: int,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
    ):
        super(GroupNorm, self).__init__()

        self.number_of_groups = number_of_groups
        self.number_of_channels = number_of_channels
        self.eps = eps
        self.dtype = dtype

        self.gamma = nn.Parameter(torch.ones(number_of_channels, dtype=self.dtype))
        self.beta = nn.Parameter(torch.zeros(number_of_channels, dtype=self.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization on x.

        Arguments:
            x (torch.Tensor): Torch tensor of shape [
                    batch_size,
                    sequence_length,
                    hidden_size,
                ]
        Returns:
            torch.Tensor: Torch tensor of shape [
                    batch_size,
                    sequence_length,
                    hidden_size
                ]
        """

        if x.dtype == torch.complex32 or x.dtype == torch.complex64:
            raise ComplexTensorException(x)

        original_shape = x.shape

        mean = torch.mean(x, dim=1, keepdim=True)
        variance = torch.var(x, dim=1, keepdim=True)

        x = (x - mean) / torch.sqrt(variance + self.eps)
        x = x.reshape(-1, self.number_of_channels)
        x *= self.gamma
        x += self.beta

        x = x.reshape(original_shape)

        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        return x


if __name__ == "__main__":
    input_: torch.Tensor = torch.randn([4, 5, 32])
    layer: nn.Module = GroupNorm(4, 4)

    out: torch.Tensor = layer(input_)
