import torch
import torch.nn as nn


class GroupNorm(nn.Module):
    def __init__(
        self,
        number_of_groups: int,
        number_of_channels: int,
        eps: float = 1e-5,
        half_point_precision: bool = False,
    ):
        super(GroupNorm, self).__init__()

        self.number_of_groups = number_of_groups
        self.number_of_channels = number_of_channels
        self.eps = eps
        self.dtype = torch.float16 if half_point_precision else torch.float32

        self.gamma = nn.Parameter(torch.ones(number_of_channels, dtype=self.dtype))
        self.beta = nn.Parameter(torch.zeros(number_of_channels, dtype=self.dtype))

        self.channels_per_group = self.number_of_channels // self.number_of_groups

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

        original_shape = x.shape
        # x = x.reshape(-1, self.number_of_groups, self.channels_per_group)

        mean = torch.mean(x, dim=1, keepdim=True)
        variance = torch.var(x, dim=1, keepdim=True)

        x = (x - mean) / torch.sqrt(variance + self.eps)
        x = x.reshape(-1, self.number_of_channels)
        x *= self.gamma
        x += self.beta

        x = x.reshape(original_shape)
        return x


if __name__ == "__main__":
    input_: torch.Tensor = torch.randn([4, 5, 32])
    layer: nn.Module = GroupNorm(4, 4)

    out: torch.Tensor = layer(input_)
