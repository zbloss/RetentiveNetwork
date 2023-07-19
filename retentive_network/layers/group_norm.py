import math
import torch
import torch.nn as nn


class ComplexGroupNorm(nn.Module):
    """
    Complex Group Normalization as described on Page 6 of
    :ref: https://openaccess.thecvf.com/content_ECCV_2018/papers/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.pdf
    :Authors: Yuxin Wu, Kaiming He

    """
    def __init__(self, number_of_groups: int, number_of_channels: int, eps: float = 1e-5):
        super(ComplexGroupNorm, self).__init__()

        self.number_of_groups = number_of_groups
        self.number_of_channels = number_of_channels
        
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(number_of_channels, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(number_of_channels, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization on `x`.

        Arguments:
            x (torch.Tensor): Tensor of shape [batch_size, sequence_length, hidden_size].
        Returns:
            torch.Tensor: Shape [-1, `number_of_channels`] x after applying Group Normalization.
        """

        x = x.reshape(-1, self.number_of_groups, self.number_of_channels // self.number_of_groups)    

        mean = torch.mean(x, dim=2, keepdim=True)
        variance = torch.var(x, dim=2, keepdim=True)
        
        x = (x - mean) / torch.sqrt(variance + self.eps)
        x = x.reshape(-1, self.number_of_channels)
        x = x * self.gamma + self.beta
        return x

if __name__ == '__main__':
    input_ = torch.randn(8, 20, 18)
    layer = ComplexGroupNorm(3, 6)
    complex_output = layer(input_)
    