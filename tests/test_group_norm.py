import pytest
import torch

from retentive_network.exceptions import ComplexTensorException
from retentive_network.layers.group_norm import GroupNorm


class TestGroupNorm:
    number_of_groups: int = 4
    number_of_channels: int = 8

    batch_size: int = 4
    sequence_length: int = 5
    hidden_size: int = 32

    sample_tensor: torch.Tensor = torch.randn(
        [batch_size, sequence_length, hidden_size]
    )

    def test_types(self):
        model = GroupNorm(
            number_of_groups=self.number_of_groups,
            number_of_channels=self.number_of_channels,
            dtype=torch.float32,
        )
        out = model(self.sample_tensor)
        assert model.dtype == torch.float32
        assert out.dtype == torch.float32

        model = GroupNorm(
            number_of_groups=self.number_of_groups,
            number_of_channels=self.number_of_channels,
            dtype=torch.float16,
        )
        out = model(self.sample_tensor)
        assert model.dtype == torch.float16
        assert out.dtype == torch.float16

        with pytest.raises(ComplexTensorException):
            model = GroupNorm(
                number_of_groups=self.number_of_groups,
                number_of_channels=self.number_of_channels,
                dtype=torch.complex32,
            )
            sample_tensor: torch.Tensor = torch.randn(
                [self.batch_size, self.sequence_length, self.hidden_size],
                dtype=torch.complex32,
            )
            out = model(sample_tensor)

        with pytest.raises(ComplexTensorException):
            model = GroupNorm(
                number_of_groups=self.number_of_groups,
                number_of_channels=self.number_of_channels,
                dtype=torch.complex64,
            )
            sample_tensor: torch.Tensor = torch.randn(
                [self.batch_size, self.sequence_length, self.hidden_size],
                dtype=torch.complex64,
            )
            out = model(sample_tensor)

    def test_shapes(self):
        model = GroupNorm(
            number_of_groups=self.number_of_groups,
            number_of_channels=self.number_of_channels,
            dtype=torch.float32,
        )
        out = model(self.sample_tensor)
        assert out.shape == self.sample_tensor.shape
        assert out.shape == (self.batch_size, self.sequence_length, self.hidden_size)
