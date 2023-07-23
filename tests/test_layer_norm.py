import torch
import torch.nn as nn

from retentive_network.layers.layer_norm import LayerNorm


class TestLayerNorm:
    number_of_channels = 3
    eps = 1e-5

    def test_types(self):
        layer = LayerNorm(
            number_of_channels=self.number_of_channels,
            eps=self.eps,
            dtype=torch.float16,
        )
        assert layer.number_of_channels == self.number_of_channels
        assert layer.eps == self.eps

        assert isinstance(layer.number_of_channels, int)
        assert isinstance(layer.eps, float)

    def test_half_point_dtype(self):
        layer = LayerNorm(
            number_of_channels=self.number_of_channels,
            eps=self.eps,
            dtype=torch.float16,
        )
        assert layer.dtype == torch.float16

    def test_no_half_point_dtype(self):
        layer = LayerNorm(
            number_of_channels=self.number_of_channels,
            eps=self.eps,
            dtype=torch.float32,
        )
        assert layer.dtype == torch.float32

    def test_forward_shape(self):
        sample_tensor = torch.randn(3, 3)
        layer = LayerNorm(
            number_of_channels=self.number_of_channels,
            eps=self.eps,
            dtype=torch.float16,
        )

        out = layer(sample_tensor)
        assert out.shape == sample_tensor.shape
