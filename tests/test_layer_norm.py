import torch
import torch.nn as nn

from retentive_network.layers.layer_norm import LayerNorm


class TestLayerNorm:
    num_channels = 3
    eps = 1e-5

    def test_types(self):
        layer = LayerNorm(
            num_channels=self.num_channels, eps=self.eps, half_point_precision=True
        )
        assert layer.num_channels == self.num_channels
        assert layer.eps == self.eps

        assert isinstance(layer.num_channels, int)
        assert isinstance(layer.eps, float)

    def test_half_point_dtype(self):
        layer = LayerNorm(
            num_channels=self.num_channels, eps=self.eps, half_point_precision=True
        )
        assert layer.dtype == torch.float16

    def test_no_half_point_dtype(self):
        layer = LayerNorm(
            num_channels=self.num_channels, eps=self.eps, half_point_precision=False
        )
        assert layer.dtype == torch.float32

    def test_forward_shape(self):
        sample_tensor = torch.randn(3, 3)
        layer = LayerNorm(
            num_channels=self.num_channels, eps=self.eps, half_point_precision=True
        )

        out = layer(sample_tensor)
        assert out.shape == sample_tensor.shape
