import torch
import torch.nn as nn

from retentive_network.layers.feed_forward import FeedForward


class TestFeedForward:
    hidden_size = 10
    feed_forward_size = 5
    model = FeedForward(hidden_size=hidden_size, feed_forward_size=feed_forward_size)

    sample_tensor = torch.randn((5, 10))

    def test_in_out_shapes_match(self):
        out = self.model(self.sample_tensor)
        assert self.sample_tensor.shape == out.shape

    def test_tensors_returned(self):
        out = self.model(self.sample_tensor)
        assert isinstance(out, torch.Tensor)
