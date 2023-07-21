import torch
import torch.nn as nn

from retentive_network.layers.feed_forward import FeedForward


class TestFeedForward:
    hidden_size = 10
    feed_forward_size = 5
    model = FeedForward(hidden_size=hidden_size, feed_forward_size=feed_forward_size)

    def test_shape(self):
        x = torch.randn((5, 10))
        out = self.model(x)
        assert x.shape == out.shape