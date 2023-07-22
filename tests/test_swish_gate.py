import torch
import torch.nn as nn
import torch.nn.functional as F
from retentive_network.layers.swish_gate import SwishGate


class TestSwishGate:
    swish_gate = SwishGate()

    def test_shape(self):
        sample_tensor = torch.randn((4, 4))
        out = self.swish_gate(sample_tensor)

        assert sample_tensor.shape == out.shape
