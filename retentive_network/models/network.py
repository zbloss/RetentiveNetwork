import torch
import torch.nn as nn

from retentive_network.layers.feed_forward import FeedForward
from retentive_network.layers.layer_norm import LayerNorm
from retentive_network.layers.multi_scale_retention import MultiScaleRetention

# from util import ComplexFFN, ComplexGroupNorm, ComplexLayerNorm


class RetentiveNetwork(nn.Module):
    def __init__(
        self,
        number_of_layers: int,
        hidden_size: int,
        number_of_heads: int,
        feed_forward_size: int,
        half_point_precision: bool = False,
    ):
        super(RetentiveNetwork, self).__init__()

        self.number_of_layers = number_of_layers
        self.hidden_size = hidden_size
        self.feed_forward_size = feed_forward_size
        self.number_of_heads = number_of_heads
        self.half_point_precision = half_point_precision

        self.retention_layers = nn.ModuleList(
            [
                MultiScaleRetention(self.hidden_size, self.number_of_heads)
                for _ in range(self.number_of_layers)
            ]
        )
        self.feed_forward_layers = nn.ModuleList(
            [
                FeedForward(self.hidden_size, self.feed_forward_size)
                for _ in range(self.number_of_layers)
            ]
        )

        self.torch_dtype: torch.dtype = (
            torch.float16 if self.half_point_precision else torch.float32
        )
        self.complex_torch_dtype: torch.dtype = (
            torch.complex32 if self.half_point_precision else torch.complex64
        )
        self.layer_norm = LayerNorm(
            self.hidden_size, half_point_precision=self.half_point_precision
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the parallel forward pass as described in
        the original paper.

        Arguments:
            x (torch.Tensor): Torch tensor of shape
                              [batch_size, sequence_length, hidden_dim].

        Returns:
            torch.Tensor: Torch tensor of shape
                          [batch_size, sequence_length, hidden_dim].
        """

        for retention_layer, feed_forward_layer in zip(
            self.retention_layers, self.feed_forward_layers
        ):
            x_layer_norm = self.layer_norm(x)
            retention_out = retention_layer(x_layer_norm) + x

            retention_out_layer_norm = self.layer_norm(retention_out)
            x = feed_forward_layer(retention_out_layer_norm) + retention_out

        return x

    def forward_recurrent(self, x, previous_Ses, n):
        """
        Implements the recurrent forward pass as described in
        the original paper.

        Arguments:
            x (torch.Tensor): Torch tensor of shape
                              [batch_size, sequence_length, hidden_dim].
            previous_Ses (list): List of floats containing previous S values.
            n (int): The current nth iteration.

        Returns:
            torch.Tensor: A Tensor of shape [batch_size, sequence_length, self.hidden_size]
            torch.Tensor: s Tensor value to be used in the next
                          recurrent retention forward pass.
        """

        ses = []
        for i in range(self.number_of_layers):
            retention_layer = self.retention_layers[i]
            feed_forward_layer = self.feed_forward_layers[i]

            x_layer_norm = self.layer_norm(x)

            retention_out, s = retention_layer.forward_recurrent(
                x_layer_norm, previous_Ses[i], n
            )
            feed_forward_in = retention_out + x
            ses.append(s)

            feed_forward_in_layer_norm = self.layer_norm(feed_forward_in)
            x = feed_forward_layer(feed_forward_in_layer_norm) + feed_forward_in

        return x, ses


if __name__ == "__main__":
    (
        batch_size,
        sequence_length,
        hidden_size,
        num_heads,
        number_of_layers,
        feed_forward_size,
    ) = (
        8,
        5,
        32,
        4,
        4,
        20,
    )

    input_ = torch.randn(batch_size, sequence_length, hidden_size)

    model = RetentiveNetwork(
        number_of_layers, hidden_size, num_heads, feed_forward_size
    )
    parallel_out = model(input_)
    s_dim = hidden_size // num_heads

    previous_Ses = [
        [torch.zeros(batch_size, s_dim, s_dim) for _ in range(num_heads)]
        for _ in range(number_of_layers)
    ]

    recurrent_out = []
    for idx in range(sequence_length):
        Y, s_ns = model.forward_recurrent(input_[:, idx, :], previous_Ses, idx + 1)
        recurrent_out.append(Y)
        previous_Ses = s_ns

    recurrent_out = torch.stack(recurrent_out, dim=1)
