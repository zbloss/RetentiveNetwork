import torch
import torch.nn as nn

from retentive_network.exceptions import InvalidHiddenSizeException
from retentive_network.layers.feed_forward import FeedForward
from retentive_network.layers.layer_norm import LayerNorm
from retentive_network.layers.multi_scale_retention import MultiScaleRetention


class RetentiveNetwork(nn.Module):
    def __init__(
        self,
        number_of_layers: int,
        hidden_size: int,
        number_of_heads: int,
        feed_forward_size: int,
        chunk_size: int,
        half_point_precision: bool = False,
        use_complex_numbers: bool = False,
    ):
        super(RetentiveNetwork, self).__init__()

        self.number_of_layers: int = number_of_layers
        self.hidden_size: int = hidden_size
        self.feed_forward_size: int = feed_forward_size
        self.number_of_heads: int = number_of_heads
        self.chunk_size: int = chunk_size
        self.half_point_precision: bool = half_point_precision
        self.use_complex_numbers: bool = use_complex_numbers

        self.torch_dtype: torch.dtype = (
            torch.float16 if self.half_point_precision else torch.float32
        )
        if self.use_complex_numbers:
            self.torch_dtype: torch.dtype = (
                torch.complex32 if self.half_point_precision else torch.complex64
            )

        self.retention_layers: nn.ModuleList = nn.ModuleList(
            [
                MultiScaleRetention(
                    hidden_size=self.hidden_size,
                    number_of_heads=self.number_of_heads,
                    chunk_size=self.chunk_size,
                    dtype=self.torch_dtype,
                )
                for _ in range(self.number_of_layers)
            ]
        )
        self.feed_forward_layers: nn.ModuleList = nn.ModuleList(
            [
                FeedForward(self.hidden_size, self.feed_forward_size)
                for _ in range(self.number_of_layers)
            ]
        )

        self.layer_norm: nn.Module = LayerNorm(self.hidden_size, dtype=self.torch_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the parallel forward pass as described in
        the original paper.

        Arguments:
            x (torch.Tensor): Torch tensor of shape
                              [batch_size, sequence_length, hidden_size].

        Returns:
            torch.Tensor: Torch tensor of shape
                          [batch_size, sequence_length, hidden_size].
        """

        for retention_layer, feed_forward_layer in zip(
            self.retention_layers, self.feed_forward_layers
        ):
            x_layer_norm: torch.Tensor = self.layer_norm(x)
            retention_out: torch.Tensor = retention_layer(x_layer_norm) + x

            retention_out_layer_norm: torch.Tensor = self.layer_norm(retention_out)
            x: torch.Tensor = (
                feed_forward_layer(retention_out_layer_norm) + retention_out
            )

        return x

    def forward_recurrent(self, x, previous_Ses, n):
        """
        Implements the recurrent forward pass as described in
        the original paper.

        Arguments:
            x (torch.Tensor): Torch tensor of shape
                              [batch_size, sequence_length, hidden_size].
            previous_Ses (list): List of floats containing previous S values.
            n (int): The current nth iteration.

        Returns:
            torch.Tensor: A Tensor of shape [batch_size, sequence_length, self.hidden_size]
            torch.Tensor: s Tensor value to be used in the next
                          recurrent retention forward pass.
        """

        ses = []
        for i in range(self.number_of_layers):
            retention_layer: nn.Module = self.retention_layers[i]
            feed_forward_layer: nn.Module = self.feed_forward_layers[i]

            x_layer_norm: torch = self.layer_norm(x)

            retention_out, s = retention_layer.forward_recurrent(
                x_layer_norm, previous_Ses[i], n
            )
            feed_forward_in: torch = retention_out + x
            ses.append(s)

            feed_forward_in_layer_norm: torch = self.layer_norm(feed_forward_in)
            x: torch = feed_forward_layer(feed_forward_in_layer_norm) + feed_forward_in

        return x, ses

    def forward_chunkwise(self, x: torch.Tensor, state: torch.Tensor = None):
        """
        Implements the chunkwise forward pass as described in
        the original paper.

        Arguments:
            x (torch.Tensor): Torch tensor of shape
                              [batch_size, sequence_length, hidden_size].
            state (torch.Tensor): previous state value returned from the previous
                                  forward_chunkwise() call. If None,
                                  a torch.zeros() state is initialized
                                  in it's place

        Returns:
            torch.Tensor: A Tensor of shape [batch_size, sequence_length, self.hidden_size]
            torch.Tensor: state Tensor value to be used in the next
                          recurrent retention forward pass of shape
                          [batch_size, sequence_length, kv_dim, kv_dim] where kv_dim
                          is hidden_size // number_of_heads
        """

        batch_size, sequence_length, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise InvalidHiddenSizeException(
                hidden_size=hidden_size, model_required_hidden_size=self.hidden_size
            )

        state = torch.zeros(
            (batch_size, hidden_size, hidden_size),
            dtype=self.torch_dtype,
        )

        for i in range(self.number_of_layers):
            x_layer_norm: torch.Tensor = self.layer_norm(x)

            retention_layer: nn.Module = self.retention_layers[i]
            feed_forward_layer: nn.Module = self.feed_forward_layers[i]

            chunkwise_out, state = retention_layer.forward_chunkwise(x, state)
            feed_forward_in: torch.Tensor = chunkwise_out + x
            feed_forward_in_layer_norm: torch = self.layer_norm(feed_forward_in)

            x: torch = feed_forward_layer(feed_forward_in_layer_norm) + feed_forward_in

        return x, state


if __name__ == "__main__":
    (
        batch_size,
        sequence_length,
        hidden_size,
        number_of_heads,
        number_of_layers,
        feed_forward_size,
        chunk_size,
    ) = (8, 5, 32, 4, 4, 20, 4)

    input_: torch.Tensor = torch.randn(batch_size, sequence_length, hidden_size)

    model: nn.Module = RetentiveNetwork(
        number_of_layers=number_of_layers,
        hidden_size=hidden_size,
        number_of_heads=number_of_heads,
        feed_forward_size=feed_forward_size,
        chunk_size=chunk_size,
    )
    parallel_out: torch.Tensor = model(input_)
    s_dim: int = hidden_size // number_of_heads

    previous_Ses = [
        [torch.zeros(batch_size, s_dim, s_dim) for _ in range(number_of_heads)]
        for _ in range(number_of_layers)
    ]

    recurrent_out = []
    for idx in range(sequence_length):
        out, s_ns = model.forward_recurrent(input_[:, idx, :], previous_Ses, idx + 1)
        recurrent_out.append(out)
        previous_Ses = s_ns

    recurrent_out: torch.Tensor = torch.stack(recurrent_out, dim=1)

    chunkwise_out, chunkwise_state = model.forward_chunkwise(
        x=input_,
        state=None,
    )
