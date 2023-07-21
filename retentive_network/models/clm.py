import torch
import torch.nn as nn

from retentive_network.models.network import RetentiveNetwork


class RetentiveNetworkCLM(nn.Module):
    """
    Huge shoutout to @Jamie-Stirling for
    breaking ground here first. The code below
    has been fit to match the rest of this repo
    but is heavily inspired from the great work
    already done here:
        * https://github.com/Jamie-Stirling/RetNet

    """

    def __init__(
        self,
        number_of_layers: int,
        hidden_size: int,
        number_of_heads: int,
        feed_forward_size: int,
        vocab_size: int,
        half_point_precision: bool = False,
        softmax: bool = False,
    ):
        super(RetentiveNetworkCLM, self).__init__()

        self.number_of_layers = number_of_layers
        self.hidden_size = hidden_size
        self.number_of_heads = number_of_heads
        self.feed_forward_size = feed_forward_size
        self.vocab_size = vocab_size
        self.half_point_precision = half_point_precision
        self.softmax = softmax

        self.torch_dtype: torch.dtype = (
            torch.float16 if self.half_point_precision else torch.float32
        )
        self.complex_torch_dtype: torch.dtype = (
            torch.complex32 if self.half_point_precision else torch.complex64
        )

        self.model: nn.Module = RetentiveNetwork(
            self.number_of_layers,
            self.hidden_size,
            self.feed_forward_size,
            self.number_of_heads,
        )
        self.embedding_layer: nn.Module = nn.Embedding(self.vocab_size, hidden_size)
        self.projection: torch.Tensor = nn.Parameter(
            torch.randn(hidden_size, self.vocab_size, dtype=self.torch_dtype)
            / hidden_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass includes passing `x` of shape
        [batch_size, sequence_length] and passes it
        through an embedding layer to shape
        [batch_size, sequence_length, hidden_dim].
        This tensor is then passed through the
        parallel pass of the RetentiveNetwork
        model before being projected into a tensor
        of shape [batch_size, self.vocab_size].

        Arguments:
            x (torch.Tensor): Tensor of shape [
                batch_size,
                sequence_length
            ].

        Returns:
            torch.Tensor: Tensor of shape [
                batch_size,
                self.vocab_size
            ].
        """

        x: torch.Tensor = self.embedding_layer(x)
        x: torch.Tensor = self.model(x)
        x: torch.Tensor = torch.matmul(x, self.projection.to(x.dtype))
        return X.real

    def forward_recurrent(self, x, previous_Ses, n):
        """
        Forward pass includes passing `x` of shape
        [batch_size, sequence_length] and passes it
        through an embedding layer to shape
        [batch_size, sequence_length, hidden_dim].
        This tensor is then passed through the
        recurrent pass of the RetentiveNetwork
        model before being projected into a tensor
        of shape [batch_size, self.vocab_size].

        Arguments:
            x (torch.Tensor): Tensor of shape [
                batch_size,
                sequence_length
            ].
            previous_Ses (list): List of floats containing previous S values.
            n (int): The current nth iteration.

        Returns:
            torch.Tensor: Tensor of shape [
                batch_size,
                self.vocab_size
            ].
        """

        x: torch.Tensor = self.embedding_layer(x)
        x, ses = self.model.forward_recurrent(x, previous_Ses, n)
        x: torch.Tensor = torch.matmul(x, self.projection.to(x.dtype))

        return x.real, ses

    def sample(self, input_ids, sample_length, temperature=1.0):
        """
        Uses the recurrent pass of the Retentive Network
        model to generate an output id response given
        a Tensor of shape [batch_size, sequence_length].

        Arguments:
            x (torch.Tensor): Tensor of shape [
                batch_size,
                sequence_length
            ].
            previous_Ses (list): List of floats containing previous S values.

        Returns:
            torch.Tensor:

        """

        s_n_1s = [
            [
                torch.zeros(
                    self.hidden_size // self.number_of_heads,
                    self.hidden_size // self.number_of_heads,
                    dtype=self.complex_torch_dtype,
                )
                .unsqueeze(0)
                .repeat(input_ids.shape[0], 1, 1)
                for _ in range(self.number_of_heads)
            ]
            for _ in range(self.number_of_layers)
        ]

        batch_size = input_ids.shape[0]
        # s_dim = self.hidden_size // self.number_of_heads
        # s_n_1s = [
        #     [
        #         torch.zeros(batch_size, s_dim, s_dim, dtype=self.complex_torch_dtype)
        #         for _ in range(self.number_of_heads)
        #     ]
        #     for _ in range(self.number_of_layers)
        # ]
        for i in range(input_ids.shape[1]):
            X, s_n_1s = self.forward_recurrent(input_ids[:, i], s_n_1s, i + 1)

        # get softmax of x (real part only)
        X = X.real / temperature
        X = torch.softmax(X, dim=-1)
        X = torch.multinomial(X, num_samples=1)
        next_char = X[:, -1]
        output_ids = []
        # now start sampling!
        for i in range(sample_length):
            X, s_n_1s = self.forward_recurrent(next_char, s_n_1s, i + 1)
            X = X.real / temperature
            X = torch.softmax(X, dim=-1)
            X = torch.multinomial(X, num_samples=1)
            next_char = X[:, -1]
            output_ids.append(next_char)

        output_ids = torch.stack(output_ids, dim=1)

        return output_ids

if __name__ == '__main__':
    batch_size = 2
    layers = 2
    hidden_dim = 16
    heads = 4
    sequence_length = 6
    ffn_size = 32
    vocab_size = 10

    X = torch.randint(0, vocab_size, (batch_size, sequence_length))

    model = RetentiveNetworkCLM(layers, hidden_dim, ffn_size, heads, vocab_size)
    Y_parallel = model(X)

    s_n_1s = [
        [
            torch.zeros(hidden_dim // heads, hidden_dim // heads, dtype=torch.complex64).unsqueeze(0).repeat(batch_size, 1, 1)
            for _ in range(heads)
        ] for _ in range(layers)
    ]

    Y_recurrent = []
    for i in range(sequence_length):
        Y, s_ns = model.forward_recurrent(X[:, i], s_n_1s, i+1)
        Y_recurrent.append(Y)
        s_n_1s = s_ns
    
    Y_recurrent = torch.stack(Y_recurrent, dim=1)

    # test sample
    sample_in = torch.randint(low=0, high=vocab_size-1, size=(batch_size, sequence_length))
    
    Y_sample = model.sample(sample_in, 5)

    assert (Y_sample.shape == (batch_size, 5))
    
    assert ((Y_parallel - Y_recurrent).abs().max() < 1e-4)
