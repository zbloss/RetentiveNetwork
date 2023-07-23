import torch
import torch.nn as nn

from retentive_network.exceptions import InvalidTemperatureException
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
        chunk_size: int,
        half_point_precision: bool = False,
        use_complex_numbers: bool = False,
        softmax: bool = False,
    ):
        super(RetentiveNetworkCLM, self).__init__()

        self.number_of_layers: int = number_of_layers
        self.hidden_size: int = hidden_size
        self.number_of_heads: int = number_of_heads
        self.feed_forward_size: int = feed_forward_size
        self.vocab_size: int = vocab_size
        self.chunk_size: int = chunk_size
        self.half_point_precision: bool = half_point_precision
        self.use_complex_numbers: bool = use_complex_numbers
        self.softmax: bool = softmax

        self.torch_dtype: torch.dtype = (
            torch.float16 if self.half_point_precision else torch.float32
        )
        if self.use_complex_numbers:
            self.torch_dtype: torch.dtype = (
                torch.complex32 if self.half_point_precision else torch.complex64
            )

        self.model: nn.Module = RetentiveNetwork(
            number_of_layers=self.number_of_layers,
            hidden_size=self.hidden_size,
            number_of_heads=self.number_of_heads,
            feed_forward_size=self.feed_forward_size,
            chunk_size=self.chunk_size,
            half_point_precision=self.half_point_precision,
            use_complex_numbers=self.use_complex_numbers,
        )
        self.embedding_layer: nn.Module = nn.Embedding(self.vocab_size, hidden_size)
        self.projection: torch.Tensor = nn.Parameter(
            torch.randn(hidden_size, self.vocab_size, dtype=self.torch_dtype)
            / hidden_size
        )

        self.softmax_layer = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass includes passing `x` of shape
        [batch_size, sequence_length] and passes it
        through an embedding layer to shape
        [batch_size, sequence_length, hidden_size].
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
        x: torch.Tensor = x.real

        if self.softmax:
            x = self.softmax_layer(x)
            x = torch.mean(x, dim=-1)

        return x

    def forward_recurrent(self, x, previous_Ses, n):
        """
        Forward pass includes passing `x` of shape
        [batch_size, sequence_length] and passes it
        through an embedding layer to shape
        [batch_size, sequence_length, hidden_size].
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
        x: torch.Tensor = x.real

        if self.softmax:
            x = self.softmax_layer(x)
            x = torch.mean(x, dim=-1)

        return x, ses

    def forward_chunkwise(self, x: torch.Tensor, state: torch.Tensor = None):
        """
        Chunkwise forward pass includes passing `x` 
        of shape [batch_size, sequence_length] and 
        passes it through an embedding layer to shape
        [batch_size, sequence_length, hidden_size].
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
        x, state = self.model.forward_chunkwise(x, state)
        x: torch.Tensor = torch.matmul(x, self.projection.to(x.dtype))
        x: torch.Tensor = x.real

        if self.softmax:
            x = self.softmax_layer(x)
            x = torch.mean(x, dim=-1)

        return x, state

    def sample(
        self,
        x: torch.Tensor,
        sample_length: int,
        temperature: float = 1.0,
        number_of_samples: int = 1,
    ):
        """
        Uses the recurrent pass of the Retentive Network
        model to generate an output id response given
        a Tensor of shape [batch_size, sequence_length].

        Arguments:
            x (torch.Tensor): Tensor of shape [
                batch_size,
                sequence_length
            ].
            sample_length (int): How long in tokens the sequence returned should be.
            temperature (float): (0.0, 1.0] Controls the "randomness" or "creativity"
                                 of the model.
            number_of_samples (int): How many samples to generate from `x`.

        Returns:
            torch.Tensor: Tensor of shape [batch_size, `sample_length`]
        """

        if temperature <= 0 or temperature > 1:
            raise InvalidTemperatureException(temperature)
        batch_size, sequence_length = x.shape

        previous_Ses = [
            [
                torch.zeros(batch_size, self.head_size, self.head_size)
                for _ in range(self.number_of_heads)
            ]
            for _ in range(self.number_of_layers)
        ]

        for idx in range(sequence_length):
            X, previous_Ses = self.forward_recurrent(x[:, idx], previous_Ses, idx + 1)

        original_x = self._multinomial_probability_distribution(
            x=X, temperature=temperature, number_of_samples=number_of_samples
        )

        samples = []
        for sample_idx in range(number_of_samples):
            next_char = original_x[:, sample_idx]

            output_ids = []
            for idx in range(sample_length):
                x, previous_Ses = self.forward_recurrent(
                    next_char, previous_Ses, idx + 1
                )

                x = self._multinomial_probability_distribution(
                    x, temperature=temperature, number_of_samples=1
                )
                next_char = x[:, -1]
                output_ids.append(next_char)

            output_ids = torch.stack(output_ids, dim=1)
            samples.append(output_ids)
        samples = torch.stack(samples, dim=1)
        return samples

    def _multinomial_probability_distribution(
        self, x: torch.Tensor, temperature: float = 1.0, number_of_samples: int = 1
    ) -> torch.Tensor:
        """
        Helper method that converts x to a real tensor if it's complex
        then applies the temperature before a softmax layer and
        finally a multinomial probability distribution.

        Arguments:
            x (torch.Tensor): Tensor of shape [batch_size, self.vocab_size].
            temperature (float): (0.0, 1.0] Controls the "randomness" or "creativity"
                                  of the model.
            number_of_samples (int): Number of samples to sample from the
                                     multinomial probability distribution.
        Returns:
            torch.Tensor: Tensor of shape [batch_size, number_of_samples]
        """

        x = x.real if torch.is_complex(x) else x
        x /= temperature
        x = self.softmax_layer(x)
        x = torch.multinomial(x, num_samples=number_of_samples)
        return x

    @property
    def head_size(self):
        return self.model.retention_layers[0].head_size


if __name__ == "__main__":
    batch_size = 16
    number_of_layers = 4
    hidden_size = 16
    number_of_heads = 8
    sequence_length = 100
    feed_forward_size = 32
    vocab_size = 10
    sample_length = 20
    number_of_samples = 3
    chunk_size = 4
    softmax = True

    X = torch.randint(0, vocab_size, (batch_size, sequence_length))

    model = RetentiveNetworkCLM(
        number_of_layers=number_of_layers,
        hidden_size=hidden_size,
        number_of_heads=number_of_heads,
        feed_forward_size=feed_forward_size,
        vocab_size=vocab_size,
        chunk_size=chunk_size,
        softmax=softmax,
    )
    parallel_out = model(X)

    head_size = model.model.retention_layers[0].head_size

    previous_Ses = [
        [torch.zeros(batch_size, head_size, head_size) for _ in range(number_of_heads)]
        for _ in range(number_of_layers)
    ]

    recurrent_out = []
    for i in range(sequence_length):
        out, s = model.forward_recurrent(X[:, i], previous_Ses, i + 1)
        recurrent_out.append(out)
        previous_Ses = s

    recurrent_out = torch.stack(recurrent_out, dim=1)

    sample = model.sample(X, sample_length, number_of_samples=number_of_samples)
    assert sample.shape == (batch_size, number_of_samples, sample_length)

    if model.softmax:
        assert (parallel_out - recurrent_out).abs().max() < 1e-4
