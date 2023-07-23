import pytest
import torch

from retentive_network.exceptions import InvalidTemperatureException
from retentive_network.models.clm import RetentiveNetworkCLM


class TestRetentiveNetworkCLM:
    batch_size = 16
    number_of_layers = 4
    hidden_size = 16
    number_of_heads = 8
    sequence_length = 100
    feed_forward_size = 32
    vocab_size = 10
    chunk_size = 2
    sample_length = 20
    number_of_samples = 3
    temperature = 0.9

    half_point_precision = False
    use_complex_numbers = False
    softmax = False

    model = RetentiveNetworkCLM(
        number_of_layers=number_of_layers,
        hidden_size=hidden_size,
        number_of_heads=number_of_heads,
        feed_forward_size=feed_forward_size,
        vocab_size=vocab_size,
        chunk_size=chunk_size,
        half_point_precision=half_point_precision,
        use_complex_numbers=use_complex_numbers,
        softmax=softmax,
    )

    sample_tensor = torch.randint(0, vocab_size, (batch_size, sequence_length))

    def test_types(self):
        assert self.model.torch_dtype == torch.float32

        model = RetentiveNetworkCLM(
            number_of_layers=self.number_of_layers,
            hidden_size=self.hidden_size,
            number_of_heads=self.number_of_heads,
            feed_forward_size=self.feed_forward_size,
            vocab_size=self.vocab_size,
            chunk_size=self.chunk_size,
            half_point_precision=True,
            use_complex_numbers=self.use_complex_numbers,
            softmax=self.softmax,
        )

        assert model.torch_dtype == torch.float16

    def test_forward(self):
        out = self.model(self.sample_tensor)
        assert out.shape == torch.Size(
            [self.batch_size, self.sequence_length, self.vocab_size]
        )

        model = RetentiveNetworkCLM(
            number_of_layers=self.number_of_layers,
            hidden_size=self.hidden_size,
            number_of_heads=self.number_of_heads,
            feed_forward_size=self.feed_forward_size,
            vocab_size=self.vocab_size,
            chunk_size=self.chunk_size,
            half_point_precision=True,
            use_complex_numbers=self.use_complex_numbers,
            softmax=True,
        )
        assert out.shape == torch.Size(
            [self.batch_size, self.sequence_length, self.vocab_size]
        )

    def test_forward_recurrent(self):
        head_size = self.model.model.retention_layers[0].head_size

        previous_Ses = [
            [
                torch.zeros(self.batch_size, head_size, head_size)
                for _ in range(self.number_of_heads)
            ]
            for _ in range(self.number_of_layers)
        ]

        recurrent_out = []
        for i in range(self.sequence_length):
            out, s = self.model.forward_recurrent(
                self.sample_tensor[:, i], previous_Ses, i + 1
            )
            recurrent_out.append(out)
            previous_Ses = s

        recurrent_out = torch.stack(recurrent_out, dim=1)

        assert recurrent_out.shape == torch.Size(
            [self.batch_size, self.sequence_length, self.vocab_size]
        )

        assert len(previous_Ses) == self.number_of_layers

        assert all([isinstance(x, list) for x in previous_Ses])
        assert all([self.number_of_heads == len(x) for x in previous_Ses])

    def test_head_size(self):
        assert self.model.head_size == (
            self.model.hidden_size // self.model.number_of_heads
        )
        assert isinstance(self.model.head_size, int)

    def test__multinomial_probability_distribution(self):
        sample_tensor = torch.randn((self.batch_size, self.number_of_samples))

        mpd = self.model._multinomial_probability_distribution(
            x=sample_tensor,
            temperature=self.temperature,
            number_of_samples=self.number_of_samples,
        )

        assert mpd.shape == torch.Size([self.batch_size, self.number_of_samples])

    def test_sample_invalid_temperature(self):
        sample_tensor = torch.randint(
            0, self.vocab_size, (self.batch_size, self.sequence_length)
        )

        invalid_temperature = -1
        with pytest.raises(InvalidTemperatureException):
            self.model.sample(
                sample_tensor,
                self.sample_length,
                temperature=invalid_temperature,
                number_of_samples=self.number_of_samples,
            )

    def test_sample(self):
        sample_tensor = torch.randint(
            0, self.vocab_size, (self.batch_size, self.sequence_length)
        )

        sample = self.model.sample(
            sample_tensor, self.sample_length, number_of_samples=self.number_of_samples
        )
        assert sample.shape == (
            self.batch_size,
            self.number_of_samples,
            self.sample_length,
        )
