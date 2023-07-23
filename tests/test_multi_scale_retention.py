import pytest
import torch
import torch.nn as nn

from retentive_network.exceptions import InvalidRetentionParametersException
from retentive_network.layers.multi_scale_retention import MultiScaleRetention


class TestMultiScaleRetention:
    batch_size = 4
    sequence_length = 5
    hidden_size = 32
    number_of_heads = 4
    chunk_size = 2
    sample_tensor = torch.randn((batch_size, sequence_length, hidden_size))

    model = MultiScaleRetention(
        hidden_size=hidden_size,
        number_of_heads=number_of_heads,
        chunk_size=chunk_size,
        dtype=torch.float32,
    )

    def test_types(self):
        assert isinstance(self.model.hidden_size, type(self.hidden_size))
        assert isinstance(self.model.number_of_heads, type(self.number_of_heads))
        assert isinstance(self.model.dtype, torch.dtype)

    def test_torch_complex(self):
        assert self.model.dtype == torch.float32

        model = MultiScaleRetention(
            hidden_size=self.hidden_size,
            number_of_heads=self.number_of_heads,
            chunk_size=self.chunk_size,
            dtype=torch.float16,
        )
        assert model.dtype == torch.float16

    def test_gammas(self):
        assert isinstance(self.model.gammas, list)
        assert len(self.model.gammas) == self.number_of_heads

    def test_weight_types(self):
        assert self.model.weight1.dtype == torch.float32
        assert self.model.weight2.dtype == torch.float32

        model = MultiScaleRetention(
            hidden_size=self.hidden_size,
            number_of_heads=self.number_of_heads,
            chunk_size=self.chunk_size,
            dtype=torch.complex64,
        )
        assert model.weight1.dtype == torch.complex64
        assert model.weight2.dtype == torch.complex64

    def test_forward(self):
        out = self.model(self.sample_tensor)
        assert isinstance(out, torch.Tensor)
        assert out.shape == self.sample_tensor.shape
        assert out.dtype == torch.float32

    def test_forward_recurrent(self):
        previous_S = [
            torch.randn(
                (
                    self.batch_size,
                    self.model.head_size,
                    self.model.head_size,
                )
            )
            for _ in range(self.number_of_heads)
        ]
        retention_outputs = []
        for idx in range(self.sequence_length):
            out, s = self.model.forward_recurrent(
                self.sample_tensor[:, idx, :], previous_S, idx
            )
            retention_outputs.append(out)
            previous_S = s

        retention_outputs = torch.stack(retention_outputs, dim=1)

        assert retention_outputs.shape == (
            self.batch_size,
            self.sequence_length,
            self.hidden_size,
        )

        assert isinstance(previous_S, list)
        assert all([isinstance(s, torch.Tensor) for s in previous_S])

    def test_head_size(self):
        assert self.model.head_size == (
            self.model.hidden_size // self.model.number_of_heads
        )
        assert isinstance(self.model.head_size, int)

    def test_invalid_parameters(self):
        bad_hidden_size = 10
        bad_number_of_heads = 11
        with pytest.raises(InvalidRetentionParametersException):
            model = MultiScaleRetention(
                hidden_size=bad_hidden_size,
                number_of_heads=bad_number_of_heads,
                chunk_size=self.chunk_size,
                dtype=torch.float16,
            )

    def test_forward_chunkwise(self):
        state = torch.zeros((self.batch_size, self.hidden_size, self.hidden_size))
        out, state = self.model.forward_chunkwise(self.sample_tensor, state)
        assert out.shape == torch.Size(
            [self.batch_size, self.sequence_length, self.hidden_size]
        )

        kv_dimension = self.hidden_size // self.number_of_heads
        assert state.shape == torch.Size([self.batch_size, kv_dimension, kv_dimension])
