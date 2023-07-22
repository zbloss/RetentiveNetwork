import pytest
import torch
import torch.nn as nn

from retentive_network.layers.multi_scale_retention import MultiScaleRetention
from retentive_network.exceptions import InvalidRetentionParametersException


class TestMultiScaleRetention:
    batch_size = 4
    sequence_length = 5
    hidden_size = 32
    number_of_heads = 4
    sample_tensor = torch.randn((batch_size, sequence_length, hidden_size))

    half_point_precision = False
    use_complex_numbers = False
    model = MultiScaleRetention(
        hidden_size, number_of_heads, half_point_precision, use_complex_numbers
    )

    def test_types(self):
        assert isinstance(self.model.hidden_size, type(self.hidden_size))
        assert isinstance(self.model.number_of_heads, type(self.number_of_heads))
        assert isinstance(
            self.model.half_point_precision, type(self.half_point_precision)
        )
        assert isinstance(
            self.model.use_complex_numbers, type(self.use_complex_numbers)
        )

        assert isinstance(self.model.torch_dtype, torch.dtype)
        assert isinstance(self.model.complex_torch_dtype, torch.dtype)

    def test_torch_complex(self):
        assert self.model.torch_dtype == torch.float32
        assert self.model.complex_torch_dtype == torch.complex64

        model = MultiScaleRetention(
            hidden_size=self.hidden_size,
            number_of_heads=self.number_of_heads,
            half_point_precision=True,
            use_complex_numbers=self.use_complex_numbers,
        )
        assert model.torch_dtype == torch.float16
        assert model.complex_torch_dtype == torch.complex32

    def test_gammas(self):
        assert isinstance(self.model.gammas, list)
        assert len(self.model.gammas) == self.number_of_heads

    def test_weight_types(self):
        assert self.model.weight1.dtype == torch.float32
        assert self.model.weight2.dtype == torch.float32

        model = MultiScaleRetention(
            hidden_size=self.hidden_size,
            number_of_heads=self.number_of_heads,
            half_point_precision=False,
            use_complex_numbers=True,
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
                half_point_precision=True,
                use_complex_numbers=self.use_complex_numbers,
            )
