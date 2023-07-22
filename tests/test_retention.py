import torch
import torch.nn as nn

from retentive_network.layers.retention import Retention


class TestRetention:
    batch_size = 4
    sequence_length = 20
    hidden_size = 100
    gamma = 0.1

    sample_tensor = torch.randn((batch_size, sequence_length, hidden_size))

    def test_types(self):
        half_point_precision = False
        use_complex_numbers = False
        model = Retention(
            hidden_size=self.hidden_size,
            gamma=self.gamma,
            half_point_precision=half_point_precision,
            use_complex_numbers=use_complex_numbers,
        )

        assert isinstance(model.hidden_size, int)
        assert isinstance(model.gamma, float)
        assert isinstance(model.half_point_precision, bool)
        assert isinstance(model.use_complex_numbers, bool)

        assert isinstance(model.torch_dtype, torch.dtype)
        assert isinstance(model.complex_torch_dtype, torch.dtype)

        assert model.torch_dtype == torch.float32
        assert model.complex_torch_dtype == torch.complex64

        assert model.weight_q.dtype == torch.float32
        assert model.weight_k.dtype == torch.float32
        assert model.weight_v.dtype == torch.float32
        assert model.theta.dtype == torch.float32

        half_point_precision = True
        use_complex_numbers = True
        model = Retention(
            hidden_size=self.hidden_size,
            gamma=self.gamma,
            half_point_precision=half_point_precision,
            use_complex_numbers=use_complex_numbers,
        )
        assert model.torch_dtype == torch.float16
        assert model.complex_torch_dtype == torch.complex32

        half_point_precision = False
        use_complex_numbers = True
        model = Retention(
            hidden_size=self.hidden_size,
            gamma=self.gamma,
            half_point_precision=half_point_precision,
            use_complex_numbers=use_complex_numbers,
        )
        assert model.torch_dtype == torch.float32
        assert model.complex_torch_dtype == torch.complex64

        assert model.weight_q.dtype == torch.complex64
        assert model.weight_k.dtype == torch.complex64
        assert model.weight_v.dtype == torch.complex64
        assert model.theta.dtype == torch.complex64

    def test_forward(self):
        half_point_precision = False
        use_complex_numbers = True
        model = Retention(
            hidden_size=self.hidden_size,
            gamma=self.gamma,
            half_point_precision=half_point_precision,
            use_complex_numbers=use_complex_numbers,
        )

        out = model(self.sample_tensor)
        assert out.shape == self.sample_tensor.shape

    def test_forward_recurrent(self):
        half_point_precision = False
        use_complex_numbers = True
        model = Retention(
            hidden_size=self.hidden_size,
            gamma=self.gamma,
            half_point_precision=half_point_precision,
            use_complex_numbers=use_complex_numbers,
        )

        previous_s = 0.12345
        n = 2

        out, S = model.forward_recurrent(self.sample_tensor, previous_s, n)

        assert out.shape == torch.Size(
            [
                self.batch_size,
                self.sequence_length,
                self.sequence_length,
                self.hidden_size,
            ]
        )

        assert S.shape == torch.Size(
            [self.batch_size, self.sequence_length, self.hidden_size, self.hidden_size]
        )

    def test_diagonal_matrix(self):
        half_point_precision = False
        use_complex_numbers = False
        model = Retention(
            hidden_size=self.hidden_size,
            gamma=self.gamma,
            half_point_precision=half_point_precision,
            use_complex_numbers=use_complex_numbers,
        )

        diagonal_matrix = model.diagonal_matrix(self.sequence_length)

        assert isinstance(diagonal_matrix, torch.Tensor)
        assert diagonal_matrix.shape == torch.Size(
            [self.sequence_length, self.sequence_length]
        )

        # Testing for values lower in the diagonal to
        # also be lower in value.
        for idx in range(diagonal_matrix.shape[0]):
            tensor_value = diagonal_matrix[idx, idx]

            assert tensor_value == diagonal_matrix[idx, :].max()
            assert tensor_value == diagonal_matrix[:, idx].max()
