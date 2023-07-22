import torch
import torch.nn as nn

from retentive_network.layers.retention import Retention


class TestRetention:
    batch_size = 4
    sequence_length = 20
    hidden_size = 100
    gamma = 0.1
    chunk_size = 4

    sample_tensor = torch.randn((batch_size, sequence_length, hidden_size))

    def test_types(self):
        model = Retention(
            hidden_size=self.hidden_size,
            gamma=self.gamma,
            chunk_size=self.chunk_size,
            dtype=torch.float32,
        )

        assert isinstance(model.hidden_size, int)
        assert isinstance(model.gamma, float)
        assert isinstance(model.dtype, torch.dtype)

        assert model.dtype == torch.float32
        assert model.project_q.dtype == torch.float32
        assert model.project_k.dtype == torch.float32
        assert model.project_v.dtype == torch.float32

        model = Retention(
            hidden_size=self.hidden_size,
            gamma=self.gamma,
            chunk_size=self.chunk_size,
            dtype=torch.complex32,
        )
        assert model.dtype == torch.complex32

        model = Retention(
            hidden_size=self.hidden_size,
            gamma=self.gamma,
            chunk_size=self.chunk_size,
            dtype=torch.complex64,
        )

        assert model.dtype == torch.complex64
        assert model.project_q.dtype == torch.complex64
        assert model.project_k.dtype == torch.complex64
        assert model.project_v.dtype == torch.complex64

    def test_forward(self):
        model = Retention(
            hidden_size=self.hidden_size,
            gamma=self.gamma,
            chunk_size=self.chunk_size,
            dtype=torch.complex64,
        )

        out = model(self.sample_tensor)
        assert out.shape == self.sample_tensor.shape

    def test_forward_recurrent(self):
        model = Retention(
            hidden_size=self.hidden_size,
            gamma=self.gamma,
            chunk_size=self.chunk_size,
            dtype=torch.complex64,
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
        model = Retention(
            hidden_size=self.hidden_size,
            gamma=self.gamma,
            chunk_size=self.chunk_size,
            dtype=torch.float32,
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

    def test_forward_chunkwise(self):
        model = Retention(
            hidden_size=self.hidden_size,
            gamma=self.gamma,
            chunk_size=self.chunk_size,
            dtype=torch.float32,
        )

        out, state = model.forward_chunkwise(x=self.sample_tensor)
        assert out.shape == self.sample_tensor.shape
        assert state.shape == (self.batch_size, self.hidden_size, self.hidden_size)

        out, state = model.forward_chunkwise(x=self.sample_tensor, state=state)
        assert out.shape == self.sample_tensor.shape
        assert state.shape == (self.batch_size, self.hidden_size, self.hidden_size)

    # TODO: test that the forward passes produce similar results.
