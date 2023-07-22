import torch
from retentive_network.models.network import RetentiveNetwork


class TestRetentiveNetwork:
    batch_size = 8
    sequence_length = 5
    hidden_size = 32
    number_of_heads = 4
    number_of_layers = 4
    feed_forward_size = 20
    half_point_precision = False

    sample_tensor = torch.randn(batch_size, sequence_length, hidden_size)

    model = RetentiveNetwork(
        number_of_layers,
        hidden_size,
        number_of_heads,
        feed_forward_size,
        half_point_precision,
    )

    def test_forward(self):
        out = self.model(self.sample_tensor)
        assert out.shape == self.sample_tensor.shape
        assert out.dtype == torch.float32

    def test_forward_recurrent(self):
        previous_Ses = [
            [
                torch.zeros(
                    self.batch_size,
                    self.model.retention_layers[0].head_size,
                    self.model.retention_layers[0].head_size,
                )
                for _ in range(self.number_of_heads)
            ]
            for _ in range(self.number_of_layers)
        ]

        recurrent_out = []
        for idx in range(self.sequence_length):
            out, s_ns = self.model.forward_recurrent(
                self.sample_tensor[:, idx, :], previous_Ses, idx + 1
            )
            recurrent_out.append(out)
            previous_Ses = s_ns

        recurrent_out = torch.stack(recurrent_out, dim=1)

        assert recurrent_out.shape == torch.Size(
            [self.batch_size, self.sequence_length, self.hidden_size]
        )

        assert len(previous_Ses) == self.number_of_layers

        assert all([isinstance(x, list) for x in previous_Ses])
        assert all([self.number_of_heads == len(x) for x in previous_Ses])

    def test_forward_layers_are_approx_equal(self):
        parallel_out = self.model(self.sample_tensor)
        assert parallel_out.shape == self.sample_tensor.shape

        previous_Ses = [
            [
                torch.zeros(
                    self.batch_size,
                    self.model.retention_layers[0].head_size,
                    self.model.retention_layers[0].head_size,
                )
                for _ in range(self.number_of_heads)
            ]
            for _ in range(self.number_of_layers)
        ]

        recurrent_out = []
        for idx in range(self.sequence_length):
            out, s_ns = self.model.forward_recurrent(
                self.sample_tensor[:, idx, :], previous_Ses, idx + 1
            )
            recurrent_out.append(out)
            previous_Ses = s_ns

        recurrent_out = torch.stack(recurrent_out, dim=1)

        assert recurrent_out.shape == torch.Size(
            [self.batch_size, self.sequence_length, self.hidden_size]
        )
        assert parallel_out.shape == torch.Size(
            [self.batch_size, self.sequence_length, self.hidden_size]
        )

        assert parallel_out.shape == recurrent_out.shape
        assert (parallel_out - recurrent_out).abs().max() < 1e-3
