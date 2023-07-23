import torch

from retentive_network.models.network import RetentiveNetwork


class TestRetentiveNetwork:
    batch_size = 8
    sequence_length = 5
    hidden_size = 32
    number_of_heads = 4
    number_of_layers = 4
    feed_forward_size = 20
    chunk_size = 2
    sample_tensor = torch.randn(batch_size, sequence_length, hidden_size)
    acceptable_diff = 1e-3

    model = RetentiveNetwork(
        number_of_layers=number_of_layers,
        hidden_size=hidden_size,
        number_of_heads=number_of_heads,
        feed_forward_size=feed_forward_size,
        chunk_size=chunk_size,
        half_point_precision=False,
        use_complex_numbers=False,
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
