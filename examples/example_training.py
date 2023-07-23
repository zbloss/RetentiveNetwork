import torch
from retentive_network.models.clm import RetentiveNetworkCLM

batch_size = 8
sequence_length = 5
hidden_size = 32
number_of_heads = 4
number_of_layers = 4
feed_forward_size = 20
chunk_size = 2
samples = 100
vocab_size = 100

sample_data = torch.randint(0, vocab_size, (samples, batch_size, sequence_length))
labels = torch.randint(0, sequence_length, (samples,batch_size))

model = RetentiveNetworkCLM(
    number_of_layers=number_of_layers,
    hidden_size=hidden_size,
    number_of_heads=number_of_heads,
    feed_forward_size=feed_forward_size,
    vocab_size=vocab_size,
    chunk_size=chunk_size,
    softmax=True
)


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

initial_out = model(sample_data[0])
initial_loss = criterion(initial_out, labels[0])

for sample, label in zip(sample_data, labels):
    optimizer.zero_grad()

    out = model(sample)
    loss = criterion(out, label)
    loss.backward()
    optimizer.step()

# state = None
# for sample in sample_data:

#     # Standard forward pass
#     out = model(sample)

#     # Recurrent forward pass, faster for inference
#     # out, recurrent_state = model.forward_recurrent(sample, recurrent_state, 1)

#     # Chunkwise forward pass, faster for training on large sequences
#     out, state = model.forward_chunkwise(sample)

