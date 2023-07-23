# RetentiveNetwork
Unofficial codebase for the "Retentive Network: A Successor to Transformer for Large Language Models" paper [https://arxiv.org/pdf/2307.08621.pdf]

The official codebase for RetNet should be made available roughly August 1st, 2023 according to Microsoft here: 

* [https://github.com/microsoft/unilm/tree/master/retnet](https://github.com/microsoft/unilm/tree/master/retnet)
* [https://github.com/microsoft/torchscale](https://github.com/microsoft/torchscale)


## Getting Started

This library can be installed using pip.

```
pip install retentive-network

```

## Example Training

The paper provides three forward passes which can all be used to train this model. However,
the `forward()` and `forward_chunkwise()` are recommended for sample data and sample data 
with long sequences respectively. The `forward_recurrent()` method, while it can be used for
training, the authors suggest using it for faster inference instead.

[example-training-script](https://github.com/zbloss/RetentiveNetwork/blob/main/examples/example_training.py)

```python

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



```