# Language Model for Text Generation

This repository contains the implementation of a language model using PyTorch to generate text. The model is trained on a dataset of Anglo-American poems and is capable of generating new text based on the learned patterns.

## Model Architecture

- **Embedding Layers**: Token and positional embeddings.
- **Transformer Blocks**: Multiple transformer blocks with multi-head attention and feedforward networks.
- **Output Layer**: Linear layer to map the final embeddings to vocabulary size.

## Key Features

- **Model**: Transformer-based language model.
- **Training**: Custom training loop with evaluation and loss calculation.
- **Text Generation**: Ability to generate text based on the dataset used.
