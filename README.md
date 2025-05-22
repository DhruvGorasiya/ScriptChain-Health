# Learnable Positional Encoding for Transformers

This project demonstrates the implementation of a learnable positional encoding method using PyTorch, addressing the challenges of positional information in transformer architectures.

## Project Overview

The project consists of two main components:
1. A theoretical analysis of issues with stacking self-attention layers with positional encoding (Check Q1.pdf)
2. A practical implementation of learnable positional encoding using PyTorch

## Implementation Details

### Learnable Positional Encoding
- Implements a custom `LearnablePositionalEncoding` class that learns positional embeddings during training
- Uses PyTorch's `nn.Parameter` to create learnable positional embeddings
- Supports variable sequence lengths and model dimensions

### Time Series Transformer
- Combines learnable positional encoding with a transformer encoder
- Uses PyTorch's built-in transformer encoder for efficient attention mechanism
- Supports batch processing of time series data

## Code Structure

- `Q2.py`: Contains the implementation of the learnable positional encoding and transformer model
- Includes a dummy dataset generation for testing purposes

## Requirements

- Python 3.x
- PyTorch

## Usage

```python
# Example usage
model = TimeSeriesTransformer(seq_len=100, d_model=10, nhead=2)
output = model(input_data)  # input_data shape: [batch_size, seq_len, d_model]
```
