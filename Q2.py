import torch          
import torch.nn as nn

batch_size = 64       # Number of sequences in each batch
seq_len = 100         # Length of each sequence (time steps)
num_features = 10     # Number of features at each time step

# Generate random data
# torch.randn creates tensor with random values from normal distribution
# Shape: [batch_size, seq_len, num_features] = [64, 100, 10]
dummy_data = torch.randn(batch_size, seq_len, num_features)

# Define Learnable Positional Encoding class
class LearnablePositionalEncoding(nn.Module):
    """
    A learnable positional encoding module for transformer architectures.
    
    This class implements positional encoding as learnable parameters instead of
    fixed sinusoidal encodings. The positional embeddings are learned during
    training, allowing the model to adapt to the specific positional patterns
    in the data.

    Attributes:
        pos_embed (nn.Parameter): Learnable positional embeddings of shape
            [1, seq_len, d_model]. The first dimension of 1 allows for
            broadcasting across the batch.

    Args:
        seq_len (int): Maximum sequence length to support.
        d_model (int): Dimension of the model's hidden states.

    Shape:
        - Input: [batch_size, seq_len, d_model]
        - Output: [batch_size, seq_len, d_model]
    """

    def __init__(self, seq_len, d_model):
        """
        Initialize the learnable positional encoding.

        Args:
            seq_len (int): Maximum sequence length to support.
            d_model (int): Dimension of the model's hidden states.
        """
        super().__init__()
        # Create learnable parameter for positional embeddings
        # Shape: [1, seq_len, d_model] - 1 for batch dimension
        # torch.randn initializes with random values
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))

    def forward(self, x):
        """
        Forward pass of the positional encoding.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor: Input tensor with positional encodings added.
        """
        # x shape: [batch_size, seq_len, d_model]
        # self.pos_embed shape: [1, seq_len, d_model]
        # Broadcasting adds positional embeddings to each sequence in batch
        # [:x.size(1), :] ensures we only use embeddings up to sequence length
        return x + self.pos_embed[:, :x.size(1), :]

# Define Transformer Model class
class TimeSeriesTransformer(nn.Module):
    """
    A transformer model for time series data with learnable positional encoding.

    This class combines learnable positional encoding with a transformer encoder
    to process time series data. It uses PyTorch's built-in transformer encoder
    for efficient implementation of the attention mechanism.

    Attributes:
        pos_encoder (LearnablePositionalEncoding): Module for adding positional
            information to the input.
        transformer (nn.TransformerEncoder): PyTorch's transformer encoder for
            processing the sequences.

    Args:
        seq_len (int): Maximum sequence length to support.
        d_model (int): Dimension of the model's hidden states.
        nhead (int): Number of attention heads in the transformer.

    Shape:
        - Input: [batch_size, seq_len, d_model]
        - Output: [batch_size, seq_len, d_model]
    """

    def __init__(self, seq_len, d_model, nhead):
        """
        Initialize the time series transformer model.

        Args:
            seq_len (int): Maximum sequence length to support.
            d_model (int): Dimension of the model's hidden states.
            nhead (int): Number of attention heads in the transformer.
        """
        super().__init__()
        # Initialize positional encoder
        self.pos_encoder = LearnablePositionalEncoding(seq_len, d_model)
        
        # Create transformer encoder
        # nn.TransformerEncoderLayer parameters:
        # - d_model: dimension of model
        # - nhead: number of attention heads
        # - batch_first=True: input shape is [batch, seq, features]
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=1
        )

    def forward(self, x):
        """
        Forward pass of the transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor: Transformed tensor of shape [batch_size, seq_len, d_model].
        """
        # Step 1: Add positional encoding to input
        x = self.pos_encoder(x) 
        # Step 2: Pass through transformer
        return self.transformer(x) 


model = TimeSeriesTransformer(seq_len=seq_len, d_model=num_features, nhead=2)
output = model(dummy_data)

# Print shapes to verify
print("Input shape:", dummy_data.shape)  # Should be [64, 100, 10]
print("Output shape:", output.shape)     # Should be [64, 100, 10]