import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Computes sinusoidal positional encodings.
    """

    def __init__(self, d_model: int, max_len: int, device: torch.device):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the model.
            max_len (int): The maximum length of the input sequence.
            device (torch.device): The device to use for computations.
        """
        super(PositionalEncoding, self).__init__()

        # Create the positional encodings matrix
        encoding = torch.zeros(max_len, d_model, device=device)
        positions = torch.arange(
            0, max_len, dtype=torch.float, device=device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float, device=device)
            * -(torch.log(torch.tensor(10000.0)) / d_model)
        )

        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)

        # Register the buffer so it's not a learnable parameter but still part of the model state
        self.register_buffer("encoding", encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to the input tensor `x`.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The tensor with positional encoding added,
            of shape (batch_size, seq_len, d_model).
        """
        seq_len = x.size(1)
        # Add positional encoding to the input embeddings
        return self.encoding[:seq_len, :]
