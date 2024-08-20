from torch import nn
import torch

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        drop_prob: float,
        device: torch.device,
    ):
        """
        Initializes the TransformerEmbedding module.

        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The dimensionality of the model.
            max_len (int): The maximum length of the input sequence.
            drop_prob (float): The dropout probability.
            device (torch.device): The device to use for computations.

        Returns:
            None
        """
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_embedding = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the TransformerEmbedding module.

        This function takes in an input tensor `x` and returns the sum of the token embedding and positional embedding of `x`,
        after applying dropout to the result.

        Args:
            x (Tensor): The input tensor to be embedded.

        Returns:
            Tensor: The embedded input tensor after applying dropout.
        """
        token_emb = self.token_embedding(x)
        positional_emb = self.positional_embedding(x)
        return self.drop_out(token_emb + positional_emb)
