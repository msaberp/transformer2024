import torch
from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):

    def __init__(
        self,
        enc_voc_size: int,
        max_len: int,
        d_model: int,
        ffn_hidden: int,
        num_head: int,
        num_layers: int,
        drop_prob: float,
        device: torch.device,
    ):
        """
        Initializes the Encoder class.

        Args:
            enc_voc_size (int): The size of the encoder vocabulary.
            max_len (int): The maximum length of the input sequence.
            d_model (int): The number of features in the input data.
            ffn_hidden (int): The number of hidden units in the feed-forward network.
            num_head (int): The number of attention heads.
            num_layers (int): The number of encoder layers.
            drop_prob (float): The dropout probability.
            device (torch.device): The device to run the model on.

        Returns:
            None
        """
        super().__init__()
        self.emb = TransformerEmbedding(
            d_model=d_model,
            max_len=max_len,
            vocab_size=enc_voc_size,
            drop_prob=drop_prob,
            device=device,
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    ffn_hidden=ffn_hidden,
                    num_head=num_head,
                    drop_prob=drop_prob,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, src_mask: torch.BoolTensor) -> torch.Tensor:
        """
        Defines the forward pass of the Encoder model.

        Args:
            x (torch.Tensor): The input tensor.
            src_mask (torch.BoolTensor): The source mask tensor.

        Returns:
            torch.Tensor: The output tensor after applying the embedding and encoder layers.
        """
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x
