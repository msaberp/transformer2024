import torch
from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(
        self,
        dec_voc_size: int,
        max_len: int,
        d_model: int,
        ffn_hidden: int,
        num_head: int,
        num_layers: int,
        drop_prob: float,
        device: torch.device,
    ):
        """
        Initializes the Decoder class.

        Args:
            dec_voc_size (int): The size of the decoder vocabulary.
            max_len (int): The maximum length of the input sequence.
            d_model (int): The number of features in the input data.
            ffn_hidden (int): The number of hidden units in the feed-forward network.
            num_head (int): The number of attention heads.
            num_layers (int): The number of decoder layers.
            drop_prob (float): The dropout probability.
            device (torch.device): The device to run the model on.

        Returns:
            None
        """
        super().__init__()
        self.emb = TransformerEmbedding(
            d_model=d_model,
            drop_prob=drop_prob,
            max_len=max_len,
            vocab_size=dec_voc_size,
            device=device,
        )

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    ffn_hidden=ffn_hidden,
                    num_head=num_head,
                    drop_prob=drop_prob,
                )
                for _ in range(num_layers)
            ]
        )

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(
        self,
        trg: torch.Tensor,
        enc_src: torch.Tensor,
        trg_mask: torch.BoolTensor,
        src_mask: torch.BoolTensor,
    ):
        """
        Defines the forward pass of the Decoder model.

        Args:
            trg (torch.Tensor): The target input tensor.
            enc_src (torch.Tensor): The encoder output tensor.
            trg_mask (torch.BoolTensor): The target mask tensor.
            src_mask (torch.BoolTensor): The source mask tensor.

        Returns:
            torch.Tensor: The output tensor after applying the decoder layers and linear transformation.
        """
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output
