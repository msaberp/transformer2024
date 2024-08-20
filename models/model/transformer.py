import torch
from torch import nn
from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_pad_idx: int,
        trg_pad_idx: int,
        trg_sos_idx: int,
        enc_voc_size: int,
        dec_voc_size: int,
        d_model: int,
        num_head: int,
        max_len: int,
        ffn_hidden: int,
        num_layers: int,
        drop_prob: float,
        device: torch.device,
    ):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        self.encoder = Encoder(
            d_model=d_model,
            num_head=num_head,
            max_len=max_len,
            ffn_hidden=ffn_hidden,
            enc_voc_size=enc_voc_size,
            drop_prob=drop_prob,
            num_layers=num_layers,
            device=device,
        )

        self.decoder = Decoder(
            d_model=d_model,
            num_head=num_head,
            max_len=max_len,
            ffn_hidden=ffn_hidden,
            dec_voc_size=dec_voc_size,
            drop_prob=drop_prob,
            num_layers=num_layers,
            device=device,
        )

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer model.

        Args:
            src (torch.Tensor): The source tensor (input) of shape (batch_size, src_seq_len).
            trg (torch.Tensor): The target tensor of shape (batch_size, trg_seq_len).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, trg_seq_len, dec_voc_size).
        """
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src: torch.Tensor) -> torch.BoolTensor:
        """
        Creates a mask for the source input tensor.

        Args:
            src (torch.Tensor): The source tensor of shape (batch_size, src_seq_len).

        Returns:
            torch.BoolTensor: The source mask tensor of shape (batch_size, 1, 1, src_seq_len).
        """
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2).to(self.device)

    def make_trg_mask(self, trg: torch.Tensor) -> torch.BoolTensor:
        """
        Creates a mask for the target input tensor.

        Args:
            trg (torch.Tensor): The target tensor of shape (batch_size, trg_seq_len).

        Returns:
            torch.BoolTensor: The target mask tensor of shape (batch_size, 1, trg_seq_len, trg_seq_len).
        """
        trg_pad_mask = (
            (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3).to(self.device)
        )
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(
            torch.ones((trg_len, trg_len), device=self.device, dtype=torch.bool)
        )
        return trg_pad_mask & trg_sub_mask
