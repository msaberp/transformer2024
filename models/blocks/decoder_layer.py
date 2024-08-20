from torch import nn, Tensor, BoolTensor
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, num_head: int, drop_prob: float):
        """
        Initializes the DecoderLayer class.

        Args:
            d_model (int): The number of features in the input data.
            ffn_hidden (int): The number of hidden units in the feed-forward network.
            num_head (int): The number of attention heads.
            drop_prob (float): The dropout probability.
        """
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_head=num_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, num_head=num_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(
        self, dec: Tensor, enc: Tensor, trg_mask: BoolTensor, src_mask: BoolTensor
    ) -> Tensor:
        """
        Defines the forward pass of the DecoderLayer.

        Args:
            dec (Tensor): The decoder input tensor of shape (batch_size, trg_seq_len, d_model).
            enc (Tensor): The encoder output tensor of shape (batch_size, src_seq_len, d_model).
            trg_mask (BoolTensor): The target mask tensor, used to mask out certain positions in the
                               decoder input sequence.
            src_mask (BoolTensor): The source mask tensor, used to mask out certain positions in the
                               encoder output sequence.

        Returns:
            Tensor: The output tensor after applying self-attention, encoder-decoder attention,
                    and a feed-forward network, with residual connections and normalization.
        """
        # Self-attention with residual connection and layer normalization
        residual = dec
        x = self.self_attention(query=dec, key=dec, value=dec, mask=trg_mask)
        x = self.norm1(residual + self.dropout1(x))

        # Encoder-decoder attention with residual connection and layer normalization
        if enc is not None:
            residual = x
            x = self.enc_dec_attention(query=x, key=enc, value=enc, mask=src_mask)
            x = self.norm2(residual + self.dropout2(x))

        # Feed-forward network with residual connection and layer normalization
        residual = x
        x = self.ffn(x)
        x = self.norm3(residual + self.dropout3(x))

        return x
