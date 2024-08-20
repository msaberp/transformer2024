from torch import nn, Tensor, BoolTensor
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, num_head: int, drop_prob: float):
        """
        Initializes the EncoderLayer class.

        Args:
            d_model (int): The number of features in the input data.
            ffn_hidden (int): The number of hidden units in the feed-forward network.
            num_head (int): The number of attention heads.
            drop_prob (float): The dropout probability.
        """
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_head=num_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x: Tensor, src_mask: BoolTensor) -> Tensor:
        """
        Defines the forward pass of the EncoderLayer class.

        Args:
            x (Tensor): The input tensor of shape (batch_size, seq_len, d_model).
            src_mask (BoolTensor): The source mask tensor used to mask out certain positions
                                   in the input tensor, typically to ignore padding tokens.

        Returns:
            Tensor: The output tensor after passing through the self-attention mechanism,
                    feed-forward network, and normalization layers, of shape
                    (batch_size, seq_len, d_model).
        """
        # Self-attention with residual connection and layer normalization
        residual = x
        x = self.attention(query=x, key=x, value=x, mask=src_mask)
        x = self.norm1(residual + self.dropout1(x))

        # Feed-forward network with residual connection and layer normalization
        residual = x
        x = self.ffn(x)
        x = self.norm2(residual + self.dropout2(x))

        return x
