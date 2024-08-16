from typing import Optional
import math
from torch import nn
from torch import Tensor, BoolTensor


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury (encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[BoolTensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the Scale Dot-Product Attention mechanism.

        Args:
            query (torch.Tensor): Query tensor of shape [batch_size, head, length_query, d_tensor].
            key (torch.Tensor): Key tensor of shape [batch_size, head, length_key, d_tensor].
            value (torch.Tensor): Value tensor of shape [batch_size, head, length_key, d_tensor].
            mask (torch.BoolTensor, optional): Mask tensor indicating which elements to mask out.
                If provided, should be of shape [batch_size, 1, length_query, length_key].

        Returns:
            torch.Tensor, torch.Tensor:
            - Computed attention-weighted values tensor of shape [batch_size, head, length_query, d_tensor].
            - Attention scores tensor of shape [batch_size, head, length_query, length_key].
        """
        # input dimentions: [batch_size, head, length, d_tensor]
        d_tensor = key.size()[-1]

        key_transpose = key.transpose(2, 3)
        scaling_factor = 1 / math.sqrt(d_tensor)
        scaled_dot_product = (query @ key_transpose) * scaling_factor

        if mask is not None:
            scaled_dot_product = scaled_dot_product.masked_fill(mask == 0, -10000)

        score = self.softmax(scaled_dot_product)
        value = score @ value

        return value, score
