from torch import nn
from models.layers.scale_dot_product_attention import ScaleDotProductAttention
from torch import Tensor, BoolTensor
from typing import Optional


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_head: int):
        """
        Initializes the MultiHeadAttention module.

        Args:
            d_model (int): The number of features in the input data.
            num_head (int): The number of attention heads.

        Returns:
            None
        """
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.attention = ScaleDotProductAttention()
        self.weight_query = nn.Linear(d_model, d_model)
        self.weight_key = nn.Linear(d_model, d_model)
        self.weight_value = nn.Linear(d_model, d_model)
        self.weight_concat = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[BoolTensor] = None,
    ) -> Tensor:
        """
        Forward pass of the Multi-Head Attention mechanism.

        Args:
            q (torch.Tensor): Query tensor of shape [batch_size, length_query, d_model].
            k (torch.Tensor): Key tensor of shape [batch_size, length_key, d_model].
            v (torch.Tensor): Value tensor of shape [batch_size, length_key, d_model].
            mask (torch.BoolTensor, optional): Mask tensor indicating which elements to mask out.
                If provided, should be of shape [batch_size, 1, length_query, length_key].

        Returns:
            torch.Tensor: Computed attention-weighted values tensor of shape [batch_size, length_query, d_model].
        """
        # 1. dot product with weight matrices
        query, key, value = (
            self.weight_query(query),
            self.weight_key(key),
            self.weight_value(value),
        )

        # 2. split tensor by number of heads
        query, key, value = self.split(query), self.split(key), self.split(value)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(query, key, value, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.weight_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor: Tensor) -> Tensor:
        """
        Splits the input tensor into multiple heads for multi-head attention.

        Args:
            tensor (Tensor): The input tensor to be split.

        Returns:
            Tensor: The split tensor with shape [batch_size, head, length, d_tensor].
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.num_head
        tensor = tensor.view(batch_size, length, self.num_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor: Tensor) -> Tensor:
        """
        Concatenates the input tensor from multi-head attention format to the original format.

        Args:
            tensor (Tensor): The input tensor to be concatenated, with shape [batch_size, head, length, d_tensor].

        Returns:
            Tensor: The concatenated tensor with shape [batch_size, length, d_model].
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
