import torch
from torch import nn, Tensor


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-12):
        """
        Initializes a LayerNorm module.
        
        Args:
            d_model (int): The number of features in the input data.
            eps (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-12.
        
        Returns:
            None
        """
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (Tensor): The input tensor to be normalized.

        Returns:
            Tensor: The normalized tensor.
        """
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
