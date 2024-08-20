from torch import nn, Tensor


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model: int, hidden: int, drop_prob: float = 0.1):
        """
        Initializes a PositionwiseFeedForward object.

        Args:
            d_model (int): The input dimension of the feed forward network.
            hidden (int): The hidden dimension of the feed forward network.
            drop_prob (float, optional): The dropout probability. Defaults to 0.1.

        Returns:
            None
        """
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the PositionwiseFeedForward module.

        Args:
            x (torch.Tensor): The input tensor to be processed.

        Returns:
            The output tensor after applying the linear layers, ReLU activation, and dropout.
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
