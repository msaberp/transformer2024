from torch import nn


class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size: int, d_model: int):
        """
        Initializes the TokenEmbedding module.

        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The dimensionality of the model.

        Returns:
            None
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
