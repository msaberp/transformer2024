### WARNING
This code was written in 2019, and I was not very familiar with transformer model in that time.
So don't trust this code too much. Currently I am not managing this code well, so please open pull requests if you find bugs in the code and want to fix.

# Transformer
My own implementation Transformer model (Attention is All You Need - Google Brain, 2017)
<br><br>
![model](image/model.png)
<br><br>

## 1. Implementations

### 1.1 Positional Encoding

![model](image/positional_encoding.jpg)
   
    
```python
class PositionalEncoding(nn.Module):
    """
    Computes sinusoidal positional encodings.
    """

    def __init__(self, d_model: int, max_len: int, device: torch.device):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the model.
            max_len (int): The maximum length of the input sequence.
            device (torch.device): The device to use for computations.
        """
        super(PositionalEncoding, self).__init__()

        # Create the positional encodings matrix
        encoding = torch.zeros(max_len, d_model, device=device)
        positions = torch.arange(
            0, max_len, dtype=torch.float, device=device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float, device=device)
            * -(torch.log(torch.tensor(10000.0)) / d_model)
        )

        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)

        # Register the buffer so it's not a learnable parameter but still part of the model state
        self.register_buffer("encoding", encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to the input tensor `x`.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The tensor with positional encoding added,
            of shape (batch_size, seq_len, d_model).
        """
        seq_len = x.size(1)
        # Add positional encoding to the input embeddings
        return self.encoding[:seq_len, :]        
```
<br><br>

### 1.2 Multi-Head Attention


![model](image/multi_head_attention.jpg)

```python
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
```
<br><br>

### 1.3 Scale Dot Product Attention

![model](image/scale_dot_product_attention.jpg)

```python
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
```
<br><br>

### 1.5 Positionwise Feed Forward

![model](image/positionwise_feed_forward.jpg)
    
```python

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
```
<br><br>

### 1.6 Encoder & Decoder Structure

![model](image/enc_dec.jpg)
    
```python
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
```
<br>

```python
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
```
<br>

```python
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
```
<br>

```python        
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
```
<br><br>

## 2. Experiments

I use Multi30K Dataset to train and evaluate model <br>
You can check detail of dataset [here](https://arxiv.org/abs/1605.00459) <br>
I follow original paper's parameter settings. (below) <br>

![conf](image/transformer-model-size.jpg)
### 2.1 Model Specification

* total parameters = 55,207,087
* model size = 215.7MB
* lr scheduling : ReduceLROnPlateau

#### 2.1.1 configuration

* batch_size = 128
* max_len = 256
* d_model = 512
* num_layers = 6
* num_heads = 8
* ffn_hidden = 2048
* drop_prob = 0.1
* init_lr = 0.1
* factor = 0.9
* patience = 10
* warmup = 100
* adam_eps = 5e-9
* epoch = 1000
* clip = 1
* weight_decay = 5e-4
<br><br>

### 2.2 Training Result

![image](saved/transformer-base/train_result.jpg)
* Minimum Training Loss = 2.852672759656864
* Minimum Validation Loss = 3.2048025131225586 
<br><br>

| Model | Dataset | BLEU Score |
|:---:|:---:|:---:|
| Original Paper's | WMT14 EN-DE | 25.8 |
| My Implementation | Multi30K EN-DE | 26.4 |

<br><br>


## 3. Reference
- [Attention is All You Need, 2017 - Google](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer - Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
- [Data & Optimization Code Reference - Bentrevett](https://github.com/bentrevett/pytorch-seq2seq/)

<br><br>

## 4. Licence
    Copyright 2019 Hyunwoong Ko.
    
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
    http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
