import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        Args:
            q: Queries tensor, shape [B, L_q, D_q]
            k: Keys tensor, shape [B, L_k, D_k]
            v: Values ​​tensor, shape [B, L_v, D_v], generally k
            scale: scaling factor, a floating point scalar
            attn_mask: Masking tensor, shape [B, L_q, L_k]
        Returns:
            Context tensor and attention tensor

        torch.bmm(input, mat2, out=None) → Tensor
        Performs a batch matrix-matrix product of matrices stored in input and mat2.
        input and mat2 must be 3-D tensors each containing the same number of matrices.
        If input is a (b \times n \times m)(b×n×m) tensor, mat2 is a (b \times m \times p)(b×m×p) tensor, 
        out will be a (b \times n \times p)(b×n×p) tensor.
        """
        #q:[B, L_q, D_q]
        #k.t:[B, D_k, L_k]
        #L_q=L_k,D_q=D_k
        attention = torch.matmul(q, k.transpose(-1,-2))#[B,L_q,L_k]
        # attention [B, L_q, L_k]
        if scale:
            attention = attention * scale
        if attn_mask is not None:
          
            attention = attention.masked_fill_(attn_mask, -np.inf)
        # softmax
        attention = self.softmax(attention)   # attention(..., seq_len_q, seq_len_k)

        # Do the dot product with V
        context = torch.matmul(attention, v)  # context(..., seq_len_q, depth_v)
        return context, attention
class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.model_dim = model_dim
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def split_heads(self, x, batch_size):
        """
            Split the last dimension into (num_heads, depth).
            Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = x.view(batch_size, -1, self.num_heads, self.dim_per_head)
        return x.permute([0, 2, 1, 3])

    def forward(self, key, value, query, attn_mask=None):
        """
        Args:
            key: Keys tensor, shape is [B, L_k, D_k]
            value: Values: tensor, shape is [B, L_v, D_v], generally k
            query: Queries tensor, shape is [B, L_q, D_q]
            attn_mask: shape is [B, L_q, L_k]
        """
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        query = self.split_heads(query, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        key = self.split_heads(key, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        value = self.split_heads(value, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # if attn_mask is not None:
        #     attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale,
            attn_mask)

        # context[batch_size * num_heads, seq_len, dim_per_head]
        # attention[batch_size * num_heads, L_q, L_k]

        # concat heads
        context = context.permute([0, 2, 1, 3])
        context = context.reshape(batch_size, -1, self.model_dim)
        # context[batch_size, seq_len_q, model_dim]

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)
        output=torch.squeeze(output,1)
        return output, attention

