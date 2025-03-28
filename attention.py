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
        """前向传播.
        Args:
            q: Queries张量，形状为[B, L_q, D_q]
            k: Keys张量，形状为[B, L_k, D_k]
            v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
            scale: 缩放因子，一个浮点标量
            attn_mask: Masking张量，形状为[B, L_q, L_k]
        Returns:
            上下文张量和attetention张量
        """
        """
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
            # 给需要mask的地方设置一个负无穷
            attention = attention.masked_fill_(attn_mask, -np.inf)
        # 计算softmax
        attention = self.softmax(attention)   # attention(..., seq_len_q, seq_len_k)

        # 和V做点积
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
        # multi-head attention之后需要做layer norm
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
            key: Keys张量，形状为[B, L_k, D_k]
            value: Values张量，形状为[B, L_v, D_v]，一般来说就是k
            query: Queries张量，形状为[B, L_q, D_q]
            attn_mask: 形状为[B, L_q, L_k]
        """
        # 残差连接
        # if len(key.shape)==2:
        #     key=torch.unsqueeze(key,1)
        #     value=torch.unsqueeze(value, 1)
        #     query = torch.unsqueeze(query, 1)
        # key=torch.unsqueeze(key,1)
        # value = torch.unsqueeze(value, 1)
        # query = torch.unsqueeze(query, 1)
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
class MyModel(nn.Module):
    def __init__(self, model_dim=2048, num_heads=1,LSTM_layers=1,hidden_size=100,dropout=0.0):
        super(MyModel, self).__init__()
        self.att1=MultiHeadAttention(model_dim=model_dim,num_heads=num_heads)
        self.LSTM1 = nn.LSTM(model_dim, hidden_size, LSTM_layers)
        self.flatten1=nn.Flatten()
        self.fc1=nn.Linear(29*hidden_size,512)
        self.fc2=nn.Linear(512,32)
        self.fc3=nn.Linear(32,1)
    def forward(self,x):
        x,_ = self.att1(x,x,x,x)
        x,_ = self.LSTM1(x)
        x = self.flatten1(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

