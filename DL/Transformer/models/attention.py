import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力机制
    Attention(Q, K, V) = softmax(Q * K^T / √d_k) * V
    """
    def __init__(self, d_k):
        """
        参数:
            d_k: 每个头的维度（d_model / num_heads）
        """
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    
    def forward(self, Q, K, V, mask=None):
        """
        参数:
            Q: Query，形状 (batch_size, num_heads, seq_len, d_k)
            K: Key，形状 (batch_size, num_heads, seq_len, d_k)
            V: Value，形状 (batch_size, num_heads, seq_len, d_k)
            mask: 掩码，可选参数
        
        返回:
            output: 注意力输出
            attention_weights: 注意力权重（用于可视化）
        """
        
        # 第一步：计算注意力分数 (Q * K^T)
        scores = Q @ K.transpose(-2, -1)  # (batch, num_heads, seq_len, seq_len)
        
        # 第二步：缩放 (除以 √d_k)
        scores = scores / math.sqrt(self.d_k)
        
        # 第三步：应用掩码（如果有的话）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 第四步：应用 softmax 得到注意力权重
        attention_weights = torch.softmax(scores, dim=-1)  # (batch, num_heads, seq_len, seq_len)
        
        # 第五步：乘以 Value
        output = attention_weights @ V  # (batch, num_heads, seq_len, d_k)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    将输入分成多个头，每个头独立进行注意力计算，然后拼接结果
    """
    def __init__(self, d_model, num_heads):
        """
        参数:
            d_model: 模型的维度（例如 512）
            num_heads: 注意力头的数量（例如 8）
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 定义线性变换层
        # 用于将输入投影到 Q、K、V 空间
        self.W_q = nn.Linear(d_model, d_model)  # Query 投影
        self.W_k = nn.Linear(d_model, d_model)  # Key 投影
        self.W_v = nn.Linear(d_model, d_model)  # Value 投影
        
        # 输出投影层（用于拼接多个头的输出后再投影）
        self.W_o = nn.Linear(d_model, d_model)
        
        # 缩放点积注意力
        self.attention = ScaledDotProductAttention(self.d_k)
    
    def forward(self, Q, K, V, mask=None):
        """
        参数:
            Q: Query，形状 (batch_size, seq_len_q, d_model)
            K: Key，形状 (batch_size, seq_len_k, d_model)
            V: Value，形状 (batch_size, seq_len_v, d_model)
            mask: 掩码，可选参数
        
        返回:
            output: 多头注意力输出，形状 (batch_size, seq_len_q, d_model)
            attention_weights: 注意力权重
        """
        batch_size = Q.size(0)
        
        # ===== 第一步：线性投影并重新形状 =====
        # 投影到 Q、K、V 空间
        Q = self.W_q(Q)  # (batch_size, seq_len_q, d_model)
        K = self.W_k(K)  # (batch_size, seq_len_k, d_model)
        V = self.W_v(V)  # (batch_size, seq_len_v, d_model)
        
        # 重新形状以分离多个头
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k)
        K = K.view(batch_size, -1, self.num_heads, self.d_k)
        V = V.view(batch_size, -1, self.num_heads, self.d_k)
        
        # 转置以便进行注意力计算
        # (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # ===== 第二步：计算注意力 =====
        # 对每个头进行注意力计算
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        # attn_output: (batch_size, num_heads, seq_len_q, d_k)
        
        # ===== 第三步：拼接多个头 =====
        # 转置回来
        # (batch_size, num_heads, seq_len_q, d_k) -> (batch_size, seq_len_q, num_heads, d_k)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # 拼接所有头
        # (batch_size, seq_len_q, num_heads, d_k) -> (batch_size, seq_len_q, d_model)
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # ===== 第四步：最终的线性投影 =====
        output = self.W_o(attn_output)  # (batch_size, seq_len_q, d_model)
        
        return output, attention_weights