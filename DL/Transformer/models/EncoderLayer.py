import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    """
    Transformer 编码器层
    
    结构：
    1. 多头自注意力层 + 残差连接 + LayerNorm
    2. 前馈网络 + 残差连接 + LayerNorm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        参数:
            d_model: 模型维度（例如 512）
            num_heads: 注意力头数量（例如 8）
            d_ff: 前馈网络隐藏层维度（例如 2048）
            dropout: dropout 概率
        """
        super(EncoderLayer, self).__init__()
        
        # 子层1：多头自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # 子层2：前馈网络
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # LayerNorm 层（用于残差连接后的归一化）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout 层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        参数:
            x: 输入张量，shape: (batch_size, seq_len, d_model)
            mask: 掩码（可选），用于 padding
        
        返回:
            输出张量，shape: (batch_size, seq_len, d_model)
        """
        # ===== 子层1：多头自注意力 =====
        # 1. 自注意力计算
        attn_output, _ = self.self_attn(x, x, x, mask)
        
        # 2. 残差连接 + Dropout + LayerNorm
        x = self.norm1(x + self.dropout(attn_output))
        
        # ===== 子层2：前馈网络 =====
        # 1. 前馈网络
        ffn_output = self.ffn(x)
        
        # 2. 残差连接 + Dropout + LayerNorm
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x