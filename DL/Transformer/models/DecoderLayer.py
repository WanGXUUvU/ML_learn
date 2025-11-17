import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    """
    Transformer 解码器层
    
    结构：
    1. 掩码多头自注意力层 + 残差连接 + LayerNorm
    2. 交叉注意力层 + 残差连接 + LayerNorm
    3. 前馈网络 + 残差连接 + LayerNorm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        参数:
            d_model: 模型维度（例如 512）
            num_heads: 注意力头数量（例如 8）
            d_ff: 前馈网络隐藏层维度（例如 2048）
            dropout: dropout 概率
        """
        super(DecoderLayer, self).__init__()
        
        # 子层1：掩码多头自注意力（用于解码器自己的序列）
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # 子层2：交叉注意力（用于关注编码器的输出）
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        
        # 子层3：前馈网络
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # LayerNorm 层（每个子层后都需要）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout 层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        参数:
            x: 解码器输入，shape: (batch_size, tgt_seq_len, d_model)
            enc_output: 编码器输出，shape: (batch_size, src_seq_len, d_model)
            src_mask: 源序列掩码（用于padding）
            tgt_mask: 目标序列掩码（用于padding + 防止看到未来信息）
        
        返回:
            输出张量，shape: (batch_size, tgt_seq_len, d_model)
        """
        # ===== 子层1：掩码自注意力 =====
        # Q、K、V 都来自解码器输入 x
        # 使用 tgt_mask 防止看到未来的词
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        
        # 残差连接 + Dropout + LayerNorm
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # ===== 子层2：交叉注意力 =====
        # Q 来自解码器（当前位置想要什么信息）
        # K、V 来自编码器输出（源序列提供信息）
        cross_attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        
        # 残差连接 + Dropout + LayerNorm
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # ===== 子层3：前馈网络 =====
        ffn_output = self.ffn(x)
        
        # 残差连接 + Dropout + LayerNorm
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x