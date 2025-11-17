import torch.nn as nn
import torch
from .positional_encoding import PositionalEncoding
from .EncoderLayer import EncoderLayer

class Encoder(nn.Module):
    """
    Transformer 编码器
    由多个编码器层堆叠而成
    """
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        """
        参数:
            d_model: 模型维度（例如 512）
            num_heads: 注意力头数量（例如 8）
            d_ff: 前馈网络隐藏层维度（例如 2048）
            num_layers: 编码器层数（例如 6）
            dropout: dropout 概率
        """
        super(Encoder, self).__init__()
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model)
        
        # 堆叠 num_layers 个编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        参数:
            x: 输入张量，shape: (batch_size, seq_len, d_model)
               注意：输入需要先经过 Embedding 层！
            mask: 掩码，用于 padding
        
        返回:
            输出张量，shape: (batch_size, seq_len, d_model)
        """
        # 添加位置编码
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # 通过每一层编码器
        for layer in self.layers:
            x = layer(x, mask)
        
        return x