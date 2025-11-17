import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    位置编码：为输入序列添加位置信息
    使用正弦和余弦函数生成位置编码
    """
    def __init__(self, d_model, max_len=5000):
        """
        参数:
            d_model: 词嵌入的维度（例如 512）
            max_len: 序列的最大长度
        """
        super(PositionalEncoding, self).__init__()
        
        # 创建一个 (max_len, d_model) 的矩阵来存储位置编码
        pe = torch.zeros(max_len, d_model)
        
        # 创建位置索引 [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).unsqueeze(1)  # shape: (max_len, 1)
        
        # 计算分母部分：10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                            -(math.log(10000.0) / d_model))
        
        # 偶数位置使用 sin，奇数位置使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列
        
        # 添加 batch 维度: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer，但不作为可训练参数（不需要梯度，但需要保存到模型）
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        参数:
            x: 输入张量，shape: (batch_size, seq_len, d_model)
        返回:
            添加了位置编码的张量
        """
        # 将位置编码加到输入上
        x = x + self.pe[:, :x.size(1), :]
        return x