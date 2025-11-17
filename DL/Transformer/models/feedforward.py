import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络（FFN）
    
    公式: FFN(x) = max(0, xW1 + b1)W2 + b2
    
    这是一个简单的两层全连接网络，对序列中的每个位置独立应用
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        参数:
            d_model: 输入和输出的维度（例如 512）
            d_ff: 隐藏层的维度（通常是 d_model 的 4 倍，例如 2048）
            dropout: dropout 概率
        """
        super(PositionwiseFeedForward, self).__init__()
        
        # 第一层：d_model -> d_ff（扩展维度）
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # 第二层：d_ff -> d_model（恢复维度）
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Dropout 层
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数（ReLU）
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        参数:
            x: 输入张量，shape: (batch_size, seq_len, d_model)
        
        返回:
            输出张量，shape: (batch_size, seq_len, d_model)
        """
        # x -> linear1 -> ReLU -> dropout -> linear2
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        x = self.linear2(x)
        
        return x