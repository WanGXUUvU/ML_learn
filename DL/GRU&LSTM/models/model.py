from torch import nn
import torch

class GRUPredictor(nn.Module):
    """
    GRU预测模型 - 适合简单的回归预测任务
    例如: 股票预测、销量预测、温度预测等
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.2):
        """
        Args:
            input_size: 输入特征维度 (例如: 5个特征)
            hidden_size: GRU隐藏层维度 (推荐: 64-128)
            num_layers: GRU层数 (推荐: 2-3)
            output_size: 输出维度 (单步预测=1, 多步预测=N)
            dropout: dropout比例 (推荐: 0.2-0.3)
        """
        super(GRUPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU层
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: (batch_size, seq_len, input_size)
               例如: (32, 60, 5) 表示32个样本,每个样本60个时间步,5个特征
        Returns:
            out: (batch_size, output_size)
               例如: (32, 1) 表示32个样本的预测值
        """
        # GRU前向传播
        gru_out, h_n = self.gru(x)
        # gru_out: (batch_size, seq_len, hidden_size)
        # h_n: (num_layers, batch_size, hidden_size)
        
        # 取最后一个时间步的输出
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_size)
        
        # 通过全连接层得到预测值
        prediction = self.fc(last_output)  # (batch_size, output_size)
        
        return prediction


class LSTMPredictor(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size=1,dropout=0.2):
        # super()的作用是调用父类的方法，初始化父类
        super(LSTMPredictor,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm=nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers>1 else 0
        )
        self.fc=nn.Linear(hidden_size,output_size)

    def forward(self,x):
        # LSTM前向传播
        lstm_out,(h_n,c_n)=self.lstm(x)
        # lstm_out: (batch_size, seq_len, hidden_size)
        # h_n: (num_layers, batch_size, hidden_size)
        # c_n: (num_layers, batch_size, hidden_size)

        # 取最后一个时间步的输出
        last_output=lstm_out[:,-1,:]  # (batch_size, hidden_size)

        # 通过全连接层得到预测值
        prediction=self.fc(last_output)  # (batch_size, output_size)

        return prediction