import torch
import torch.nn as nn
from .Encoder import Encoder
from .Decoder import Decoder

class Transformer(nn.Module):
    """
    完整的 Transformer 模型
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                 num_heads=8, d_ff=2048, num_layers=6, dropout=0.1):
        """
        参数:
            src_vocab_size: 源语言词表大小
            tgt_vocab_size: 目标语言词表大小
            d_model: 模型维度
            num_heads: 注意力头数量
            d_ff: 前馈网络隐藏层维度
            num_layers: 编码器和解码器的层数
            dropout: dropout 概率
        """
        super(Transformer, self).__init__()
        
        # 嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 嵌入缩放（论文建议）
        self.embedding_scale = d_model ** 0.5
        
        # 编码器和解码器
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)
        
        # 输出线性层（特征 → 词概率）
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        参数:
            src: 源序列，shape: (batch_size, src_seq_len)
            tgt: 目标序列，shape: (batch_size, tgt_seq_len)
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码（包含 padding 掩码和因果掩码）
        
        返回:
            logits，shape: (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # 嵌入 + 缩放
        src_embed = self.src_embedding(src) * self.embedding_scale
        tgt_embed = self.tgt_embedding(tgt) * self.embedding_scale
        
        # 编码器
        enc_output = self.encoder(src_embed, src_mask)
        
        # 解码器
        dec_output = self.decoder(tgt_embed, enc_output, src_mask, tgt_mask)
        
        # 输出层
        logits = self.output_linear(dec_output)
        
        return logits