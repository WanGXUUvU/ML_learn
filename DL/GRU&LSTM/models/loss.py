import torch

import torch.nn as nn
import torch.nn.functional as F


class GRULoss(nn.Module):
    """GRU 模型的损失函数类"""
    
    def __init__(self, loss_type='mse'):
        """
        初始化损失函数
        
        Args:
            loss_type (str): 损失函数类型，支持 'mse', 'mae', 'huber', 'cross_entropy'
        """
        super(GRULoss, self).__init__()
        self.loss_type = loss_type
        
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'mae':
            self.criterion = nn.L1Loss()
        elif loss_type == 'huber':
            self.criterion = nn.HuberLoss()
        elif loss_type == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"不支持的损失函数类型: {loss_type}")
    
    def forward(self, predictions, targets):
        """
        计算损失
        
        Args:
            predictions (torch.Tensor): 模型预测值
            targets (torch.Tensor): 真实标签
            
        Returns:
            torch.Tensor: 损失值
        """
        return self.criterion(predictions, targets)


class SequenceLoss(nn.Module):
    """序列预测的损失函数"""
    
    def __init__(self, loss_type='mse', reduction='mean'):
        super(SequenceLoss, self).__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        
        if loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction=reduction)
        elif loss_type == 'mae':
            self.criterion = nn.L1Loss(reduction=reduction)
        else:
            raise ValueError(f"不支持的损失函数类型: {loss_type}")
    
    def forward(self, predictions, targets, mask=None):
        """
        计算序列损失
        
        Args:
            predictions (torch.Tensor): 预测序列 [batch_size, seq_len, ...]
            targets (torch.Tensor): 目标序列 [batch_size, seq_len, ...]
            mask (torch.Tensor, optional): 掩码 [batch_size, seq_len]
            
        Returns:
            torch.Tensor: 损失值
        """
        loss = self.criterion(predictions, targets)
        
        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            if self.reduction == 'mean':
                loss = loss.sum() / mask.sum()
        
        return loss


class MultiTaskLoss(nn.Module):
    """多任务学习的损失函数"""
    
    def __init__(self, task_weights=None):
        """
        初始化多任务损失
        
        Args:
            task_weights (dict): 各任务的权重，例如 {'task1': 1.0, 'task2': 0.5}
        """
        super(MultiTaskLoss, self).__init__()
        self.task_weights = task_weights or {}
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets, task_type='regression'):
        """
        计算多任务损失
        
        Args:
            predictions (dict): 各任务的预测值
            targets (dict): 各任务的真实标签
            task_type (str or dict): 任务类型
            
        Returns:
            torch.Tensor: 总损失
        """
        total_loss = 0
        losses = {}
        
        for task_name in predictions.keys():
            pred = predictions[task_name]
            target = targets[task_name]
            weight = self.task_weights.get(task_name, 1.0)
            
            if isinstance(task_type, dict):
                t_type = task_type.get(task_name, 'regression')
            else:
                t_type = task_type
            
            if t_type == 'regression':
                loss = self.mse_loss(pred, target)
            elif t_type == 'classification':
                loss = self.ce_loss(pred, target)
            
            losses[task_name] = loss.item()
            total_loss += weight * loss
        
        return total_loss, losses