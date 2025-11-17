from torch.utils.data import Dataset
import torch
from data.preprocess import DataPreprocessor
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, data_path, seq_len=24, label_col='Active_Power', 
                 drop_cols=None, preprocessor=None, is_train=True,
                 skip_process=False):  # ← 新增参数
        """
        Args:
            data_path: CSV文件路径
            seq_len: 时间步长度
            label_col: 目标列名
            drop_cols: 要删除的列
            preprocessor: 预处理器实例（验证集应该使用训练集的preprocessor）
            is_train: 是否是训练集
            skip_process: 如果为True，直接加载CSV，不做预处理
        """
        self.seq_len = seq_len
        self.label_col = label_col
        self.is_train = is_train
        
        if skip_process:
            # ===== 直接加载已预处理的数据 =====
            import pandas as pd
            df = pd.read_csv(data_path)
            print(f"直接加载已预处理的数据: {len(df)} 行")
            
            # 如果是验证集，需要传入preprocessor以便后续使用
            if preprocessor is not None:
                self.preprocessor = preprocessor
            
        elif preprocessor is None:
            # ===== 训练集：完整预处理 =====
            print("=" * 50)
            print("创建训练集...")
            print("=" * 50)
            self.preprocessor = DataPreprocessor(data_path)
            df = self.preprocessor.process()
            
        else:
            # ===== 验证集：使用训练集的参数 =====
            print("=" * 50)
            print("创建验证集（使用训练集的标准化参数）...")
            print("=" * 50)
            self.preprocessor = preprocessor
            self.preprocessor.filepath = data_path
            self.preprocessor.load_data()
            self.preprocessor.clean_data()
            self.preprocessor.feature_engineering()
            
            # ===== 使用训练集的scaler进行标准化 =====
            if hasattr(self.preprocessor, 'scaler') and hasattr(self.preprocessor, 'normalize_cols'):
                normalize_cols = self.preprocessor.normalize_cols
                df = self.preprocessor.data
                df[normalize_cols] = self.preprocessor.scaler.transform(df[normalize_cols])
                print(f"✅ 使用训练集的标准化参数转换了 {len(normalize_cols)} 列")
            else:
                raise ValueError("训练集的preprocessor缺少scaler或normalize_cols")
        
        # 确定要删除的列
        drop_cols = drop_cols or ['timestamp']
        
        # 确定特征列
        feature_cols = [c for c in df.columns if c not in drop_cols + [label_col]]
        
        # ===== 关键改进2：检查特征是否包含lag和rolling =====
        lag_features = [c for c in feature_cols if 'lag' in c]
        rolling_features = [c for c in feature_cols if 'rolling' in c]
        print(f"\n特征统计:")
        print(f"  总特征数: {len(feature_cols)}")
        print(f"  滞后特征: {len(lag_features)}")
        print(f"  滑动特征: {len(rolling_features)}")
        print(f"  其他特征: {len(feature_cols) - len(lag_features) - len(rolling_features)}")
        
        if len(lag_features) == 0:
            print("⚠️  警告：没有滞后特征，模型可能学习不到时序依赖！")
        if len(rolling_features) == 0:
            print("⚠️  警告：没有滑动特征，模型可能学习不到趋势！")
        
        # 提取特征和标签
        features = df[feature_cols].values
        labels = df[label_col].values
        
        # ===== 关键改进3：检查数据质量 =====
        if np.isnan(features).any():
            nan_count = np.isnan(features).sum()
            print(f"❌ 错误：特征中有 {nan_count} 个NaN值！")
            raise ValueError("特征中包含NaN，请检查预处理流程")
        
        if np.isinf(features).any():
            print("❌ 错误：特征中有Inf值！")
            raise ValueError("特征中包含Inf")
        
        # 创建滑动窗口
        X_list = []
        y_list = []
        
        for i in range(len(df) - seq_len):
            X_seq = features[i:i+seq_len]
            y_val = labels[i+seq_len]
            
            X_list.append(X_seq)
            y_list.append(y_val)
        
        # 转换为tensor
        self.data = torch.tensor(X_list, dtype=torch.float32)
        self.labels = torch.tensor(y_list, dtype=torch.float32)
        
        self.feature_cols = feature_cols
        
        # ===== 关键改进4：数据统计信息 =====
        print(f"\n数据集创建完成:")
        print(f"  原始数据: {len(df)} 行")
        print(f"  样本数: {len(self.data)}")
        print(f"  特征维度: {seq_len} 时间步 × {len(feature_cols)} 特征")
        print(f"  数据形状: X={self.data.shape}, y={self.labels.shape}")
        print(f"  标签范围: [{self.labels.min():.4f}, {self.labels.max():.4f}]")
        print(f"  标签均值: {self.labels.mean():.4f}, 标准差: {self.labels.std():.4f}")
        print("=" * 50 + "\n")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def get_preprocessor(self):
        """返回预处理器（用于验证集）"""
        return self.preprocessor
    
    def inverse_transform_labels(self, normalized_labels):
        """
        反标准化预测结果
        Args:
            normalized_labels: 标准化后的标签 (numpy array 或 tensor)
        Returns:
            原始尺度的标签
        """
        if torch.is_tensor(normalized_labels):
            normalized_labels = normalized_labels.cpu().numpy()
        
        # 只反标准化目标列
        if self.label_col in self.preprocessor.normalize_cols:
            # 找到目标列在normalize_cols中的索引
            label_idx = self.preprocessor.normalize_cols.index(self.label_col)
            
            # 创建一个和scaler输入形状一致的数组
            n_features = len(self.preprocessor.normalize_cols)
            temp = np.zeros((len(normalized_labels), n_features))
            temp[:, label_idx] = normalized_labels.flatten()
            
            # 反标准化
            inverse = self.preprocessor.scaler.inverse_transform(temp)
            return inverse[:, label_idx]
        else:
            return normalized_labels