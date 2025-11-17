import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def load_data(self):
        """Load data from a CSV file."""
        if self.filepath == "6-Site_3-C.csv":
            self.data = pd.read_csv(self.filepath, encoding='gbk')
            self.data.drop(index=0, inplace=True)
        else:
            self.data = pd.read_csv(self.filepath)
        print("Data loaded successfully.")
        return self.data
    
    def clean_data(self, outlier_method='iqr', outlier_action='interpolate'):
        """Clean the data by handling missing values, duplicates, and outliers."""
        if self.data is not None:
            # 处理缺失值 - 使用新版pandas语法
            self.data.ffill(inplace=True)
            
            # 处理重复值
            self.data.drop_duplicates(keep='last', inplace=True)
            
            # 处理异常值
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if outlier_method == 'iqr':
                    Q1 = self.data[col].quantile(0.25)
                    Q3 = self.data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                elif outlier_method == 'zscore':
                    rolling_mean = self.data[col].rolling(window=10, center=True).mean()
                    rolling_std = self.data[col].rolling(window=10, center=True).std()
                    lower_bound = rolling_mean - 3 * rolling_std
                    upper_bound = rolling_mean + 3 * rolling_std
                    
                elif outlier_method == 'percentile':
                    lower_bound = self.data[col].quantile(0.01)
                    upper_bound = self.data[col].quantile(0.99)
                
                outliers = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                
                if outlier_action == 'interpolate':
                    self.data.loc[outliers, col] = np.nan
                    self.data[col] = self.data[col].interpolate(method='linear')
                    
                elif outlier_action == 'cap':
                    self.data.loc[self.data[col] < lower_bound, col] = lower_bound
                    self.data.loc[self.data[col] > upper_bound, col] = upper_bound
                    
                elif outlier_action == 'rolling_median':
                    rolling_median = self.data[col].rolling(window=5, center=True).median()
                    self.data.loc[outliers, col] = rolling_median[outliers]
                    
                elif outlier_action == 'forward_fill':
                    self.data.loc[outliers, col] = np.nan
                    self.data[col] = self.data[col].ffill()
        
            print("Time series data cleaned successfully.")
        else:
            print("Data not loaded. Please load the data first.")
        
        return self.data

    def feature_engineering(self):
        """对时间序列数据进行特征工程"""
        if self.data is not None:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data['year'] = self.data['timestamp'].dt.year
            self.data['month'] = self.data['timestamp'].dt.month
            self.data['day'] = self.data['timestamp'].dt.day
            self.data['hour'] = self.data['timestamp'].dt.hour
            self.data['minute'] = self.data['timestamp'].dt.minute
            self.data['dayofweek'] = self.data['timestamp'].dt.dayofweek
            self.data['is_weekend'] = (self.data['dayofweek'] >= 5).astype(int)
            self.data['minutes of day'] = self.data['hour'] * 60 + self.data['minute']
            
            # 周期性特征转换
            self.data['hour_sin'] = np.sin(2 * np.pi * self.data['hour'] / 24)
            self.data['hour_cos'] = np.cos(2 * np.pi * self.data['hour'] / 24)
            self.data['month_sin'] = np.sin(2 * np.pi * self.data['month'] / 12)
            self.data['month_cos'] = np.cos(2 * np.pi * self.data['month'] / 12)
            # 滞后特征
            for lag in [1, 2, 3, 6, 12, 24]:
                self.data[f'Active_Power_lag_{lag}'] = self.data['Active_Power'].shift(lag)

            # 滚动窗口特征
            for window in [3, 6, 12, 24]:
                self.data[f'Active_Power_roll_mean_{window}'] = self.data['Active_Power'].rolling(window=window).mean()
                self.data[f'Active_Power_roll_std_{window}'] = self.data['Active_Power'].rolling(window=window).std()
            
            self.data.dropna(inplace=True)
            self.data.reset_index(drop=True, inplace=True)
            print("Feature engineering completed successfully.")
        else:
            print("Data not loaded. Please load the data first.")
        
        return self.data

    def normalize_data(self, method='robust'):
        """
        Normalize numerical columns in the dataset.
        """
        if self.data is not None:
            # 排除时间相关特征
            exclude_cols = ['year', 'month', 'day', 'hour', 'minute', 'dayofweek', 
                          'is_weekend', 'minutes of day', 
                          'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
            
            # 获取所有数值列
            all_numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            
            # 只标准化原始特征列
            normalize_cols = [col for col in all_numeric_cols if col not in exclude_cols]
            
            print(f"准备标准化的列: {normalize_cols}")
            
            if method == 'zscore':
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                
            elif method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                
            elif method == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            
            # 只标准化指定的列
            if len(normalize_cols) > 0:
                self.data[normalize_cols] = scaler.fit_transform(self.data[normalize_cols])
            
            # 保存scaler用于预测时的逆变换
            self.scaler = scaler
            self.normalize_cols = normalize_cols
            
            print(f"Data normalized successfully using {method} method.")
            print(f"标准化后的数据范围检查:")
            for col in normalize_cols[:3]:  # 检查前3列
                print(f"  {col}: min={self.data[col].min():.4f}, max={self.data[col].max():.4f}, mean={self.data[col].mean():.4f}")
        else:
            print("Data not loaded. Please load the data first.")

        return self.data
    
    def process(self, outlier_method='iqr', outlier_action='interpolate', normalize_method='robust'):
        """一键式数据预处理 - 修改顺序"""
        self.load_data()
        # 先清洗和标准化原始数据，再做特征工程
        self.clean_data(outlier_method=outlier_method, outlier_action=outlier_action)
        self.feature_engineering()
        self.normalize_data(method=normalize_method)
  
        
        # ===== 新增：最终检查是否有NaN或Inf =====
        if self.data.isnull().any().any():
            print("⚠️ 警告：数据中仍有NaN值！")
            print(self.data.isnull().sum()[self.data.isnull().sum() > 0])
            # 最后再填充一次
            self.data.ffill(inplace=True)
            self.data.bfill(inplace=True)
        
        if np.isinf(self.data.select_dtypes(include=[np.number]).values).any():
            print("⚠️ 警告：数据中有Inf值！")
            # 替换Inf为NaN，然后填充
            self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.data.ffill(inplace=True)
            self.data.bfill(inplace=True)
        
        print("数据预处理完成，最终检查通过。")
        return self.data
    
    def inverse_normalize(self, data):
        """反标准化，用于恢复预测结果的原始尺度"""
        if hasattr(self, 'scaler'):
            return self.scaler.inverse_transform(data)
        else:
            print("Scaler not found. Please normalize data first.")
            return data

    def get_data(self):
        """Return the processed data."""
        return self.data