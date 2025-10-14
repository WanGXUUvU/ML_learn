import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

def load_data(file_path):
    if file_path == '/kaggle/input/power-data/6-Site_3-C.csv':
        data = pd.read_csv(file_path, encoding='gbk')
        df = pd.DataFrame(data)
        df = df.drop(index=0)  # 修复：需要赋值
    else:
        data = pd.read_csv(file_path)
        df = pd.DataFrame(data)
    return df
#异常值检测
def detect_outliers(df, threshold=3):
    """
    对所有数值型列进行异常值检测，返回每列的异常值DataFrame字典。
    """
    outliers_dict = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        z_col = f"{col}_Z_Score"
        df[z_col] = stats.zscore(df[col], nan_policy='omit')
        outliers = df[df[z_col].abs() > threshold]
        outliers_dict[col] = outliers
        print(f"Column: {col}, Outliers detected: {len(outliers)}")
    return outliers_dict
#异常值处理 前向填充
def handle_outliers(df, threshold=3):
    outliers = detect_outliers(df, threshold)
    for col, outlier_df in outliers.items():
        df[col] = df[col].mask(df[col].index.isin(outlier_df.index))
        df[col] = df[col].ffill()
    return df

def load_and_process_data(file_path):
    df = load_data(file_path)
    df = df.ffill()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['minutes of day'] = df['hour'] * 60 + df['minute']
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df
#绘制异常值图形
def plot_outliers(df, outliers):
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['Active_Power'], label='Active Power', color='blue')
    plt.scatter(outliers['timestamp'], outliers['Active_Power'], label='Outliers', color='red')
    plt.title('Active Power with Outliers')
    plt.xlabel('Time')
    plt.ylabel('Active Power')
    plt.legend()
    plt.show()  

if __name__ == "__main__":
    datasets=['/kaggle/input/power-data/5-Site_1.csv',
              '/kaggle/input/power-data/10-Site_2.csv',
              '/kaggle/input/power-data/11-Site_4.csv',
              '/kaggle/input/power-data/8-Site_5.csv',
              '/kaggle/input/power-data/6-Site_3-C.csv']
    
    for file_path in datasets:
        print('='*50)
        print(f"Processing dataset: {file_path}")
        print('='*50)
        df = load_and_process_data(file_path)
        outliers = detect_outliers(df)
        print(f"Detected {len(outliers)} outliers in dataset: {file_path}")
        df = handle_outliers(df)
        print('-'*50)
        df=detect_outliers(df)