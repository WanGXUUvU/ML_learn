import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
def process_data(file_path):
    if file_path == '/kaggle/input/power-data/6-Site_3-C.csv':
        data = pd.read_csv(file_path, encoding='gbk')
        df = pd.DataFrame(data)
        df = df.drop(index=0)  # 修复：需要赋值
    else:
        data = pd.read_csv(file_path)
        df = pd.DataFrame(data)
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
def load_data(file_path):
    df = process_data(file_path)
    X = df.drop(['Active_Power', 'timestamp'], axis=1)
    y = df['Active_Power']
    return X, y
def simple_gpu_prediction(csv_file):
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 读取和预处理数据
    df = pd.read_csv(csv_file)
    features = ['Current_Phase_Average_Mean', 'Global_Horizontal_Radiation', 'Wind_Speed']
    
    # 关键：标准化数据
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features].fillna(df[features].median()))
    y = df['Active_Power'].fillna(df['Active_Power'].median()).values
    
    # 转为张量并移到GPU
    X = torch.FloatTensor(X).to(device)
    y = torch.FloatTensor(y).reshape(-1, 1).to(device)
    
    # 创建模型
    model = nn.Linear(3, 1).to(device)
    
    # 初始化权重
    nn.init.xavier_uniform_(model.weight)
    nn.init.zeros_(model.bias)
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # 训练
    for epoch in range(100):
        pred = model(X)
        loss = criterion(pred, y)
        
        if torch.isnan(loss):
            print(f"NaN detected at epoch {epoch}")
            break
            
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.6f}')
    
    return model, scaler, device

X,y=load_data('/kaggle/input/power-data/5-Site_1.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 时序数据按时间顺序划分，不能随机打乱
split_idx = int(len(X) * 0.8)  # 前80%作为训练集
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")

# 添加数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# 转换为张量
X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

model = nn.Linear(X_train.shape[1], 1).to(device)

# 初始化权重
nn.init.xavier_uniform_(model.weight)
nn.init.zeros_(model.bias)

criterion = nn.MSELoss()
# 降低学习率
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

losses = []

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # 检查 NaN
    if torch.isnan(loss):
        print(f"NaN detected at epoch {epoch}")
        break
    
    loss.backward()
    
    # 添加梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    losses.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        # 计算验证损失
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)
        print(f'Epoch [{epoch+1}/100], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')