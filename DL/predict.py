import numpy as np
import torch
import pandas as pd
import torch.nn as nn

train_data=pd.read_csv('/kaggle/input/power-data/5-Site_1.csv')
test_data=pd.read_csv('/kaggle/input/power-data/10-Site_2.csv')

df=pd.DataFrame(train_data)
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
X=df.drop(['Active_Power','timestamp'],axis=1)
y=df['Active_Power']
X=torch.FloatTensor(X.values)
y=torch.FloatTensor(y.values).reshape(-1,1)

loss=nn.MSELoss()
features=X.shape[1]
## nn.Sequential是一个容器，可以将多个神经网络层组合在一起
def net():
    net=nn.Sequential(
        nn.Linear(features,1))
    return net
def log_rmse(net,features,labels):
    with torch.no_grad():
        # 将小于1的值设为1，使得取对数时数值更稳定
        clipped_preds = torch.clamp(net(features), 1, float('inf'))
        rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    ##train_iter是一个可迭代对象，每次迭代返回一个小批量数据
    train_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_features, train_labels),
        batch_size, shuffle=True)
    #train_ls, test_ls分别存储每个epoch的训练和测试误差
    train_ls, test_ls = [], []
    # 这里使用的是Adam优化算法
    # optimizer是用于更新网络权重的优化器
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

num_epochs = 100
learning_rate = 0.01
weight_decay = 0
batch_size = 64
# 训练集和验证集
df_val=pd.DataFrame(test_data)
df_val = df_val.ffill()
df_val['timestamp'] = pd.to_datetime(df['timestamp'])
df_val['year'] = df_val['timestamp'].dt.year
df_val['month'] = df_val['timestamp'].dt.month
df_val['hour'] = df_val['timestamp'].dt.hour
df_val['minute'] = df_val['timestamp'].dt.minute
df_val['minutes of day'] = df_val['hour'] * 60 + df_val['minute']
df_val['hour_sin'] = np.sin(2 * np.pi * df_val['hour'] / 24)
df_val['hour_cos'] = np.cos(2 * np.pi * df_val['hour'] / 24)
df_val['month_sin'] = np.sin(2 * np.pi * df_val['month'] / 12)
df_val['month_cos'] = np.cos(2 * np.pi * df_val['month'] / 12)
X_val=df_val.drop(['Active_Power','timestamp'],axis=1)
y_val=df_val['Active_Power']
X_val=torch.FloatTensor(X_val.values)
y_val=torch.FloatTensor(y_val.values).reshape(-1,1)
train_ls, test_ls = train(net(), X, y, X_val, y_val,
                          num_epochs, learning_rate, weight_decay, batch_size)
import matplotlib.pyplot as plt
def semilogy(x, y, xlabel, ylabel, xlim, ylim, legend):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(legend)
    plt.show()
semilogy(range(1, num_epochs + 1), train_ls, 'epoch', 'rmse', [1, num_epochs], [1, 100], ['train'])
semilogy(range(1, num_epochs + 1), test_ls, 'epoch', 'rmse', [1, num_epochs], [1, 100], ['test'])
# 训练后直接在验证集上测试
model=net()
# ... 训练代码 ...

# 1. 划分训练集和验证集
split_idx = int(len(X) * 0.8)
train_features = X[:split_idx]
train_labels = y[:split_idx]
val_features = X[split_idx:]
val_labels = y[split_idx:]

# 2. 创建模型并训练
model = net()
train_ls, val_ls = train(model, train_features, train_labels, 
                        val_features, val_labels,
                        num_epochs=100, learning_rate=0.01, 
                        weight_decay=0.001, batch_size=64)

# 3. 简单验证：计算最终验证误差
final_val_rmse = log_rmse(model, val_features, val_labels)
print(f"最终验证RMSE: {final_val_rmse:.4f}")

# 4. 可视化训练过程（可选）
import matplotlib.pyplot as plt
plt.plot(range(len(train_ls)), train_ls, label='Train RMSE')
plt.plot(range(len(val_ls)), val_ls, label='Validation RMSE')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.show()