import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import RFE
import joblib


## 第一步：数据预处理
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

#第二步：加载数据集
datasets=['/kaggle/input/power-data/5-Site_1.csv',
          '/kaggle/input/power-data/10-Site_2.csv',
          '/kaggle/input/power-data/11-Site_4.csv',
          '/kaggle/input/power-data/12-Site_5.csv',
          '/kaggle/input/power-data/6-Site_3-C.csv']

for index, train_dataset in enumerate(datasets):
    print(f"第{index+1}/5轮:使用的数据集是:{train_dataset}")
    print(f"{'='*50}")
    df = process_data(train_dataset)
    x_train = df.drop(['Active_Power', 'timestamp'], axis=1)
    y_train = df['Active_Power']

    # 移除 n_jobs 参数
    model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    model.fit(x_train, y_train)

    print("SVR模型训练完成!")
    save_dir = '/kaggle/working/'
    dataset_name = train_dataset.split('/')[-1].replace('.csv', '')
    model_filename = f'{save_dir}/SVR_model_trained_on_{dataset_name}.joblib'
    joblib.dump(model, model_filename)
    print(f"模型已保存为: {model_filename}")

    # 修复测试部分的变量名错误
    for test_dataset in datasets:
        if test_dataset != train_dataset:
            print(f"使用{model_filename}在{test_dataset}上进行测试")
            test_df = process_data(test_dataset)  # 修复变量名
            X_test = test_df.drop(['Active_Power', 'timestamp'], axis=1)  # 修复变量名
            y_test = test_df['Active_Power']  # 修复变量名
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            print(f'SVR模型 - 均方误差 (MSE): {mse:.4f}')
            print(f'SVR模型 - 均方根误差 (RMSE): {rmse:.4f}')
            print(f'SVR模型 - 平均绝对误差 (MAE): {mae:.4f}')
            print(f'SVR模型 - R²得分: {r2:.4f}')
            print("\n")
            print(f"{'='*50}\n")