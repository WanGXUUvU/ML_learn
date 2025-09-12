import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # 随机森林回归模型
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.feature_selection import RFE, SelectFromModel
import joblib
import os

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    
    df=pd.DataFrame(data)
    if file_path =='/kaggle/input/power-data/6-Site_3-C.csv':
        df.drop(index=0)
    df=df.ffill()
    df['timestamp']=pd.to_datetime(df['timestamp'])
    df['year']=df['timestamp'].dt.year
    df['month']=df['timestamp'].dt.month
    df['hour']=df['timestamp'].dt.hour
    df['minute']=df['timestamp'].dt.minute
    df['minutes of day']=df['hour']*60+df['minute']

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df = df.drop('Hail_Accumulation', axis=1, errors='ignore')
    return df

datasets=['/kaggle/input/power-data/5-Site_1.csv',
          '/kaggle/input/power-data/10-Site_2.csv',
          '/kaggle/input/power-data/11-Site_4.csv',
          '/kaggle/input/power-data/6-Site_3-C.csv',
          '/kaggle/input/power-data/8-Site_5.csv']
#axis=1表示删除列
for index,train_dataset in enumerate(datasets):
    print(f"第{index+1}/5轮，使用的数据集为:{train_dataset}")
    print(f"{'='*50}")
    df=preprocess_data(train_dataset)
    
    
    X_train=df.drop(['Active_Power', 'timestamp'], axis=1)
    y_train=df['Active_Power']
    model=RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=-1)

    # 特征选择
    selector = RFE(model, n_features_to_select=15, step=1)
    selector = selector.fit(X_train, y_train)
    X_train_rfe=X_train.loc[:,selector.support_]
    print(f"选择的特征有: {X_train_rfe.columns.tolist()}")
    #.fit代表模型的训练
    model.fit(X_train_rfe, y_train)

    save_dir='/kaggle/working/'
    dataset_name = train_dataset.split('/')[-1].replace('.csv', '')  # 提取数据集名
    model_filename = f'{save_dir}/model_trained_on_{dataset_name}.joblib'
    joblib.dump(model, model_filename)
    print(f"模型已保存到: {model_filename}")
    
    for i ,test_dataset in enumerate(datasets):
        if i==index:
            continue
        else:

            test_df=preprocess_data(test_dataset)
            X_test=test_df.drop(['Active_Power','timestamp'],axis=1)
            y_test=test_df['Active_Power']
            X_test_rfe = X_test.reindex(columns=X_train_rfe.columns, fill_value=0)
            y_pred=model.predict(X_test)

            mse=mean_squared_error(y_test, y_pred)
            rmse=np.sqrt(mse)
            r2=r2_score(y_test, y_pred)
            mae=mean_absolute_error(y_test,y_pred)
            print(f"在 {test_dataset} 上的测试结果-------------")
            print(f'随机森林回归模型 - 均方误差 (MSE): {mse:.4f}')
            print(f'随机森林回归模型 - 均方根误差 (RMSE): {rmse:.4f}')
            print(f'随机森林回归模型 - 平均绝对误差 (MAE): {mae:.4f}')
            print(f'随机森林回归模型 - R²得分: {r2:.4f}')
