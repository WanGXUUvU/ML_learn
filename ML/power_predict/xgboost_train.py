import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import scipy.stats as stats

def simple_outlier_detection(df, column='Active_Power', z_threshold=3, show_info=True):
    """简单的Z-Score异常值检测和处理"""
    # 确保列是数值类型
    if not pd.api.types.is_numeric_dtype(df[column]):
        print(f"警告: {column} 不是数值类型，正在转换...")
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df = df.dropna(subset=[column])  # 删除无法转换的行
    
    # 计算Z分数
    try:
        z_scores = np.abs(stats.zscore(df[column]))
        outliers = z_scores > z_threshold
        
        if show_info:
            print(f"数据总数: {len(df)}, 异常值: {outliers.sum()} ({outliers.sum()/len(df)*100:.1f}%)")
        
        # 简单处理：用95%分位数限制异常值
        df_clean = df.copy()
        if outliers.sum() > 0:
            lower_bound = df[column].quantile(0.05)
            upper_bound = df[column].quantile(0.95)
            df_clean.loc[outliers, column] = df_clean.loc[outliers, column].clip(lower_bound, upper_bound)
            if show_info:
                print(f"异常值已限制到范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return df_clean
    
    except Exception as e:
        print(f"Z-Score计算失败: {e}")
        print(f"{column} 列的数据类型: {df[column].dtype}")
        print(f"{column} 列的前5个值: {df[column].head()}")
        return df

def preprocess_data(file_path, clean_outliers=True):
    if file_path == '/kaggle/input/power-data/6-Site_3-C.csv':
        data = pd.read_csv(file_path, encoding='gbk')
        df = pd.DataFrame(data)
        df = df.drop(index=0)  # 修复：需要赋值
    else:
        data = pd.read_csv(file_path)
        df = pd.DataFrame(data)
        df = df.drop(columns=['Hail_Accumulation'], errors='ignore')  # 删除不存在的列时不报错
    
    df = df.ffill()
    
    # 转换所有非数值列为数值类型
    for col in df.columns:
        if col not in ['timestamp']:  # 移除 'Active_Power' 的排除
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 特别处理 Active_Power 列，确保它是数值类型
    if 'Active_Power' in df.columns:
        df['Active_Power'] = pd.to_numeric(df['Active_Power'], errors='coerce')
        # 检查转换后是否有 NaN 值
        nan_count = df['Active_Power'].isna().sum()
        if nan_count > 0:
            print(f"警告: Active_Power列中有 {nan_count} 个无法转换的值，将用前后值填充")
    
    # 处理转换后的 NaN 值
    df = df.ffill().bfill()
    
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
    
    # 简单异常值清理
    if clean_outliers:
        print(f"\n异常值检测 - {file_path.split('/')[-1]}:")
        df = simple_outlier_detection(df, 'Active_Power')
    
    return df

#选择重要的特征
def select_important_features(df, target, threshold=0.1):
    corr = df.corr()[target]
    important_features = corr[abs(corr) > threshold].index.tolist()
    #删除目标变量本身
    important_features.remove(target)
    # 安全删除timestamp（如果存在的话）
    if 'timestamp' in important_features:
        important_features.remove('timestamp')
    return important_features
#训练模型
def train_xgboost_model(datasets):
    models = []
    for dataset in datasets:
        df = preprocess_data(dataset)
        x = df.drop(['Active_Power', 'timestamp'], axis=1)
        y = df['Active_Power']
        # 按时间顺序划分训练集和验证集
        split_idx = int(len(df) * 0.8)
        #选择重要特征
        important_features = select_important_features(df, 'Active_Power', threshold=0.1)
        print(f"重要特征: {important_features}")
        x = x[important_features]
        X_train, X_val = x.iloc[:split_idx], x.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        evals_result = {}
        # 时序优化的最佳参数组合
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,           # 减少树数量
            learning_rate=0.1,          # 提高学习率补偿
            max_depth=8,                # 减少深度防过拟合
            min_child_weight=3,         # 增加最小叶子权重
            reg_alpha=0.3,              # 较强L1正则
            tree_method='hist',
            device='cuda',
            eval_metric='rmse',
            early_stopping_rounds=15,   # 更早停止
            random_state=42
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=20
        )
        
        # 获取训练过程评估结果
        evals_result = model.evals_result()
        # 绘制学习曲线
        plt.figure(figsize=(8,4))
        plt.plot(evals_result['validation_0']['rmse'], label='train')
        plt.plot(evals_result['validation_1']['rmse'], label='val')
        plt.legend() #legend代表图例
        plt.xlabel('epoch')
        plt.ylabel('RMSE')
        plt.title(f'learn-line - {dataset.split("/")[-1]}')
        plt.tight_layout()
        plt.show()

        # 保存模型和对应的特征列表
        save_dir = '/kaggle/working/'
        dataset_name = dataset.split('/')[-1].replace('.csv', '')
        model_filename = f'{save_dir}/xgboost_model_trained_on_{dataset_name}.joblib'
        features_filename = f'{save_dir}/features_trained_on_{dataset_name}.joblib'
        
        joblib.dump(model, model_filename)
        joblib.dump(important_features, features_filename)  # 保存特征列表
        print(f"模型已保存为: {model_filename}")
        print(f"特征列表已保存为: {features_filename}")
    return model
#绘制数据集热力图
def plot_heatmap(datasets):
    dataset = select_dataset_interactively(datasets)   # 让用户选择一个数据集文件路径
    df = preprocess_data(dataset)                      # 读取并预处理该数据集，得到DataFrame
    plt.figure(figsize=(12, 10))                       # 设置画布大小
    corr = df.corr()                                   # 计算所有数值型特征的相关系数矩阵
    print(corr['Active_Power'].sort_values)
    plt.imshow(corr, cmap='coolwarm', interpolation='nearest')  # 用imshow画相关系数热力图
    plt.colorbar()                                     # 添加颜色条
    plt.xticks(range(len(corr)), corr.columns, rotation=90)     # 设置x轴标签为特征名并旋转
    plt.yticks(range(len(corr)), corr.columns)                  # 设置y轴标签为特征名
    plt.title('Feature Correlation Heatmap')            # 设置标题
    plt.tight_layout()                                  # 自动调整子图参数以填充整个图像区域
    plt.show()                                          # 显示
#特征选择
def select_model_interactively():
    model_list = [
        "5-Site_1", "10-Site_2", "11-Site_4", "6-Site_3-C", "8-Site_5"
    ]
    print("请选择你要加载的模型：")
    for idx, name in enumerate(model_list):
        print(f"{idx}: {name}")
    while True:
        try:
            idx = int(input("请输入编号（如0、1、2...）："))
            if 0 <= idx < len(model_list):
                print(f"你选择的是：{model_list[idx]}")
                return f"xgboost_model_trained_on_{model_list[idx]}.joblib"
            else:
                print("输入编号超出范围，请重新输入。")
        except Exception:
            print("输入无效，请输入数字编号。")
def select_dataset_interactively(datasets):
    print("请选择你要加载的数据集：")
    for idx, dataset in enumerate(datasets):
        print(f"{idx}: {dataset}")
    while True:
        try:
            idx = int(input("请输入编号（如0、1、2...）："))
            if 0 <= idx < len(datasets):
                print(f"你选择的是：{datasets[idx]}")
                return datasets[idx]
            else:
                print("输入编号超出范围，请重新输入。")
        except Exception:
            print("输入无效，请输入数字编号。")
def test_xgboost_model(datasets):
    model_filename = select_model_interactively()
    model = joblib.load(f'/kaggle/working/{model_filename}')
    print(f"已加载模型：{model_filename}")
    
    # 加载对应的特征列表
    features_filename = model_filename.replace('xgboost_model_', 'features_')
    try:
        training_features = joblib.load(f'/kaggle/working/{features_filename}')
        print(f"已加载特征列表：{training_features}")
    except FileNotFoundError:
        print("未找到特征列表文件，请重新训练模型")
        return

    # 修改数据集选择逻辑
    print("请选择你要测试的数据集：")
    print("0: 对所有数据集进行测试")
    for idx, dataset in enumerate(datasets):
        print(f"{idx+1}: {dataset}")
    
    while True:
        try:
            choice = int(input("请输入编号（0表示全部，1-5表示具体数据集）："))
            if choice == 0:
                # 对所有数据集进行测试
                print("\n开始对所有数据集进行测试...")
                print("="*60)
                
                all_results = []
                
                for i, test_dataset in enumerate(datasets):
                    print(f"\n正在测试数据集 {i+1}/5: {test_dataset.split('/')[-1]}")
                    print("-" * 40)
                    
                    df = preprocess_data(test_dataset)
                    x_test = df.drop(['Active_Power', 'timestamp'], axis=1)
                    y_test = df['Active_Power']
                    timestamps = df['timestamp']  # 保存时间戳用于绘图

                    # 确保测试数据使用与训练时相同的特征
                    x_test = x_test.reindex(columns=training_features, fill_value=0)
                    
                    y_pred = model.predict(x_test)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    # 保存结果
                    result = {
                        'dataset': test_dataset.split('/')[-1],
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2
                    }
                    all_results.append(result)
                    
                    print(f'MSE: {mse:.4f}')
                    print(f'RMSE: {rmse:.4f}')
                    print(f'MAE: {mae:.4f}')
                    print(f'R²: {r2:.4f}')
                    
                    # 绘制预测结果对比图（只显示前1000个点避免图形过于密集）
                    plot_prediction_comparison(timestamps, y_test, y_pred, test_dataset.split('/')[-1], max_points=1000)
                
                # 输出汇总结果
                print("\n" + "="*60)
                print("所有数据集测试汇总结果:")
                print("="*60)
                print(f"{'数据集':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
                print("-" * 55)
                
                total_rmse = 0
                for result in all_results:
                    print(f"{result['dataset']:<20} {result['rmse']:<10.4f} {result['mae']:<10.4f} {result['r2']:<10.4f}")
                    total_rmse += result['rmse']
                
                avg_rmse = total_rmse / len(all_results)
                print("-" * 55)
                print(f"{'平均RMSE':<20} {avg_rmse:<10.4f}")
                break
                
            elif 1 <= choice <= len(datasets):
                # 选择单个数据集测试
                test_dataset = datasets[choice-1]
                print(f"你选择的是：{test_dataset}")
                
                df = preprocess_data(test_dataset)
                x_test = df.drop(['Active_Power', 'timestamp'], axis=1)
                y_test = df['Active_Power']
                timestamps = df['timestamp']  # 保存时间戳用于绘图

                # 确保测试数据使用与训练时相同的特征
                x_test = x_test.reindex(columns=training_features, fill_value=0)
                print(f"测试特征维度：{x_test.shape}")
                print(f"使用的特征：{list(x_test.columns)}")

                y_pred = model.predict(x_test)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                print(f"\n在 {test_dataset.split('/')[-1]} 上的测试结果:")
                print("-" * 40)
                print(f'XGBoost模型 - 均方误差 (MSE): {mse:.4f}')
                print(f'XGBoost模型 - 均方根误差 (RMSE): {rmse:.4f}')
                print(f'XGBoost模型 - 平均绝对误差 (MAE): {mae:.4f}')
                print(f'XGBoost模型 - R²得分: {r2:.4f}')
                
                # 绘制预测结果对比图
                plot_prediction_comparison(timestamps, y_test, y_pred, test_dataset.split('/')[-1])
                break
                
            else:
                print(f"输入编号超出范围，请输入0-{len(datasets)}之间的数字。")
        except ValueError:
            print("输入无效，请输入数字编号。")

def plot_prediction_comparison(timestamps, y_true, y_pred, dataset_name, max_points=2000):
    """
    绘制预测结果与真实结果的对比折线图
    
    参数:
    timestamps: 时间戳
    y_true: 真实值
    y_pred: 预测值
    dataset_name: 数据集名称
    max_points: 最大显示点数（避免图形过于密集）
    """
    # 如果数据点太多，进行采样
    if len(y_true) > max_points:
        step = len(y_true) // max_points
        timestamps = timestamps[::step]
        y_true = y_true[::step]
        y_pred = y_pred[::step]
        print(f"数据点过多，已采样显示前 {len(y_true)} 个点")
    
    plt.figure(figsize=(15, 8))
    
    # 绘制真实值和预测值
    plt.plot(timestamps, y_true, label='真实值 (True)', color='blue', alpha=0.7, linewidth=1)
    plt.plot(timestamps, y_pred, label='预测值 (Predicted)', color='red', alpha=0.7, linewidth=1)
    
    plt.title(f'预测结果对比 - {dataset_name}', fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('Active Power', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 自动调整x轴标签角度
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 另外绘制散点图显示预测精度
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=1)
    
    # 绘制理想预测线 (y=x)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测线')
    
    plt.xlabel('真实值', fontsize=12)
    plt.ylabel('预测值', fontsize=12)
    plt.title(f'预测精度散点图 - {dataset_name}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
# 可选：快速查看所有数据集的异常值情况
def quick_outlier_summary(datasets):
    """快速查看所有数据集的异常值情况"""
    print("=== 异常值快速检查 ===")
    for dataset in datasets:
        print(f"\n{dataset.split('/')[-1]}:")
        try:
            df = preprocess_data(dataset, clean_outliers=False)  # 不清理，只检查
            
            # 确保 Active_Power 是数值类型
            if not pd.api.types.is_numeric_dtype(df['Active_Power']):
                df['Active_Power'] = pd.to_numeric(df['Active_Power'], errors='coerce')
                df = df.dropna(subset=['Active_Power'])
            
            z_scores = np.abs(stats.zscore(df['Active_Power']))
            outliers = (z_scores > 3).sum()
            print(f"  异常值: {outliers}/{len(df)} ({outliers/len(df)*100:.1f}%)")
            
        except Exception as e:
            print(f"  检查失败: {e}")
            print(f"  数据类型: {df['Active_Power'].dtype if 'df' in locals() else '未知'}")

if __name__ == "__main__":
    datasets=['/kaggle/input/power-data/5-Site_1.csv',
              '/kaggle/input/power-data/10-Site_2.csv',
              '/kaggle/input/power-data/11-Site_4.csv',
              '/kaggle/input/power-data/6-Site_3-C.csv',  
                '/kaggle/input/power-data/8-Site_5.csv']
    
    # 可选：快速查看异常值情况
    quick_outlier_summary(datasets)
    
    # 训练模型（现在会自动清理异常值）
    train_xgboost_model(datasets)
    # 测试模型
    #test_xgboost_model(datasets)

    # 绘制热力图
    #plot_heatmap(datasets)
