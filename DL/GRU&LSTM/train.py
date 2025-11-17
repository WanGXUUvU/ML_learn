import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.model import LSTMPredictor 
from data.dataset import TimeSeriesDataset
from data.preprocess import DataPreprocessor
from tqdm import tqdm
import pandas as pd

def train():
    """训练函数 - 修复版"""
    
    # ========== 1. 配置参数 ==========
    print("配置参数...")
    
    BATCH_SIZE = 64          # 增大批次
    NUM_EPOCHS = 50          # 增加轮数
    LEARNING_RATE = 0.0001   # 学习率
    SEQ_LEN = 24             # 时间步长度
    
    # ========== 2. 设备设置 ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # ========== 3. 加载和预处理数据（关键修改）==========
    print("="*50)
    print("加载和预处理数据...")
    print("="*50)
    
    # ===== 只预处理一次 =====
    preprocessor = DataPreprocessor('/Users/wangxu/Documents/学习记录/AI 学习/ML/power_predict/data/5-Site_1.csv')
    preprocessor.load_data()
    preprocessor.clean_data()
    preprocessor.feature_engineering()
    # ↑ 不调用 normalize_data()
    
    df = preprocessor.get_data()  # 获取未标准化的数据
    
    # 划分
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"\n数据划分:")
    print(f"  总数据: {len(df)} 行")
    print(f"  训练集: {len(train_df)} 行 (前80%)")
    print(f"  验证集: {len(val_df)} 行 (后20%)")
    
    # ===== 保存临时文件（简单方案）=====
    train_df.to_csv('/tmp/train_temp.csv', index=False)
    val_df.to_csv('/tmp/val_temp.csv', index=False)
    
    # ===== 创建训练集（第一次也是唯一一次标准化）=====
    print("\n创建训练集...")
    train_dataset = TimeSeriesDataset(
        data_path='/tmp/train_temp.csv',
        preprocessor=None,  # 会执行完整的process（包括标准化）
        is_train=True
    )
    
    # ===== 创建验证集（使用训练集的标准化参数）=====
    print("\n创建验证集...")
    val_dataset = TimeSeriesDataset(
        data_path='/tmp/val_temp.csv',
        preprocessor=train_dataset.get_preprocessor(),
        is_train=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # ========== 4. 创建模型 ==========
    print("\n创建模型...")
    
    sample_data, _ = train_dataset[0]
    input_size = sample_data.shape[1]
    
    model = LSTMPredictor(
        input_size=input_size,
        hidden_size=256,      # 增大
        num_layers=3,         # 增加层数
        output_size=1,
        dropout=0.3           # 增加dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}\n")
    
    # ========== 5. 损失函数和优化器 ==========
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ===== 新增：学习率调度器 =====
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # ========== 6. 训练循环 ==========
    print("开始训练...")
    print("=" * 70)
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(NUM_EPOCHS):
        # ===== 训练阶段 =====
        model.train()
        train_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Train", leave=False)
        for batch_idx, (x, y) in enumerate(train_pbar):
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x)
            
            if y.dim() == 1:
                y = y.unsqueeze(1)
            
            loss = criterion(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            
            # ===== 新增：梯度裁剪 =====
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # ===== 验证阶段 =====
        model.eval()
        val_loss = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Val", leave=False)
        with torch.no_grad():
            for x, y in val_pbar:
                x = x.to(device)
                y = y.to(device)
                
                pred = model(x)
                
                if y.dim() == 1:
                    y = y.unsqueeze(1)
                
                loss = criterion(pred, y)
                val_loss += loss.item()
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_val_loss = val_loss / len(val_loader)
        
        # ===== 学习率调度 =====
        scheduler.step(avg_val_loss)
        
        # ===== 打印结果 =====
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}] | "
              f"Train: {avg_train_loss:.4f} | "
              f"Val: {avg_val_loss:.4f} | "
              f"LR: {current_lr:.6f}", end="")
        
        # ===== 保存最佳模型 + 早停 =====
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_model.pth')
            print(" ← 最佳!")
        else:
            patience_counter += 1
            print(f" (patience: {patience_counter}/{max_patience})")
            
            if patience_counter >= max_patience:
                print(f"\n早停：验证损失 {max_patience} 轮未改善")
                break
    
    # ========== 7. 训练完成 ==========
    print("=" * 70)
    print(f"\n训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳模型已保存到: best_model.pth")


if __name__ == '__main__':
    train()