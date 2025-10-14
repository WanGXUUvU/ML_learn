import torch.nn as nn
import torch
def model():
    return nn.Sequential(
        #1代表输入通道数，6代表输出通道数，5代表卷积核大小
        nn.Conv2d(1,6,kernel_size=5,stride=1,padding=2),
        nn.ReLU(),
        nn.AvgPool2d(2,2),
        nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0),
        nn.ReLU(),
        nn.AvgPool2d(2,2),
        # Flatten 代表展平多维的输入数据为一维
        nn.Flatten(),
        nn.Linear(16*4*4,120),
        nn.ReLU(),
        nn.Linear(120,84),
        nn.ReLU(),
        nn.Linear(84,10)
    )
def evaluate_accuracy(data_iter,net,device):
    # acc_sum:正确预测的数量，n:样本总数
    acc_sum,n = 0.0,0
    with torch.no_grad():
        for X,y in data_iter:
            X,y = X.to(device),y.to(device)
            acc_sum += (net(X).argmax(dim=1)==y).float().sum().cpu().item()
            n += y.shape[0]
    return acc_sum/n
def train(num_epochs,train_iter,test_iter,net,optimizer,device):
    net.to(device)
    loss=nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        net.train()
        #train_l_sum:训练损失总和，train_acc_sum:训练正确预测的数量，n:样本总数，batch_count:批量数量
        train_l_sum,train_acc_sum,n,batch_count=0.0,0.0,0,0
        for X,y in train_iter:
            X,y = X.to(device),y.to(device)
            y_hat = net(X)
            l = loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1)==y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter,net,device)
        print(f'epoch {epoch+1}, loss {train_l_sum/batch_count:.4f}, train acc {train_acc_sum/n:.3f}, test acc {test_acc:.3f}')
        