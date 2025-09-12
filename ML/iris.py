import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#加载鸢尾花数据集
data= load_iris()

#将数据转换为DataFrame
df=pd.DataFrame(data.data, columns=data.feature_names)
df['target']=data.target
df['species']=df['target'].apply(lambda x: data.target_names[x])

X=df.drop(columns=['target','species'])
y=df['target']

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model_dt=DecisionTreeClassifier(random_state=42) ##random_state随机数种子

model_dt.fit(X_train, y_train)

y_pred_dt=model_dt.predict(X_test)

acc_dt=accuracy_score(y_test, y_pred_dt)

print(f'Decision Tree Accuracy: {acc_dt:.4f}')



## 使用支持向量机进行预测
from sklearn.svm import SVC

model_svc=SVC(random_state=42)
model_svc.fit(X_train, y_train)

y_pred_svc=model_svc.predict(X_test)

acc_svc=accuracy_score(y_test, y_pred_svc)

print(f'Support Vector Machine Accuracy: {acc_svc:.4f}')


#模型F1分数评估、精度、召回
from sklearn.metrics import classification_report

y_pred_dt=model_dt.predict(X_test)
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

y_pred_svc=model_svc.predict(X_test)
print("Support Vector Machine Classification Report:")
print(classification_report(y_test, y_pred_svc))
