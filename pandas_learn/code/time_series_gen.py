import pandas as pd
##生成时间序列
rng = pd.date_range('2023-01-01', periods=10, freq='D')
print(rng)
new_data=rng+pd.Timedelta(5,unit='D')
print(new_data)
df=pd.DataFrame({'Date': rng, 'Value': range(10)})
