import pandas as pd
import matplotlib.pyplot as plt
Data = {
    'Year': [2020, 2021, 2022, 2023],
    'Sales': [150, 200, 250, 300]
}

df = pd.DataFrame(Data)
df.plot(x='Year', y='Sales', kind='bar')
plt.title('Sales by Year')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.show()

