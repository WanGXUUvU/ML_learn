import pandas as pd
data ={
    "school_name": "ABC primary school",
    "class": "Year 1",
    "students": [
    {
        "id": "A001",
        "name": "Tom",
        "math": 60,
        "physics": 66,
        "chemistry": 61
    },
    {
        "id": "A002",
        "name": "James",
        "math": 89,
        "physics": 76,
        "chemistry": 51
    },
    {
        "id": "A003",
        "name": "Jenny",
        "math": 79,
        "physics": 90,
        "chemistry": 78
    }]
}
df=pd.DataFrame(data['students'])
print(df)
df1=pd.json_normalize(data,record_path=['students'],meta=['school_name','class'])
print(df1)
df1.to_json('output.json',orient='records')  #保存为json文件