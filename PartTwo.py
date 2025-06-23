
from pathlib import Path
import pandas as pd

#loading data from csv file
with open("p2-texts /hansard40000.csv", mode="r",encoding='utf-8') as file:
    df=pd.read_csv(file,header=0)

#testing further results with local excel dublicate of hansard40000.csv
#checking number rows with Labour (Co-op) before replacing the value. Number is the same as in Excel
filtered = df[df["party"] == 'Labour (Co-op)']
#print(filtered)
num_rows = filtered.shape[0]
print(num_rows)
#replacing value in column "party"
df["party"] = df["party"].replace({"Labour (Co-op)":"Labour"}) 
#checking number of rows after replacment - received 0 rows using "Labour (Co-op)". The function is ready to commit
filtered = df[df["party"] == 'Labour']
num_rows = filtered.shape[0]
print(num_rows)


