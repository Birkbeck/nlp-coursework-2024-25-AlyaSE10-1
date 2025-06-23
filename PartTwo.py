
from pathlib import Path
import pandas as pd

#loading data from csv file
with open("p2-texts /hansard40000.csv", mode="r",encoding='utf-8') as file:
    df=pd.read_csv(file,header=0)

#a(i)
#testing further results with local excel dublicate of hansard40000.csv
#checking number rows with Labour (Co-op) before replacing the value. Number is the same as in Excel
filtered = df[df["party"] == 'Labour (Co-op)']
#print(filtered)
#num_rows = filtered.shape[0]
#print(num_rows)
#replacing value in column "party"
df["party"] = df["party"].replace({"Labour (Co-op)":"Labour"}) 
#checking number of rows after replacment - received 0 rows using "Labour (Co-op)". The function is ready to commit
#filtered = df[df["party"] == 'Labour']
#num_rows = filtered.shape[0]
#print(num_rows)

#a(ii)
#Searching four most common party names(verified against test Excel)
top4_party_names = df["party"].value_counts().head(4)
print(top4_party_names)

#remove any rows where the value of the ‘party’ column is not one of the four most common party names(verified against test Excel)
top4_df = df[df["party"].isin(top4_party_names.index)]
num_rows = top4_df.shape[0]
print(num_rows)
'''
#removing value, rows will stay,  replacing with "" 
top4_df["party"] = df["party"].replace({"Speaker":""}) 
num_rows = top4_df.shape[0]
print(num_rows)'''





