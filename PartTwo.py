
from pathlib import Path
import pandas as pd

#loading data from csv file
with open("p2-texts /hansard40000.csv", mode="r",encoding='utf-8') as file:
    df=pd.read_csv(file,header=0)

#a(i)
#testing further results with local Excel dublicate of hansard40000.csv
#checking number rows with Labour (Co-op) before replacing the value. Number is the same as in Excel
filtered = df[df["party"] == 'Labour (Co-op)']
#print(filtered)
#num_rows = filtered.shape[0]
#print(num_rows)
#rename the ‘Labour (Co-op)’ value in ‘party’ column to ‘Labour’
df["party"] = df["party"].replace({"Labour (Co-op)":"Labour"}) 
#checking number of rows after replacment - received 0 rows using "Labour (Co-op)". The function is ready to commit
#filtered = df[df["party"] == 'Labour']
#num_rows = filtered.shape[0]
#print(num_rows)

#a(ii)
#Searching four most common party names(verified against test Excel)
top4_party_names = df["party"].value_counts().head(4)
#print(top4_party_names)

#remove any rows where the value of the ‘party’ column is not one of the four most common party names(verified against test Excel)
top4_df = df[df["party"].isin(top4_party_names.index)]
num_rows = top4_df.shape[0]
#print(num_rows)

#removing value, rows will stay,  replacing with "" 
top4_df.loc[top4_df["party"] == "Speaker", "party"] = ""

num_rows = top4_df.shape[0]
#print(num_rows)

#remove any rows where the value in the ‘speech_class’ column is not ‘Speech’
top4_df_no_sc = top4_df[top4_df["speech_class"] != 'Speech']
#print(top4_df_no_sc)

#remove any rows where the text in the ‘speech’ column is less than 1000 characters long
final_df = top4_df_no_sc[top4_df_no_sc['speech'].str.len() >=1000]
print(final_df.shape)










