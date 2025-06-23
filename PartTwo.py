
from pathlib import Path
import pandas as pd

with open("p2-texts /hansard40000.csv", mode="r",encoding='utf-8') as file:
    df=pd.read_csv(file,header=1)
print(df)

'''def read_speeches(csv_path):
    csv_path = p2-texts/hansard40000.csv
    print(Path.cwd())
    if not file.exists():
        raise FileNotFoundError(f"File not found")
    #print(path)
    df = pd.read_csv(file, header=1)
    print(df.head())

#print(read_speeches(Path))

if __name__ == "__main__":  
    read_speeches()'''

