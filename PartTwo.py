
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
import numpy as np

#loading data from csv file.The df is the source for our next tasks
with open("p2-texts /hansard40000.csv", mode="r",encoding='utf-8') as file:
    df=pd.read_csv(file,header=0)

#a(i) Data preprocessing
#testing further results with local Excel dublicate of hansard40000.csv
#checking number rows with Labour (Co-op) before replacing the value. Number is the same as in Excel
filtered = df[df["party"] == 'Labour (Co-op)']
#print(filtered)
#num_rows = filtered.shape[0]
#print(num_rows)

#rename the ‘Labour (Co-op)’ value in ‘party’ column to ‘Labour’. Merging observation with the same party name
df["party"] = df["party"].replace({"Labour (Co-op)":"Labour"}) 

#num_rows = filtered.shape[0]
#print(df.shape)

#a(ii)
#removing value Speaker before finding the top4. As I am preparing the dataset for further training and "party" column will be my target, value Speaker is not a party name, So it is a missing value. I clean only value not rows

df["party"] = df["party"].replace("Speaker",np.nan)
#testing whether Speaker value was replaced
#filtered = df[df["party"]== "Speaker"]
#print(filtered.shape)
#Now when I cleaned not relevant Speaker I am searching for top 4 party names that I will predict on a future steps of the task. By default value_counts will ignore the NA "party" value.  
top4 = df["party"].value_counts(dropna=True).head(4).index
#checking the leaders, that Speaker is not there. Numbers and Leader are verified against excel#
#print(top4)
df_top4 =  df[df["party"].isin(top4)]
#print("Top4", df_top4.shape)

#I will use column 'speech" for predicting the party name. That is why on this step I remove any rows where the value in the ‘speech_class’ column is not ‘Speech’, so has not relevant data. The number of rows is the same as in previous dataframe because we have speech in all observations
df_top4 = df_top4[df_top4["speech_class"] == 'Speech']
#print("Only speech", df_top4.shape)

#For detecting nessesary features I need a long text, there is no nesessary information in small speech. That is why I remove any rows where the text in the ‘speech’ column is less than 1000 characters long
df_prepared = df_top4[df_top4['speech'].str.len() >=1000]
#Checking my final dataframe. This is my prepared dataset with information about 4 classes where I have enough information for orediction. Dataset is still not balanced but I get rid off parties with extremely small number of observation 
print(df_prepared.shape)

#b
#1.Create a vectorizer using default parameters, except for omitting English stopwords and setting max_features to
3000.
vectorizer = TfidfVectorizer(stop_words="english",max_features=3000)

#2.Vectorize the speech column of the initial dataframe 
X = vectorizer.fit_transform(df_prepared['speech'])
#3.The goal of the assignment is to predict the political party, out target is column "party"
y = df_prepared["party"]
#4.Split the data into a train and test set, using stratified sampling(when we divide observations we keep the same proportions between classes as we have in the intial dataset, helps with inbalanced classes like we have)
# with a random seed of 26 (seed helps to get the same number each time we run the code)
#I decided to split data 80/20 as we have normal size dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    stratify=y,
    random_state=26
)
#5.Checking the results with print and excel. We have a nice dimentions where the number of rows are 80/20 of the intial filtered dataframe and the columns are max_features, 3000
print("train size:", X_train.shape)
print("test size", X_test.shape)

'''#c
#1.Train Random forest classifier with n_estimators = 300(number of trees), keep the same seed. 
#Training base on training data and then predict using test observations (X_test)
print(y_train.value_counts())
random_f= RandomForestClassifier(n_estimators=300, random_state=26)
random_f.fit(X_train,y_train)
y_pred_random_f = random_f.predict(X_test)
#2.Train SVM with linear kernel
svm = SVC(kernel='linear', random_state=26)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
#3.Print the scikit-learn macro-average f1 score and classification report for each classifier on the test set
print("Macro-average f1 score for Random forest:", f1_score(y_test,y_pred_random_f,average="macro"))
print("Classification_report for Random forest:\n", classification_report(y_test,y_pred_random_f))
print("Macro-average f1 score for SVM:",f1_score(y_test,y_pred_svm,average="macro"))
print("Classification_report for SVM:\n", classification_report(y_test,y_pred_svm))'''
















