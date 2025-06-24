
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score

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

#b
#1.Create a vectorizer using default parameters, except for omitting English stopwords and setting max_features to
3000.
vectorizer = TfidfVectorizer(stop_words="english",max_features=3000)
#2.Prepare the dataframe, remove the rows without target(NaN)
#print(df.shape)
filtered_df = df[df["party"].notna()]
#print("Filtered df", filtered_df.shape)
#3.Vectorize the speech column of the initial dataframe 
X = vectorizer.fit_transform(filtered_df['speech'])
#4.The goal of the assignment is to predict the political party, out target is column "party"
y = filtered_df["party"]
#S5.plit the data into a train and test set, using stratified sampling(when we divide observations we keep the same proportions between classes as we have in the intial dataset, helps with inbalanced classes like we have)
# with a random seed of 26 (seed helps to get the same number each time we run the code)
#I decided to split data 80/20 as we have normal size dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    stratify=y,
    random_state=26
)
#6.Checking the results with print and excel. We have a nice dimentions where the number of rows are 80/20 of the intial filtered dataframe and the columns are max_features, 3000
#print("train size:", X_train.shape)
#print("test size", X_test.shape)

#c
#1.Train Random forest classifier with n_estimators = 300(number of trees), keep the same seed. 
#Training base on training data and then predict using test observations (X_test)
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
print("Classification_report for SVM:\n", classification_report(y_test,y_pred_svm))
















