
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
import numpy as np
import spacy
from collections import Counter, defaultdict


nlp = spacy.load("en_core_web_sm")

#loading data from csv file.The df is the source for our next tasks
with open("p2-texts /hansard40000.csv", mode="r",encoding='utf-8') as file:
    df=pd.read_csv(file,header=0)

#a(i) Data preprocessing
#testing further results with local Excel dublicate of hansard40000.csv
#checking number rows with Labour (Co-op) before replacing the value. Number is the same as in Excel
filtered = df[df["party"] == 'Labour (Co-op)']
#print(filtered.shape[0])

#rename the ‘Labour (Co-op)’ value in ‘party’ column to ‘Labour’. Merging observation with the same party name
df["party"] = df["party"].replace({"Labour (Co-op)":"Labour"}) 
#print(filtered.shape[0])

#a(ii)
#removing value Speaker before finding the top4. As I am preparing the dataset for further training and "party" column will be my target,
# value Speaker is not a party name, So it is a missing value. I clean only value not rows with np.nan because when I replaced with "" it was treated as a string and I still saw "Speaker" rows
df["party"] = df["party"].replace("Speaker",np.nan)
#testing whether Speaker value was replaced
#filtered = df[df["party"]== "Speaker"]
#print(filtered.shape)

#Now when I cleaned not relevant Speaker I am searching for top 4 party names that I will predict on a future steps of the task. 
top4 = df["party"].value_counts(dropna=True).head(4).index
#checking the leaders, that Speaker is not there. Numbers and Leader are verified against excel#
#print(top4)
df_top4 =  df[df["party"].isin(top4)]
#print("Top4", df_top4.shape)

#I will use column 'speech" for predicting the party name. That is why on this step I remove any rows where 
# the value in the ‘speech_class’ column is not ‘Speech’, so has not relevant data. The number of rows is the same as in previous dataframe because we have speech in all observations
df_top4 = df_top4[df_top4["speech_class"] == 'Speech']
#print("Only speech", df_top4.shape)

#For detecting nessesary features I need a long text, there is no nesessary information in small speech. 
# That is why I remove any rows where the text in the ‘speech’ column is less than 1000 characters long
df_prepared = df_top4[df_top4['speech'].str.len() >=1000]
#Checking my final dataframe. This is my prepared dataset with information about 4 classes. 
print(df_prepared.shape)

#b
#1.Make a function vectorizer, that include different parameters that I will reuse afterwards 
def create_vectorizer(stop_words=None,max_features=None,ngram_range=(1,1),tokenizer=None,lowercase=True,vocabulary=None):
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        max_features=max_features,
        ngram_range=ngram_range,
        tokenizer=tokenizer,
        lowercase=lowercase,
        vocabulary=vocabulary
    )
    return vectorizer
# Create a vectorizer using default parameters, except for omitting English stopwords and setting max_features to 3000.
vectorizer = create_vectorizer(stop_words="english",max_features=3000)

#2.Vectorize the speech column of the dataframe 
X = vectorizer.fit_transform(df_prepared['speech'])
#3.The goal of the assignment is to predict the political party, out target is column "party"
y = df_prepared["party"]
#4.Split the data into a train and test set, using stratified sampling(when I divide observations I keep the same proportions between classes
# as I have in the intial dataset, helps with inbalanced classes like I have)
# with a random seed of 26 (seed helps to get the same number each time we run the code)
#I decided to split data 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    stratify=y,
    random_state=26
)
#5.Checking the results with print and excel. 
# I have a nice dimentions where the number of rows are 80/20 of the intial filtered dataframe and the columns are max_features, 3000
#print("train size:", X_train.shape)
#print("test size", X_test.shape)

#1.Train Random forest classifier with n_estimators = 300(number of trees), keep the same seed. Added class_weight balanced because I have not many samples for  Liberal Democrat
#Training base on training data and then predict using test observations (X_test)

random_f= RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=26)
random_f.fit(X_train,y_train)
y_pred_random_f = random_f.predict(X_test)
  
#2.Train SVM with linear kernel

svm = SVC(kernel='linear', random_state=26)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

#3. I received a warning Precision is ill-defined and being set to 0.0 in labels with no predicted samples.Testing the training procedure
#print(np.unique(y_pred_random_f, return_counts=True))
#print(np.unique(y_pred_svm, return_counts=True))

#3.Print the scikit-learn macro-average f1 score and classification report for each classifier on the test set
#I received good f1score (0.82 for RF and 0.87 for SVM) for Conservative class (where I have many samples). With Labour and Scottish National party the results are medium and with Liberal Democrat they are very poor,
# I solved problem with warning that we do not have balanced dataset but still we do not have enough observations 
#I definitely need to reconsider feature selections and methods for better results because we can not distinguish the paterns nesessary for class detection
print("Macro-average f1 score for Random forest:", f1_score(y_test,y_pred_random_f,average="macro"))
print("Classification_report for Random forest:\n", classification_report(y_test,y_pred_random_f))
print("Macro-average f1 score for SVM:",f1_score(y_test,y_pred_svm,average="macro"))
print("Classification_report for SVM:\n", classification_report(y_test,y_pred_svm))


#d
#1. Adjust the parameters of the Tfidfvectorizer so that unigrams, bi-grams and tri-grams will be considered as features, limiting the total number of features to
#3000. As I am analysing the political speeches stopwords can be useful for detection the paterns. 
# Like the opposition may say "not good", insted of "good" and so on. As there is a grey area
vectorizer=create_vectorizer(ngram_range=(1,3),max_features=3000)
#2.Vectorize the speech column of the dataframe 
X = vectorizer.fit_transform(df_prepared['speech'])
y = df_prepared["party"]
#3.Split the data as before
X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    stratify=y,
    random_state=26
)
#4. Training Random forest with the  new vectorizer
random_f= RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=26)
random_f.fit(X_train,y_train)
y_pred_random_f = random_f.predict(X_test)
#5.Train SVM with the new vectorizer
svm = SVC(kernel='linear', random_state=26)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
#6.Print the scikit-learn macro-average f1 score and classification report for each classifier on the test set
print("Macro-average f1 score for Random forest with updated vectorizer:", f1_score(y_test,y_pred_random_f,average="macro"))
print("Classification_report for Random forest with updated vectorizer:\n", classification_report(y_test,y_pred_random_f))
print("Macro-average f1 score for SVM with updated vectorizer:",f1_score(y_test,y_pred_svm,average="macro"))
print("Classification_report for SVM with updated vectorizer:\n", classification_report(y_test,y_pred_svm))

#e
#Ty to find more information about the speeches.I want to find out what can be the most frequent NERs and POS in the speaches per class, decided to go with spacy
ner_org_counter = defaultdict(Counter)
ner_person_counter = defaultdict(Counter)
pos_verb_counter = defaultdict(Counter)
pos_adj_counter = defaultdict(Counter)
X = df_prepared['speech']
y = df_prepared["party"]
X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    stratify=y,
    random_state=26
)
#use training dataset for preventing data leakage
for text,label in zip(X_train,y_train):
    doc = nlp(text)

    #Distinguish NERs
    for ent in doc.ents:
        if ent.label_ == "ORG":
            ner_org_counter[label][ent.text] +=1
        elif ent.label_ == "PERSON":
            ner_person_counter[label][ent.text] += 1
    #Detect POS
    for token in doc:
        if token.pos_ == "VERB":
            pos_verb_counter[label][token.lemma_] += 1 
        elif token.pos_ == "ADJ":
            pos_adj_counter[label][token.lemma_] += 1
#printing the results for understading whether idea with NER and POS detectionw worth a try

'''top_n = 500

for label in y_train.unique():
    print(f"\nClass: {label}")
    print("Top ORG entities:")
    for org,count in ner_org_counter[label].most_common(top_n):
        print(f" {org}:{count}")
    print("Top Person entities:")    
    for person,count in ner_person_counter[label].most_common(top_n):
        print(f" {person}:{count}")    
    print("Top VERBs:")    
    for verb,count in pos_verb_counter[label].most_common(top_n):
        print(f" {verb}:{count}") 
    print("Top ADJs:")    
    for adj,count in pos_adj_counter[label].most_common(top_n):
        print(f" {adj}:{count}") '''

#The size of fetaures reached 23000.  Reducing noice by cutting the number of features 
top_n = 500
for label in ner_org_counter:
    ner_org_counter[label] = Counter(dict(ner_org_counter[label].most_common(top_n)))
for label in ner_person_counter:
    ner_person_counter[label] = Counter(dict(ner_person_counter[label].most_common(top_n)))    
for label in pos_verb_counter:
    pos_verb_counter[label] = Counter(dict(pos_verb_counter[label].most_common(top_n)))    
for label in pos_adj_counter:
    pos_adj_counter[label] = Counter(dict(pos_adj_counter[label].most_common(top_n))) 

 #Collecting all features, using set in order to avoid dublicates
all_features = set()
for label in ner_org_counter:
    for org in ner_org_counter[label]:
        all_features.add(f"ORG:{org}")
for label in ner_person_counter:
    for person in ner_person_counter[label]:
        all_features.add(f"PERSON:{person}")
for label in pos_verb_counter:
    for verb in pos_verb_counter[label]:
        all_features.add(f"VERB:{verb}")
for label in pos_adj_counter:
    for adj in pos_adj_counter[label]:
        all_features.add(f"ADJ:{adj}")  
#Convert to list for fix order
all_features = list(all_features)
#print(len(all_features))


#Building tokenizer using the feature I detected with Spacy
feature_set = set(all_features)

def bespoke_tokenizer(text):
    doc = nlp(text)
    tokens = []
    #NERs
    for ent in doc.ents:
        if ent.label_ in {"ORG", "PERSON"}:
            tag = f"{ent.label_}:{ent.text}"
            if tag in feature_set:
                tokens.append(tag)
    #POCs lemmas
    for token in doc:
        if token.pos_ in {"VERB", "ADJ"}:
            tag = f"{token.pos_}:{token.lemma_}"
            if tag in feature_set:
                tokens.append(tag)
    return tokens        
for text,label in zip(X_train,y_train):
    doc = nlp(text)

#print("Total feature form custom tokeniser", len(all_features))
 
 #Feeding Tfidvectorizer with a custom tokenizer
vectorizer_bespoke = create_vectorizer(ngram_range=(1,1),tokenizer=bespoke_tokenizer, lowercase=False, vocabulary=all_features,  max_features=3000)

#Vectorize the speeches with bespoke tokenizer
X_bespoke = vectorizer_bespoke.fit_transform(df_prepared['speech'])
y = df_prepared['party']

#Split samples 
X_train, X_test, y_train, y_test = train_test_split(X_bespoke,y,test_size=0.2,stratify=y, random_state = 26)

#Feature selection for noise reduction and preventing overfiting
selector = SelectKBest(chi2, k=50)
selector.fit(X_train,y_train)
X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test) 

#Train Random forest
random_f= RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=26)
random_f.fit(X_train_sel,y_train)
y_pred_random_f = random_f.predict(X_test_sel)
  
#Train SVM with linear kernel

svm = SVC(kernel='linear', class_weight="balanced", random_state=26)
svm.fit(X_train_sel, y_train)
y_pred_svm = svm.predict(X_test_sel)


#Evaluating 2 classifiers 
#print("Macro-average f1 score for Random forest with custom tokenizer:", f1_score(y_test,y_pred_random_f,average="macro"))
print("Classification_report for Random forest with custom tokenizer:\n", classification_report(y_test,y_pred_random_f))
#print("Macro-average f1 score for SVM with custom tokenizer:",f1_score(y_test,y_pred_svm,average="macro"))
print("Classification_report for SVM with custom tokenizer:\n", classification_report(y_test,y_pred_svm))