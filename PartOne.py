#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
from pathlib import Path
import pandas as pd
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
from math import log2
#from nltk.tokenize.punkt import PunktLanguageVars



nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    cmudict = nltk.corpus.cmudict.dict()
    tokens = []
    sentences = sent_tokenize(text)
    total_sentences = len(sentences)
    tokens.extend(nltk.word_tokenize(text))
    tokens_cleaned = [token.lower() for token in tokens if token.isalnum()]
    total_words = len(tokens_cleaned)
    total_syllables = sum (count_syl(token,d) for token in tokens_cleaned) 
    fk_grade = 0.39 * (total_words/total_sentences) + 11.8 * (total_syllables/total_words) - 15.59
    return fk_grade
    #pass


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word."""
    d = {}
    vowels =  "aeiouy"
    i = 0
    prev_vowel = False
    word = word.lower()
    if word in d:
        return max([len([ph for ph in pron if ph[-1].isdigit()]) for pron in d[word]])
    else:
        for char in vowels:
            if char in vowels:
                i += 1
                prev_vowel = True
            else:
                prev_vowel = False
        return max(1,i)    
    #pass


def read_novels(path=Path.cwd() / "p1-texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    #print(f"Looking in: {path}")
    data = []
    for file in path.glob("*.txt"):
        #print(file)
        #exit()
        title, author, year = file.stem.split("-")
        #print(title)
        text = file.read_text(encoding="utf-8")
        text_polished = text.replace('\n',' ')
        data.append({
            "text": text_polished,
            "title": title.strip(),
            "author": author.strip(),
            "year": year})
    df = pd.DataFrame(data)
    df = df.sort_values(by="year").reset_index(drop=True)
    return df
#a = read_novels()
#print(a)
   # pass


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    df["parsed"] = df["text"].apply(nlp)
    #renamed column token_spacy to parsed. need to change
    '''if len("token_spacy") > nlp.max_length:
        chunks = ["token_spacy"[i:i+nlp.max_length] for i in range(0,len("token_spacy"), nlp.max_length)]
        docs = [nlp(chunk) for chunk in chunks]
        return docs
    else:
        return ["token_spacy"]'''
        
    return df
    #df.to_pickle("dataframe_parsed.pkl")


    #pass


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    tokens = []
    tokens.extend(nltk.word_tokenize(text))
        #print(tokens)
    #punct_symbols = PunktLanguageVars()
    #punct_symbols_l = punct_symbols.punct_chars
    tokens_cleaned = [token.lower() for token in tokens if token.isalnum()]
    #return tokens_cleaned
    ttr = len(set(tokens_cleaned)) / len(tokens_cleaned)
    return ttr
#print(a)
   #pass


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    #results = {}
    #for i, row in df.iterrows():
    #    results[row["title"]] = nltk_ttr(row["text"])
    df["ttr"] = df["text"].apply(nltk_ttr) 
    return df



def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    #results = {}
    #fks_grade = []
    cmudict = nltk.corpus.cmudict.dict()
    #for i, row in df.iterrows():
        #results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
        #title = row["title"]
        #fks = round(fk_level(row["text"], cmudict), 4)
        #fks_grade.append(fks) 
    df["fks"] = df["text"].apply(lambda text: round(fk_level(text, cmudict), 4))
    
    return df


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list.
    Calculating PMI using formula I(x, y) = log2 (P(x,y)/ P(x)P(y) where P(x,y) is joint probability when hear and subject is in novel together and P(x) and P(y) are independent probabilities"""
    sub_ind_counter = Counter()
    sub_verb_counter = Counter() #P(x,y)
    verb_ind_counter = 0
    pair_counter = 0
    corpus_size = len(doc)
    checking_list= []
    #using function subject_be_verb_count for receiving list of subjects
    syn_subj_options = ["nsubj", "nsubjpass"]
    syn_subj_l = subjects_by_verb_count(doc, target_verb)
    #print(syn_subj_l)
    #subjects_ind = []
    #calculating independing probability per word
    for token in doc:
        if token.lemma_ == target_verb and token.pos_ == "VERB":
            verb_ind_counter +=1
            for child in token.children:
                if child.text in syn_subj_l:
                    subj = child.text.lower()
                    sub_verb_counter[subj] +=1 #subject for this verb
                    pair_counter +=1 #both subject and 
                    #print(subj)
        #calculate number of the syntactiv objects when hear is not a target. I chose to still calculate the number only if the words are syntactic objects because this approach is analyzing syntactic patterns
        elif token.pos_ == "VERB":
            for child in token.children:
                if child.text in syn_subj_l and child.dep_ in syn_subj_options:
                    subj = child.text.lower()
                    sub_ind_counter[subj] += 1
                    print(subj)


    #Computing probabilities. I decided to use cooccurance (pair counter) in order to focus on measurement how strong subject and verb are associated. I also could use the whole corpus (cleaned one) as we have in Jurafsky, chapter 6 but this approach will use many itrelevant tokens so I chose pairs 
        '''pmi_scores = {}
        for subj in sub_verb_counter:
            prob_sv = sub_verb_counter[subj]/pair_counter
            prob_s = sub_ind_counter[subj]/pair_counter
            prob_v = verb_ind_counter/pair_counter
            pmi = log2(prob_sv/(prob_s*prob_v))
            pmi_scores[subj] = pmi
    print(checking_dictionary = {subj:pmi} )
            #print(subj)'''

        






def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list.
    We are searchning nominal subjects("smbd hears")   and passive nominal subject ("hear smth") """
    subjects = []
    syn_sub_counter = Counter()
    syn_subj_l = ["nsubj", "nsubjpass"]
    for token in doc:
        if token.lemma_ == verb and token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ in syn_subj_l:
                   subjects.append(child.text.lower())
    # [child for child, count in syn_sub_counter.most_common(10)]
    #counting the frequency
    subjects_frequency = Counter(subjects)
    #sorting the subjects by frequency (the most frequent will be the first)
    sorted_subjects = sorted(subjects_frequency.items(), key=lambda x: x[1], reverse= True)
    #testing that we have decreasing frequency
    #print(sorted_subjects[:10])
    #results
    return [subject for subject, count in sorted_subjects[:10]]

                    


def syntactic_objects(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    
    #title_syn_obj = {}
    syn_obj_l = ["dobj", "podj", "iobj"]
    #for i in range(len(df)):
        #syntactic_obj.update([token.dep_ for token in "tokens_spacy" if token.dep_ != " "])
        #ten_syntatic_obj = syntactic_obj.most_common(10)
        #title = df['title']
        #tokens = df['tokens_parsed']
    syn_obj_counter = Counter()
    for token in doc:
        if token.dep_ in syn_obj_l:
            syn_obj_counter[token.text.lower()] += 1
    syn_obj = [word for word, count in syn_obj_counter.most_common(10)]

    #print(syn_obj) 
    #for token in syn_obj:
    #    print(f"{token.text} - {token.dep_}")
            #title_syn_obj[title] = syn_obj
    return syn_obj
    
    #for title, deps in  title_syn_obj():
    #   print(f"{title}:{deps}")




if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "p1-texts" / "novels" #(/Users/alinasysko/BBK/NLP/Coursework/p1-texts/novels") 
    #print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    #print(df.head())
    nltk.download("cmudict")
    parse(df)
    #print(df.head())
    #print(nltk_ttr(df)) #Alina
    #print(get_ttrs(df))
    #print(get_fks(df))
    #df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    #print(adjective_counts(df))
    
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")
    
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")

    '''for i, row in df.iterrows():
        print(row["title"])
        print(syntactic_objects(row["parsed"]))
        print("\n")'''


