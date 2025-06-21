#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
from pathlib import Path
import pandas as pd
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
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
    df["tokens_spacy"] = df["text"].apply(nlp)
    df.to_pickle("dataframe_parsed.pkl")


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
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    pass



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "p1-texts" / "novels" #(/Users/alinasysko/BBK/NLP/Coursework/p1-texts/novels") 
    #print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    #print(df.head())
    nltk.download("cmudict")
    #parse(df)
    #print(df.head())
    #print(nltk_ttr(df)) #Alina
    print(get_ttrs(df))
    print(get_fks(df))
    #df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(adjective_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

