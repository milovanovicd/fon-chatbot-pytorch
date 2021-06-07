import nltk
import numpy as np
#nltk.download('punkt') # package with pretrained tokenizer
from nltk.stem.porter import PorterStemmer

# Kreiramo stemmer
stemmer = PorterStemmer()

"""
Tokenization
Delimo rečenicu u niz reči/tokena
Token može biti reč,interpukcijski znak ili broj
"""
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

"""
Stemming
Pronalazimo koren reči
Primer:
words = ["take", "taking", "taking"]
words = [stem(w) for w in words]
-> ["tak", "tak", "tak"]
"""
def stem(word):
    return stemmer.stem(word.lower())

"""
bag of words niz:
1 za svaku reč koja postoji u rečenici u odnosu na niz svih reči, 
0 u suprotnom
Primer:
sentence = ["hello", "how", "are", "you"]
all_words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
bag_of_words   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
"""
def bag_of_words(tokenized_sentence, all_words):
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words: 
            bag[idx] = 1.0

    return bag