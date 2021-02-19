from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import re


class LemmaTokenizer(object):
    '''
    Creates a functionb to allow me to Lemmatize within a count vectorizer
    '''
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        for t in word_tokenize(articles):
            if re.match('\b[^\d\W]+\b',t):
                return self.wnl.lemmatize(t)
            else:
                continue

def tokenize(text):
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]
