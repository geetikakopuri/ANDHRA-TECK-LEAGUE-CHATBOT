import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download necessary resources from NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Create a set of stopwords to remove them from tokens
stop_words = set(stopwords.words('english'))

# Initialize the stemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Split sentence into an array of words/tokens.
    A token can be a word or punctuation character, or number.
    """
    # Use word_tokenize from NLTK to tokenize the sentence
    tokens = word_tokenize(sentence)
    # Convert tokens to lowercase and remove stopwords
    tokens = [token.lower() for token in tokens if token.isalnum() and token.lower() not in stop_words]
    return tokens

def stem(word):
    """
    Stemming = finding the root form of the word.
    """
    return stemmer.stem(word)

def bag_of_words(tokenized_sentence, words):
    """
    Return a bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise.
    """
    # Stem each word in the tokenized sentence
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
