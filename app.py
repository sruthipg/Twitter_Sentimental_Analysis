# Import necessary libraries
import pandas as pd
import re
import numpy as np
import contractions
from unidecode import unidecode
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import pickle

urlPattern        = r"(http://[^ ]*|https://[^ ]*|www\.[^ ]*)"
userPattern       = r'@[^\s]+'
alphaPattern      = r"[^a-z0-9<>]"
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1\1"
smileemoji        = r"[8:=;]['`\-]?[)d]+"
sademoji          = r"[8:=;]['`\-]?\(+"
neutralemoji      = r"[8:=;]['`\-]?[\/|l*]"
lolemoji          = r"[8:=;]['`\-]?p+"
pattern = r'<[^>]*>'


def load_contractions_dict(file_path):
    # Read the CSV file into a DataFrame
    contractions = pd.read_csv(file_path, index_col='Contraction')
    
    # Convert index and 'Meaning' column to lowercase
    contractions.index = contractions.index.str.lower()
    contractions['Meaning'] = contractions['Meaning'].str.lower()
    
    # Convert DataFrame to dictionary
    contractions_dict = contractions['Meaning'].to_dict()
    
    return contractions_dict

# Load the Word2Vec model
def load_word2vec_model(model_path):
    return Word2Vec.load(model_path)

# Preprocess the tweet
def preprocess_apply(tweet, contractions_dict):

    tweet = tweet.lower()

    # Replace all URls with '<url>'
    tweet = re.sub(urlPattern,'<url>',tweet)
    # Replace @USERNAME to '<user>'.
    tweet = re.sub(userPattern,'<user>', tweet)

    # Replace 3 or more consecutive letters by 2 letter.
    tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

    # Replace all emojis.
    tweet = re.sub(r'<3', '<heart>', tweet)
    tweet = re.sub(smileemoji, '<smile>', tweet)
    tweet = re.sub(sademoji, '<sadface>', tweet)
    tweet = re.sub(neutralemoji, '<neutralface>', tweet)
    tweet = re.sub(lolemoji, '<lolface>', tweet)
    tweet = re.sub(pattern, '', tweet)
    for contraction, replacement in contractions_dict.items():
        tweet = tweet.replace(contraction, replacement)

    # Remove non-alphanumeric and symbols
    tweet = re.sub(alphaPattern, ' ', tweet)

    # Adding space on either side of '/' to seperate words (After replacing URLS).
    tweet = re.sub(r'/', ' / ', tweet)
    return tweet

def load_tokenizer(tokenizer_file):
    with open(tokenizer_file, 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
    return loaded_tokenizer

def twoLetters(listOfTokens):
    twoLetterWord = []  # Initialize an empty list to store tokens that meet the conditions
    for token in listOfTokens:  # Iterate over each token in the input list
        if len(token) <= 2 or len(token) >= 21:  # Check if the length of the token satisfies the conditions
            twoLetterWord.append(token)  # If the condition is met, add the token to the list
    return twoLetterWord 

def gensim_lemmatizer(text, stop_words):
    lemmatizer = WordNetLemmatizer()  # Initialize WordNetLemmatizer
    return [lemmatizer.lemmatize(word) for word in text if word not in stop_words]

def removeWords(listOfTokens, listOfWords):
    return [token for token in listOfTokens if token not in listOfWords]

def processCorpus(corpus, language):
    stopwords = nltk.corpus.stopwords.words(language)

    processed_corpus = []
    for document in corpus:
        document = document.replace(u'\ufffd', '8')   # Replaces the ASCII '�' symbol with '8'
        document = document.replace(',', '')          # Removes commas
        document = document.rstrip('\n')              # Removes line breaks
        document = document.casefold()                # Makes all letters lowercase
        document = re.sub(r'[^a-zA-Z\s]', '', document)

        document = re.sub('\W_',' ', document)        # removes specials characters and leaves only words
        document = re.sub("\S*\d\S*"," ", document)   # removes numbers and words concatenated with numbers IE h4ck3r. Removes road names such as BR-381.
        document = re.sub("\S*@\S*\s?"," ", document) # removes emails and mentions (words with @)
        document = re.sub(r'http\S+', '', document)   # removes URLs with http
        document = re.sub(r'www\S+', '', document)    # removes URLs with www


        listOfTokens = word_tokenize(document) # It tokenizes the processed document.
        twoLetterWord = twoLetters(listOfTokens) # It identifies and removes two-letter words.

        listOfTokens = removeWords(listOfTokens, stopwords)
        listOfTokens = removeWords(listOfTokens, twoLetterWord)
        listOfTokens = gensim_lemmatizer(listOfTokens, stopwords) # It removes stopwords and lemmatizes the remaining tokens.
        processed_document = " ".join(listOfTokens)
        processed_document = unidecode(processed_document)  # Apply unidecode to the processed document string
        processed_corpus.append(processed_document)

    return processed_corpus

def predict_sentiment(tokenized_processed_corpus, keras_model, tokenizer, max_sequence_length):
    processed_sequences = tokenizer.texts_to_sequences(tokenized_processed_corpus)
    padded_processed_sequences = pad_sequences(processed_sequences, maxlen=max_sequence_length)
    sentiment_prediction = keras_model.predict(padded_processed_sequences)
    print(sentiment_prediction)
    predicted_sentiment = 0 if sentiment_prediction[0][0] > sentiment_prediction[0][1] else 1

    return predicted_sentiment


# Streamlit app
def main():
    st.title("Twitter Sentiment Analysis")
    language = 'english'

    # Load Contractions 
    contractions_file_path = "/Users/sruthipg/Documents/Twitter/CA_2/contractions.csv"
    contractions_dict = load_contractions_dict(contractions_file_path)

    # Load Word2Vec model
    vector_model_path = "/Users/sruthipg/Documents/Twitter/CA_2/word2vec.model"
    w2v_model = load_word2vec_model(vector_model_path)
    vocab_size = len(w2v_model.wv)

    # Load Keras model
    keras_model_path = "/Users/sruthipg/Documents/Twitter/CA_2/keras_model.h5"
    keras_model = tf.keras.models.load_model(keras_model_path)

    # Load tokenizer and max_sequence_length
    tokenizer_path = "/Users/sruthipg/Documents/Twitter/CA_2/tokenizer.pickle"
    tokenizer = load_tokenizer(tokenizer_path)  # Load your tokenizer
    max_sequence_length = 500  # Set your max sequence length

    # Input for new tweet
    new_tweet = st.text_input("Enter a new tweet:")

    if st.button("Predict Sentiment"):
        processed_text = preprocess_apply(new_tweet, contractions_dict)
        texts = []
        texts.append(processed_text)
        processed_corpus = processCorpus(texts, language)
        tokenized_processed_corpus = [sentence.split() for sentence in processed_corpus]
        sentiment = predict_sentiment(tokenized_processed_corpus, keras_model, tokenizer, max_sequence_length)
        print(sentiment)
        if sentiment == 1:
            predicted_sentiment = "Positive"
        else: predicted_sentiment = "Negative"
        st.write(f'The predicted sentiment is: {predicted_sentiment}')
    
    
if __name__ == "__main__":
    main()
