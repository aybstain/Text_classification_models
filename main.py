#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import gradio as gr


# Define a function to preprocess the input text
def preprocess_text(text):
    # Replace special characters, numbers, and punctuation marks with spaces
    text = ''.join([c if c.isalpha() else ' ' for c in text])
    # Convert to lowercase
    text = text.lower()
    # Tokenize with nltk
    nltk.download('punkt')
    words = nltk.word_tokenize(text)
    # Remove stop words with nltk
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Stem with nltk
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    # Join the processed words
    processed_text = ' '.join(words)
    return processed_text

# Create a dictionary to map class indices to class labels
class_dict = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}

# Function to get class label from the model prediction
def get_class_label(text):
    tokenizer = Tokenizer(num_words=10000)
    max_length = 126
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    # Convert the preprocessed text to sequences using the tokenizer
    seq = tokenizer.texts_to_sequences([preprocessed_text])
    # Pad the sequence
    padded_seq = pad_sequences(seq, maxlen=max_length, padding='post')
    # Make prediction using the trained CNN model
    prediction = model_cnn.predict(padded_seq)
    # Get the class index with highest probability
    class_index = np.argmax(prediction)
    # Get the class label from the dictionary
    class_label = class_dict[class_index]
    return class_label


# Create the Gradio interface
gr_interface = gr.Interface(fn=get_class_label, 
                            inputs=gr.Textbox(textarea=False, placeholder="Enter your text here..."),
                            outputs=gr.Label())

# Launch the interface
gr_interface.launch(share=True)

