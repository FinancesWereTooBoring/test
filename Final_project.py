import pandas as pd
import numpy as np
import string
import nltk
import keras


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

#%% Preprocessing data
input_text = open('Text_lecture1.txt', encoding = 'utf8').read()

text = ""
for line in input_text:
    removed_enter = line.replace('\n', ". ") #remove the enters from the YouTube transcript
    final_line = removed_enter.replace(" Music", "")
     
    text += final_line

string.punctuation = string.punctuation.replace(".", "")
    
preprocessed_text = ""
for char in text:
    if char not in string.punctuation:
        preprocessed_text += char.lower()

preprocessed_text = preprocessed_text.lower()

#%%statistics
#Sentences
sentences = nltk.sent_tokenize(preprocessed_text)
num_sentences = len(sentences)

#words

string.punctuation = string.punctuation + '.' #add period again, to only look at the different words
word_text = ""
for char in preprocessed_text:
    if char not in string.punctuation :
        word_text += char
    
words = nltk.word_tokenize(word_text)
num_words = len(words)
#average sentence length
average_length = num_words/num_sentences

#unique words
unique_words = set(words)
num_unique = len(unique_words)

#%% Training

#Parameters based on the statistics
vocab_size = 1800  #chosen on the number of unique words, should not be more than the number of unique words
oov_tok = '<OOV>' #how the words outside the vocab_size are labelled
embedding_dim = 50 #the dimension of the word embedding
padding_type='post' #??
trunc_type='post' #??
seq_length = 50 #the length of sequences used in the LSTM

# tokenizes sentences
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts([preprocessed_text])
word_index = tokenizer.word_index #displays the index of the different words
tokens = tokenizer.texts_to_sequences([preprocessed_text])[0] #changes the words in text to their corresponding index

dataX = []
dataY = []

for i in range(0, len(tokens) - seq_length-1 , 1):
  seq_in = tokens[i:i + seq_length]
  seq_out = tokens[i + seq_length]

  if seq_out==1: #Skip samples where target word is OOV
    continue
    
  dataX.append(seq_in)
  dataY.append(seq_out)
 
training_data = len(dataX)
X = np.array(dataX)

# one hot encodes the output variable
y = np.array(dataY)
y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=seq_length ))       #The words embedding with dimensions is added
model.add(Bidirectional(keras.layers.LSTM(64)))                                 #The LSTM is added, not sure if we need Bidirectional and what value for LSTM is best
model.add(Dropout(0.2))                                                         #Dropout decreases overfitting, by leaving some LSTM cell out of the backpropagation for each epoch
model.add(Dense(vocab_size, activation='softmax'))                              #Dense makes sure the output vector is of size: vocab_size
# can add more LSTM layers, see the characterbased prediction LSTM model 2


model.compile(loss='categorical_crossentropy',  #this was used in the example but we can also use MSE or something like that
              optimizer='adam',                 #The optimizer instance, can also use gradient descent SGD
              metrics=['accuracy'])             #The way accuracy is measured

num_epochs = 10         #the number of epochs
batch = 64              #the batch size, bigger batch means less time to update and learn, thus worse results?
split = 0.2             #The train test split, 0.2 => 20% test, 80% train

history = model.fit(X, y, epochs=num_epochs, batch_size = batch, verbose=1, validation_split=split)

#%% Words prediction

#To retrieve the words need to go from index to word
index_word = dict(map(reversed, tokenizer.word_index.items()))

def next_words(input_str, num_predict):             #method to predict the next n words, given an input string
    output = ""
    for i in range(num_predict):
        token = tokenizer.texts_to_sequences([input_str])
        
        prediction = model.predict(token, verbose = 0)
        predict = index_word[np.argmax(prediction[0])]
        output += predict + " " 
        
        input_str = input_str.split(" ", 1)[1]
        input_str += " " + predict
    return output

#pick a random place to predict from already seen text

randomseed = np.random.randint(0, len(dataX)-1)
pattern = dataX[randomseed] #retrieves the sequence length from the randomseed
random_predict_seen = ""
for value in pattern:
    random_predict_seen += index_word[value] + " "


num_words_pred = 2   
output = next_words("Today we are going to talk about the brain of the brain of", num_words_pred)


"""
random_predict_unseen = "blablabla"     #add a sentence from which it predicts the next few words
output_unseen= next_words(random_predict_unseen, num_words_pred)
"""







    

    
    

