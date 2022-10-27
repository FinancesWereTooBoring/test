import nltk
import keras
import pickle
import numpy as np
import nltk
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Embedding
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
import pdb


class LSTM_word_pred():
    def __init__(self, data, seq_length, embedding_dim):
        self.preprocessed_text = data
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.vocab_size = len(set(nltk.word_tokenize(data))) - 100
        self.oov_tok = '<OOV>'
        self.padding_type='post'
        self.trunc_type='post'
        self.history = "Not yet defined"
        self.model = "Not yet defined"
        self.first_iter = "Not yet defined"
        self.dataX = "Not yet defined"
        self.tokenizer = ""
        self.index_word = ""
        
    def create_a_model(self, epochs_num, batch_size, ratio):
        self.tokenizer = Tokenizer(num_words = self.vocab_size, oov_token=self.oov_tok)
        self.tokenizer.fit_on_texts([self.preprocessed_text])
        word_index = self.tokenizer.word_index #displays the index of the different words
        tokens = self.tokenizer.texts_to_sequences([self.preprocessed_text])[0]
        self.index_word = dict(map(reversed, self.tokenizer.word_index.items()))
        

        self.dataX = []
        dataY = []
        for i in range(0, len(tokens) - self.seq_length-1 , 1):
            seq_in = tokens[i:i + self.seq_length]
            seq_out = tokens[i + self.seq_length]

            if seq_out==1: #Skip samples where target word is OOV
                continue
                
            self.dataX.append(seq_in)
            dataY.append(seq_out)
        
        X = np.array(self.dataX)
        y = np.array(dataY)
        y = np_utils.to_categorical(dataY)

        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size, self.embedding_dim, input_length=self.seq_length ))       #The words embedding with dimensions is added
        #self.model.add(Bidirectional(keras.layers.LSTM(64)))                                 #The LSTM is added, not sure if we need Bidirectional and what value for LSTM is best
        self.model.add(LSTM(64))
        self.model.add(Dropout(0.2))                                                         #Dropout decreases overfitting, by leaving some LSTM cell out of the backpropagation for each epoch
        self.model.add(Dense(self.vocab_size, activation='softmax'))                              #Dense makes sure the output vector is of size: vocab_size


        self.model.compile(loss='categorical_crossentropy',
                    optimizer='adam',                 
                    metrics=['accuracy'])  
        self.history = self.model.fit(X, y, epochs=epochs_num, batch_size = batch_size,
                                    verbose=1, validation_split=ratio)
        
    def get_first_iter(self): 
        
        randomseed = np.random.randint(0, len(self.dataX)-1)
        pattern = self.dataX[randomseed] #retrieves the sequence length from the randomseed
        random_predict_seen = ""
        for value in pattern:
            random_predict_seen += self.index_word[value] + " "
        return random_predict_seen
        
           
    
    def save__current_model(self, name_hist, name_model):
        self.model.save(name_model + ".h5")
        pickle.dump(self.history, open(name_hist + ".p", "wb"))
        
        
    
        

            



        
    
