from preprocessing import *
from model import *
from predicting import*

actual_text = get_preprocessed_data(["Text_lecture1.txt", "Text_lecture2.txt", "Text_lecture3.txt"])
my_model = LSTM_word_pred(actual_text, 50, 50)
my_model.create_a_model(20, 64, 0.2)
next_words(my_model, "The brain is", 5)

