import numpy as np
import pdb

def next_words(model_x, input_str_x, num_predict_x):
    pdb.set_trace()
    try:
        return prediction(model_x, input_str_x, num_predict_x)
    except:
        print("Please, try again")
        return prediction(model_x, model_x.get_first_iter(), num_predict_x)
    
def prediction(model_y, input_str_y, num_predict_y):
        output = ""
        for i in range(num_predict_y):
            token = model_y.tokenizer.texts_to_sequences([input_str_y])
            
            prediction = model_y.model.predict(token, verbose = 0)
            predict = model_y.index_word[np.argmax(prediction[0])]
            output += predict + " " 
            
            input_str_y = input_str_y.split(" ", 1)[1]
            input_str_y += " " + predict
        return output