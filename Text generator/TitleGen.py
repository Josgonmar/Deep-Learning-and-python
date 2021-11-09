#Deep Learning neural network that creates new headlines from a NYT dataset of news.
import os
from tensorflow.keras.models import load_model
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras import utils
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

data_path = 'Training_data/'
words_count = 0
max_len = 0
def import_headlines(data_path):
    temp_headlines = []
    for file in os.listdir(data_path):
        if 'Articles' in file:
            headlines_raw = pd.read_csv(data_path + file)
            temp_headlines.extend(list(headlines_raw.headline.values))
    temp_headlines = [hl for hl in temp_headlines if hl != 'Unknown'] #We get rid of the 'Unknown' data headlines
    return temp_headlines
    
def token_my_text(headline_list):
    global words_count
    tokens = Tokenizer()
    tokens.fit_on_texts(headline_list)
    text_sequence = create_token_subsets(tokens,headline_list)
    words_count = len(tokens.word_index)+1
    return text_sequence
    
def create_token_subsets(token,headlines):
    input_sequence = []
    for line in headlines:
        token_list = token.texts_to_sequences([line])[0]
        for i in range(1,len(token_list)):
            partial_sequence = token_list[:i+1]
            input_sequence.append(partial_sequence)
    return input_sequence

def padding_sequences(text_sequence):
    global max_len
    max_len = max([len(x) for x in text_sequence])
    text_sequence = np.array(pad_sequences(text_sequence,maxlen=max_len,padding='pre'))
    return text_sequence

def create_and_compile_model(predictors,labels):
    global max_len, words_count
    input_len = max_len - 1
    
    model = Sequential()
    model.add(Embedding(words_count, 10, input_length=input_len))
    model.add(LSTM(100))
    model.add(Dropout(0.1))  
    model.add(Dense(words_count, activation='softmax'))
    #model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(predictors,labels,epochs=30,verbose=1)
    return model
    
def predict_text(headlines,seed,model):
    global max_len
    token = Tokenizer()
    token.fit_on_texts(headlines)
    txt_seed = seed
    for i in range(1,5):
        token_seed = token.texts_to_sequences([txt_seed])[0]
        token_seed = pad_sequences([token_seed],maxlen=max_len-1,padding='pre')
        prediction = model.predict_classes(token_seed,verbose=0)
        txt_seed = txt_seed + " " + token.sequences_to_texts([prediction])[0]
    print("Predicted headline:", txt_seed)
    
if __name__ == "__main__":
    all_headlines = import_headlines(data_path)
    txt_sequence = token_my_text(all_headlines)
    txt_sequence = padding_sequences(txt_sequence)
    mymodel = load_model('NYT_headlines_generator')
    text_seed = input('Enter seed text to predict: ')
    predict_text(all_headlines,text_seed,mymodel)
    
    #Once the model is trained and done, these sentences are no longer needed
    #predictors = txt_sequence[:,:-1]
    #targets = txt_sequence[:,-1]
    #labels = utils.to_categorical(targets,num_classes=words_count)
    #mymodel = create_and_compile_model(predictors,labels)
    #mymodel.save('NYT_headlines_generator')