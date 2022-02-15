"""
Author: Sarah Ackerschewski
Bachelor Thesis
WS 21/22

https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/
"""
import random
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

def read_train_data():
    german_sent = []
    count = 0
    with open("../data/MT_data/europarl-v7.de-en.de", "r", encoding="UTF-8") as de_f:
        for line in de_f.readlines():
            if count <= 10000:
                line = line.strip()
                german_sent.append(line)
                count += 1

    english_sent = []
    count = 0
    with open("../data/MT_data/europarl-v7.de-en.en", "r", encoding="UTF-8") as en_f:
        for line in en_f.readlines():
            if count <= 10000:
                line = line.strip()
                english_sent.append(line)
                count += 1

    return german_sent, english_sent


def _create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def _encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X



# one hot encode target sequence
def _encode_output(sequences, vocab_size):
    ylist = list()

    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = np.array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


def preprocess(ger_sent, eng_sent):
    randomized = list(zip(ger_sent, eng_sent))
    random.shuffle(randomized)
    ger_sent, eng_sent = zip(*randomized)
    ger_sent = ger_sent[:10000]
    eng_sent = eng_sent[:10000]
    len_dict = {}

    ger_tokenizer = _create_tokenizer(ger_sent)
    ger_vocab_size = len(ger_tokenizer.word_index)+1
    ger_length = max(len(line.split()) for line in ger_sent)
    len_dict["src_voc"] = ger_vocab_size
    len_dict["src_len"] = ger_length

    eng_tokenizer = _create_tokenizer(eng_sent)
    eng_vocab_size = len(eng_tokenizer.word_index)+1
    eng_length = max(len(line.split()) for line in eng_sent)
    len_dict["tar_voc"] = eng_vocab_size
    len_dict["tar_len"] = eng_length

    xtrain = _encode_sequences(ger_tokenizer, ger_length, ger_sent[:9000])

    ytrain = _encode_sequences(eng_tokenizer, eng_length, eng_sent[:9000])
    print(ytrain.shape)
    ytrain = _encode_output(ytrain, eng_vocab_size)
    #y_encoder = OneHotEncoder(handle_unknown = 'ignore')
    #y_encoder.fit(ytrain)
    #ytrain = y_encoder.transform(ytrain).toarray()
    #ytrain =  ytrain.reshape(ytrain.shape[0], ytrain.shape[1], 1)

    #print(ytrain.shape)

    xtest = _encode_sequences(ger_tokenizer, ger_length, ger_sent[9000:])
    ytest = _encode_sequences(eng_tokenizer, eng_length, eng_sent[9000:])
    ytest = _encode_output(ytest, eng_vocab_size)
    #ytest = y_encoder.transform(ytest).toarray()#.reshape(ytest.shape[0], eng_length, eng_vocab_size)

    return xtrain, ytrain, xtest, ytest, len_dict, eng_tokenizer


def translator(n_units=256):
    ger, eng = read_train_data()

    trainX, trainY, testX, testY, len_dict, eng_tokenizer = preprocess(ger, eng)

    src_vocab = len_dict["src_voc"]
    print(src_vocab)
    src_timesteps = len_dict["src_len"]
    tar_vocab = len_dict["tar_voc"]
    print(tar_vocab)
    tar_timesteps = len_dict["tar_len"]
    print(tar_timesteps)


    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))

    model.compile(optimizer='adam', loss='categorical_crossentropy')
    print("shapes train", trainX.shape, trainY.shape)
    model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(testX, testY), verbose=2)

    return model, eng_tokenizer


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate target given source sequence
def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)[0]
    integers = [np.argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)