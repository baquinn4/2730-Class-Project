
import csv
import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
from keras.preprocessing import text
from keras.utils import np_utils
import keras
from keras.preprocessing import sequence
import numpy as np
import pickle


# Basic packages
import collections
import matplotlib.pyplot as plt
from pathlib import Path

# Packages for data preparation
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Packages for modeling
from keras import models
from keras import layers
from keras import regularizers


from keras.layers import Dropout
from keras.models import model_from_json
from keras.models import load_model
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from nltk.tokenize import RegexpTokenizer
import time

start_t = time.clock()

tk = Tokenizer()
NUM_WORDS_DICT = 100000
VAL_SIZE = 200000
NUM_EPOCHS = 10
BATCH_SIZE = 512
MAX_LENGTH = 24
import sys

path = os.getcwd()
os.chdir( path + "/dataset")

df = pd.DataFrame()

def rem_stopwords(input_tweet):


	sw_list = stopwords.words('english')
	
	indicator_list = ["good","bad","love","no","hate"]

	words = input_tweet.split()
	clean_list = [word for word in words if (word not in sw_list or word in indicator_list)
	and len(word) > 1]
	
	return " ".join(clean_list)


def clean_tweet(input_tweet):

	return re.sub(r'@\w+', '', input_tweet)


def load_and_split_data(csvfile):
	
	print("\nParsing dataset file.......")
	df = pd.read_csv(csvfile,error_bad_lines=False)

	

	#create subfile of really large dataset
	
	
	
	
	df = pd.read_csv("Smaller_Dataset.csv",error_bad_lines=False)
	df = df[['Sentiment','SentimentText']]
	df.SentimentText = df.SentimentText.apply(rem_stopwords).apply(clean_tweet)
	print(df.head(10))
	X_train, X_test, y_train, y_test = train_test_split(df.SentimentText, df.Sentiment, test_size=0.2,
		random_state=123)

	

	df_col_len = int(df['SentimentText'].str.encode(encoding='utf-8').str.len().max())
	print(df_col_len)

	print('# Train data samples:', X_train.shape[0])
	print('# Test data samples:', X_test.shape[0])
	assert X_train.shape[0] == y_train.shape[0]
	assert X_test.shape[0] == y_test.shape[0]
	
	tk = Tokenizer(num_words=NUM_WORDS_DICT,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
		lower=True,
		split=" ")

	tk.fit_on_texts(X_train)

	xtrain_seq = tk.texts_to_sequences(X_train)
	xtest_seq = tk.texts_to_sequences(X_test)

	seq_lengths = X_train.apply(lambda x: len(x.split(' ')))
	print(seq_lengths.describe())
	X_train_seq_trunc = pad_sequences(xtrain_seq, maxlen=MAX_LENGTH)
	X_test_seq_trunc = pad_sequences(xtest_seq, maxlen=MAX_LENGTH)
	print(X_train_seq_trunc[10])

	le = LabelEncoder()
	y_train_le = le.fit_transform(y_train)
	y_test_le = le.fit_transform(y_test)
	y_train_cate = to_categorical(y_train_le)
	y_test_cate = to_categorical(y_test_le)


	X_train_emb, X_valid_emb, y_train_emb, y_valid_emb = train_test_split(X_train_seq_trunc, y_train_cate, test_size=0.1, random_state=37)
	assert X_valid_emb.shape[0] == y_valid_emb.shape[0]
	assert X_train_emb.shape[0] == y_train_emb.shape[0]

	print('Shape of validation set:',X_valid_emb.shape)
	
	datalist = [[X_train_seq_trunc,y_train_cate,X_test_seq_trunc,y_test_cate],[X_train_emb,y_train_emb,X_valid_emb,y_valid_emb]]

	print(datalist[0][1])
	print(len(df))
	return datalist

def load_embeddings(glovefile,dimension):

	gf = glovefile
	embedded_dict = {}
	glove = open(gf)
	for line in glove:
		values = line.split()
		word = values[0]
		vector = np.asarray(values[1:], dtype='float32')
		embedded_dict[word] = vector
	glove.close()

	embedded_matrix = np.zeros((NUM_WORDS_DICT, dimension))

	for w, i in tk.word_index.items():
		if i < NUM_WORDS_DICT:
			vect = embedded_dict.get(w)

			if vect is not None:
				embedded_matrix[i] = vect
		else:
			break

	return embedded_matrix

def learn_embeddings(filename):

	#test to see if we can find some Sentiment text in dictionary

	text = ["good","bad","exciting","wow","hate"]

	for word in text:
		if word in embedded_dict.keys():
			print("Found the word {} in the dict".format(word))

def validation_model(model, X_train, y_train, X_valid, y_valid):

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
    
    hist = model.fit(X_train,y_train,epochs=NUM_EPOCHS,batch_size=BATCH_SIZE,validation_data = (X_valid, y_valid),callbacks=[earlyStopping],verbose = 1)
    
    
    return hist

def cnn_model(filename,booli):

	e_matrix = load_embeddings(filename,100)
	print(e_matrix)

	emb_model = models.Sequential()

	if booli == True:
		emb_model.add(layers.Embedding(NUM_WORDS_DICT, 100, input_length=MAX_LENGTH))
	else:
		emb_model.add(layers.Embedding(NUM_WORDS_DICT, 100, input_length=MAX_LENGTH))
		emb_model.layers[0].set_weights([e_matrix])
		emb_model.layers[0].trainable = False

	emb_model.add(layers.Flatten())
	emb_model.add(layers.Dense(2, activation='softmax'))
	emb_model.summary()
	
	
	return emb_model

def test_model(model, X_train_seq_trunc, y_train_cate, X_test_seq_trunc, y_test_cate, epoch_stop):
    
    
    model.fit(X_train_seq_trunc, y_train_cate, epochs=epoch_stop, batch_size=BATCH_SIZE, verbose=1)
    results = model.evaluate(X_test_seq_trunc, y_test_cate)
    
    return results

def rnn_model(filename):

	model = models.Sequential()
	emb_model.add(layers.Embedding(NUM_WORDS_DICT, 100, input_length=MAX_LENGTH))
	model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.50))
	model.add(Dense(2, activation='softmax'))
	model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
	return model

def eval_metric(history, metric_name):
    
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]

    e = range(1, NUM_EPOCHS + 1)

    plt.plot(e, metric, 'bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
    plt.legend()
    plt.show()

def real_test(trained_model,sample_data,Tokenizer):
	tk = Tokenizer
	real_list = []
	real_list_np = np.zeros(24,1)
	tk = RegexpTokenizer(r'\w+')
	sample_data_list = tk.tokenize(sample_data)
	labels = ["positivie","negative"]


listofdata = load_and_split_data("Smaller_Dataset.csv")

m = cnn_model("glove_6B_100d.txt", False)
m2 = cnn_model("glove_6B_100d.txt", True)

emb_hist = validation_model(m,listofdata[1][0],listofdata[1][1],listofdata[1][2],listofdata[1][3])
emb_hist2 = validation_model(m2,listofdata[1][0],listofdata[1][1],listofdata[1][2],listofdata[1][3])
#emb_hist3 = validation_model(m3,listofdata[1][0],listofdata[1][1],listofdata[1][2],listofdata[1][3])



print(emb_hist.history['accuracy'][-1])
#print(emb_hist3.history['accuracy'][-1])
print(emb_hist2.history['accuracy'][-1])

learn_embeddings_file = "LE_model.sav"
pre_train_embeddings_file = "PT_model.sav"
#rnn_file = "rnn_model.sav"
#m3 = rnn_model("glove_6B_100d.txt")

emb_results = test_model(m,listofdata[0][0],listofdata[0][1],listofdata[0][2],listofdata[0][3],3)
emb_results2 = test_model(m2,listofdata[0][0],listofdata[0][1],listofdata[0][2],listofdata[0][3],3)
#emb_results3 = test_model(m3,listofdata[0][0],listofdata[0][1],listofdata[0][2],listofdata[0][3],3)
#pickle.dump(emb_results, open(learn_embeddings_file, 'w+'))
##pickle.dump(emb_results2, open(pre_train_embeddings_file, 'w+'))
#pickle.dump(emb_results3, open(rnn_model, 'rb'))

print('Test accuracy of pre-trained word embeddings model: {0:.2f}%'.format(emb_results[1]*100))
print('Test accuracy of learned embeddings model: {0:.2f}%'.format(emb_results2[1]*100))
#print('Test accuracy of RNN embeddings model: {0:.2f}%'.format(emb_results3[1]*100))
end_t = time.clock()

exc_t = end_t - start_t
print("Total time to train and test: " + str(exc_t))

print()
'''
eval_metric(emb_hist,'accuracy')
eval_metric(emb_hist2,'accuracy')
'''
#eval_metric(emb_hist3, 'accuracy')