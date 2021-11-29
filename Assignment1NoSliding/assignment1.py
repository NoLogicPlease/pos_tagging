import copy
import os
import re
import tarfile
import zipfile
import numpy as np
import pandas as pd
from Tokenizer import Tokenizer
from dataset_tools import create_dataset, create_trainable, get_num_classes
from tqdm import tqdm
from urllib import request
import collections
import tensorflow as tf
import gensim
from sklearn.metrics import f1_score
import gensim.downloader as gloader
from tensorflow import keras
from Model import Model

WINDOW_LENGTH = 5
EMBEDDING_SIZE = 50

df, max_seq_len = create_dataset()

emb_model = gloader.load(f"glove-wiki-gigaword-{50}")
glove_dict = emb_model.key_to_index
glove_matrix = emb_model.vectors

tokenizer = Tokenizer(df[df['split'] == 'train']['text'], EMBEDDING_SIZE, glove_dict, glove_matrix)
tokenizer.tokenize()
v2_val_to_key = tokenizer.get_val_to_key()
v2_matrix = tokenizer.build_embedding_matrix()

tokenizer.dataset_sentences = df[df['split'] == 'val']['text']
tokenizer.tokenize()
v3_matrix = tokenizer.build_embedding_matrix()
v3_val_to_key = tokenizer.get_val_to_key()

tokenizer.dataset_sentences = df[df['split'] == 'test']['text']
tokenizer.tokenize()
v4_matrix = tokenizer.build_embedding_matrix()
v4_val_to_key = tokenizer.get_val_to_key()

num_classes = get_num_classes(df[df['split'] == 'train'])
x_train, y_train, tok = create_trainable(df[df['split'] == 'train'], v3_val_to_key, max_seq_len,
                                         num_classes=num_classes)
x_val, y_val, tok = create_trainable(df[df['split'] == 'val'], v3_val_to_key, max_seq_len, num_classes=num_classes)
x_test, y_test, tok = create_trainable(df[df['split'] == 'test'], v4_val_to_key, max_seq_len, num_classes=num_classes)

compile_info = {
    'optimizer': keras.optimizers.Adam(learning_rate=1e-3),
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy']
}

training_info = {
    'verbose': 1,
    'epochs': 50,
    'batch_size': 64,
    'callbacks': [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=10)]
}

model_params = {
    'compile_info': compile_info,
    'value_to_key': v4_val_to_key,
    'embedding_dim': EMBEDDING_SIZE,
    'max_seq_len': max_seq_len,
    'num_labels': num_classes,
    'embedding_matrix': v3_matrix
}


# BASELINE
baseline_class = Model('baseline', **model_params)
baseline_class.train_model(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, training_info=training_info)

'''
# GRU
gru_class = Model('gru', **model_params)
gru_class.train_model(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, training_info=training_info)

# TWO LSTM
twolstm_class = Model('two_lstm', **model_params)
twolstm_class.train_model(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, training_info=training_info)


# TWO DENSE
twodense_class = Model('two_dense', **model_params)
twodense_class.train_model(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, training_info=training_info)
'''

# baseline_class.embedding_matrix = v4_matrix
# baseline_class.value_to_key = v4_val_to_key


# model.layers[idx_of_your_embedding_layer].set_weights(my_new_embedding_matrix)

baseline_class.model.layers[0].set_weights([v4_matrix])

y_pred = baseline_class.predict_data(x_test, prediction_info=prediction_info)
y_pred = [np.argmax(el) for el in y_pred]  # from one hot to label index
y_true = [np.argmax(el) for el in y_test]
f1_score = f1_score(y_true, y_pred, average='macro')
print(f1_score)
