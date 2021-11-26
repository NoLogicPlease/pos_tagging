import copy
import os
import re
import tarfile
import zipfile
import numpy as np
import pandas as pd
from Tokenizer import Tokenizer
from dataset_tools import create_dataset, create_trainable
from tqdm import tqdm
from urllib import request
import collections
import tensorflow as tf
import gensim
import gensim.downloader as gloader
from tensorflow import keras
from Model import create_model, train_model, predict_data

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
x_train, y_train, tok = create_trainable(df[df['split'] == 'train'], v3_val_to_key, max_seq_len, num_classes=num_classes)
x_val, y_val, tok = create_trainable(df[df['split'] == 'val'], v3_val_to_key, max_seq_len, num_classes=num_classes)

compile_info = {
    'optimizer': keras.optimizers.Adam(learning_rate=1e-3),
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy']
}

training_info = {
    'verbose': 1,
    'epochs': 10,
    'batch_size': 64,
    'callbacks': [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=10)]
}
bidirect_model = create_model(compile_info, v3_val_to_key, EMBEDDING_SIZE, max_seq_len,
                              num_labels=46, embedding_matrix=v3_matrix)

model = train_model(model=bidirect_model, x_train=x_train, y_train=y_train,
                    x_val=x_val, y_val=y_val, training_info=training_info)
