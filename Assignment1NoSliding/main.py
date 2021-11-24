import copy
import os
import re
import tarfile
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from urllib import request
import collections
import tensorflow as tf
import gensim
import gensim.downloader as gloader
from tensorflow import keras
from tensorflow.keras import layers

WINDOW_LENGTH = 5
EMBEDDING_SIZE = 50


class TwoWay:
    def __init__(self):
        self.d = {}

    def add(self, k, v):
        self.d[k] = v
        self.d[v] = k

    def remove(self, k):
        self.d.pop(self.d.pop(k))

    def get(self, k):
        return self.d[k]


class KerasTokenizer(object):
    """
    A simple high-level wrapper for the Keras tokenizer.
    """

    def __init__(self, build_embedding_matrix=False, embedding_dimension=None, embedding_model=None,
                 embedding_model_type=None, tokenizer_args=None):
        if build_embedding_matrix:
            assert embedding_model_type is not None
            assert embedding_dimension is not None and type(embedding_dimension) == int

        self.build_embedding_matrix = build_embedding_matrix
        self.embedding_dimension = embedding_dimension
        self.embedding_model_type = embedding_model_type
        self.embedding_model = embedding_model
        self.embedding_matrix = None
        self.vocab = None

        tokenizer_args = {} if tokenizer_args is None else tokenizer_args
        assert isinstance(tokenizer_args, dict) or isinstance(tokenizer_args, collections.OrderedDict)

        self.tokenizer_args = tokenizer_args

    def build_vocab(self, data, **kwargs):
        print('Fitting tokenizer...')
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(**self.tokenizer_args)
        self.tokenizer.fit_on_texts(data)  # [Hi = 1, I = 2, ...]
        print('Fit completed!')

        self.vocab = self.tokenizer.word_index

        if self.build_embedding_matrix:
            print('Checking OOV terms...')
            self.oov_terms = check_OOV_terms(embedding_model=self.embedding_model,
                                             word_listing=list(self.vocab.keys()))

            print('Building the embedding matrix...')
            self.embedding_matrix, self.embedding_model = build_embedding_matrix(embedding_model=self.embedding_model,
                                                                                 word_to_idx=self.vocab,
                                                                                 vocab_size=len(self.vocab) + 1,
                                                                                 embedding_dimension=self.embedding_dimension,
                                                                                 oov_terms=self.oov_terms,
                                                                                 dataset=data)
            print('Done!')

    def get_info(self):
        return {
            'build_embedding_matrix': self.build_embedding_matrix,
            'embedding_dimension': self.embedding_dimension,
            'embedding_model_type': self.embedding_model_type,
            'embedding_matrix': self.embedding_matrix.shape if self.embedding_matrix is not None else self.embedding_matrix,
            'embedding_model': self.embedding_model,
            'vocab_size': len(self.vocab) + 1,
        }

    def tokenize(self, text):
        return text

    def convert_tokens_to_ids(self, tokens):
        if type(tokens) == str:
            return self.tokenizer.texts_to_sequences([tokens])[0]
        else:
            return self.tokenizer.texts_to_sequences(tokens)

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.sequences_to_texts(ids)


dataset_folder = os.path.join(os.getcwd(), "Datasets")
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/dependency_treebank.zip"

dataset_path_zip = os.path.join(dataset_folder, "dependency_treebank.zip")


def download_dataset(download_path: str, url: str):
    if not os.path.exists(download_path):
        print("Downloading dataset...")
        request.urlretrieve(url, download_path)
        print("Download complete!")


def extract_dataset(download_path: str, extract_path: str):
    print("Extracting dataset... (it may take a while...)")
    with zipfile.ZipFile(download_path, 'r') as loaded_zip:
        loaded_zip.extractall(extract_path)
    print("Extraction completed!")


download_dataset(dataset_path_zip, url)
extract_dataset(dataset_path_zip, dataset_folder)
dataset_path = os.path.join(dataset_folder, "dependency_treebank")


def encode_dataset(dataset_path: str, debug: bool = True) -> (pd.DataFrame, int):
    dataframe_rows = []
    s_lengths = []
    for index, file in tqdm(enumerate(sorted(os.listdir(dataset_path)))):
        file_name = os.path.join(dataset_path, file)
        with open(file_name) as f:
            lines = f.readlines()
        full_file = ''.join(lines)  # since lines is a list we use a single string called full_file
        full_file = re.sub(r'(\t\d+)', '', full_file)  # remove numbers from each lines of dataset
        full_file = re.sub(r'(\t)', ' ', full_file)  # replace \t with a space
        sentences = full_file.split('\n\n')
        for s in sentences:  # separate all words from their tags
            text = ''.join(re.findall(r'.+ ', s))
            labels = ''.join(re.findall(r' .+', s))
            labels = re.sub(r' (.+)', r'\1 ', labels)
            labels = re.sub('\n', ' ', labels)
            s_lengths.append(len(labels.split(' ')))
            if len(labels.split(' ')) == 250:
                print(file)
                print(text)
            # split into train, val and test
            if index <= 100:
                split = 'train'
            elif 100 < index <= 150:
                split = 'val'
            else:
                split = 'test'

            # create a single row of dataframe
            dataframe_row = {
                "text": text,
                "POStagging": labels,
                "split": split,
            }
            dataframe_rows.append(dataframe_row)

    # transform the list of rows in a proper dataframe
    df = pd.DataFrame(dataframe_rows)
    df = df[["text",
             "POStagging",
             "split"]]
    dataframe_path = os.path.join(dataset_folder, "dependency_treebank_df.pkl")
    df.to_pickle(dataframe_path)
    return df, max(s_lengths)


df, max_seq_len = encode_dataset(dataset_path)
df['POStagging'] = df['POStagging'].str.lower()
df['text'] = df['text'].str.lower()

print(df)


def check_OOV_terms(embedding_model: gensim.models.keyedvectors.KeyedVectors,
                    word_listing):
    """
    Checks differences between pre-trained embedding model vocabulary
    and dataset specific vocabulary in order to highlight out-of-vocabulary terms.

    :param embedding_model: pre-trained word embedding model (gensim wrapper)
    :param word_listing: dataset specific vocabulary (list)

    :return
        - list of OOV terms
    """

    embedding_vocabulary = set(embedding_model.index_to_key)
    oov = set(word_listing).difference(embedding_vocabulary)
    print("Total OOV terms: {0} ({1:.2f}%)".format(len(list(oov)), float(len(list(oov))) / len(word_listing)))
    return list(oov)


def build_embedding_matrix(embedding_model: gensim.models.keyedvectors.KeyedVectors,
                           embedding_dimension: int,
                           word_to_idx,
                           vocab_size: int,
                           oov_terms,
                           dataset) -> np.ndarray:
    """
    Builds the embedding matrix of a specific dataset given a pre-trained word embedding model

    :param embedding_model: pre-trained word embedding model (gensim wrapper)
    :param word_to_idx: vocabulary map (word -> index) (dict)
    :param vocab_size: size of the vocabulary
    :param oov_terms: list of OOV terms (list)

    :return
        - embedding matrix that assigns a high dimensional vector to each word in the dataset specific vocabulary (shape |V| x d)
    """

    embedding_matrix = np.zeros((vocab_size, embedding_dimension), dtype=np.float32)

    for word, idx in tqdm(word_to_idx.items()):
        try:
            embedding_vector = embedding_model[word]
        except (KeyError, TypeError):
            # embedding_vector = np.random.uniform(low=-0.05, high=0.05, size=embedding_dimension)
            neighbour_words = []
            for sen in (dataset):  # look for word in sentence
                for i, wanted_word in enumerate(sen):
                    if wanted_word == word:
                        neighbour_words.append(sen[i - 1])  # append previous word in list of neighbours
                        neighbour_words.append(sen[i + 1])  # append next word in list of neighbours
            avg_matrix = np.zeros((len(neighbour_words), EMBEDDING_SIZE))  # initialize matrix of avgs

            length_in_vocab = 0  # to check if neighbours are OOV
            for i, el in enumerate(neighbour_words):
                try:
                    avg_matrix[i] = embedding_model[el]  # check not OOV
                    length_in_vocab += 1  # we don't want to use the zero columns of avg_matrix
                except (KeyError, TypeError):  # the model doesn't exist
                    pass
            if length_in_vocab == 0:
                embedding_vector = np.random.uniform(low=-0.05, high=0.05, size=embedding_dimension)
            else:
                embedding_vector = np.mean(avg_matrix[:length_in_vocab], axis=0)
                embedding_model.vectors = np.vstack(embedding_model.vectors, embedding_vector)
                embedding_model.key_to_index[word] = len(embedding_model.key_to_index)
                embedding_model.index_to_key.append(word)
                embedding_model.expandos["count"] = np.append(embedding_model.expandos["count"],
                                                              len(embedding_model.expandos["count"]))
        embedding_matrix[idx] = embedding_vector

    return embedding_matrix, embedding_model


def concatenate_embeddings(emb_model1, emb_model2, word_to_index_model2, emb_mat1, emb_mat2):
    emb_mat1 = copy.deepcopy(emb_mat1)
    for idx, word in tqdm(enumerate(word_to_index_model2.items())):
        try:
            check = emb_model1[word]
        except (KeyError, TypeError):
            try:
                check = emb_model2[word]
            except (KeyError, TypeError):
                emb_mat1 = np.vstack((emb_mat1, emb_mat2[idx]))
    return emb_mat1


tokenizer_args = {
    'oov_token': 1,  # The vocabulary id for unknown terms during text conversion
    'lower': True,
    'filters': ''
}

embedding_model_glove = gloader.load(f"glove-wiki-gigaword-{EMBEDDING_SIZE}")
v1_tokenizer = KerasTokenizer(tokenizer_args=tokenizer_args,
                              build_embedding_matrix=True,
                              embedding_model=embedding_model_glove,
                              embedding_dimension=EMBEDDING_SIZE,
                              embedding_model_type="glove")
# v2 =v1 + oov1
v1_tokenizer.build_vocab(df[df["split"] == "train"]['text'].values)

# v3 = v1 + oov1 + oov2
emb_matrix, emb_model = build_embedding_matrix(embedding_model=v1_tokenizer.embedding_model,
                                               embedding_dimension=EMBEDDING_SIZE,
                                               word_to_idx=v1_tokenizer.vocab, vocab_size=len(v1_tokenizer.vocab) + 1,
                                               oov_terms=None,
                                               dataset=df[df["split"] == "val"]['text'].values)
# v4 = v1 + oov1 + oov2 + oov3
emb_matrix, emb_model = build_embedding_matrix(embedding_model=emb_model,
                                               embedding_dimension=EMBEDDING_SIZE,
                                               word_to_idx=v1_tokenizer.vocab, vocab_size=len(v1_tokenizer.vocab) + 1,
                                               oov_terms=None,
                                               dataset=df[df["split"] == "test"]['text'].values)

quit()

v4_emb_matrix = concatenate_embeddings(emb_model1=v1_tokenizer.embedding_model,
                                       emb_model2=None,
                                       word_to_index_model2=None)


def create_trainable(dataset, tokenizer):
    """
    Converts input text sequences using a given tokenizer

    :param texts: either a list or numpy ndarray of strings
    :tokenizer: an instantiated tokenizer

    :return
        text_ids: a nested list on token indices
    """
    text_ids = tokenizer.convert_tokens_to_ids(dataset['text'])
    x_train = []
    y_train = []
    label_tokenizer = TwoWay()

    label_id = 1
    for sen in dataset["POStagging"]:
        for label in sen.split():
            try:
                check_label = label_tokenizer.d[label]
            except KeyError:
                label_tokenizer.add(label_id, label)
                label_id += 1

    for sen, tagging in zip(text_ids, dataset["POStagging"]):
        tmp = [0] * (max_seq_len - len(sen)) + sen
        x_train.append(tmp)
        tmp = [0] * (max_seq_len - len(sen)) + [label_tokenizer.get(e) for e in tagging.split()]
        y_train.append(tmp)
    return x_train, y_train, label_tokenizer


# Train, Val
x_train, y_train, label_tok = create_trainable(df[df["split"] == "train"], tokenizer)
x_val, y_val, label_tok = create_trainable(df[df["split"] == "val"], tokenizer)

x_train = np.array(x_train)
x_val = np.array(x_val)
y_train = np.array(y_train)
y_val = np.array(y_val)
num_labels = 46
y_train = tf.keras.utils.to_categorical(y_train, 46)
y_val = tf.keras.utils.to_categorical(y_val, 46)

print(x_train.shape)


# per vedere la conversione
# tokens = tokenizer.convert_ids_to_tokens(list(text_ids)) #verify token
# print(tokens)


def create_model(compile_info: dict) -> keras.Model:
    bidirect_model = keras.models.Sequential()
    bidirect_model.add(layers.Embedding(input_dim=len(tokenizer.vocab) + 1,
                                        output_dim=EMBEDDING_SIZE,
                                        input_length=max_seq_len,
                                        mask_zero=True,
                                        weights=tokenizer.embedding_matrix if tokenizer.embedding_matrix is None else [
                                            tokenizer.embedding_matrix],
                                        trainable=False
                                        ))
    bidirect_model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
    bidirect_model.add(layers.TimeDistributed(layers.Dense(num_labels, activation="softmax")))

    bidirect_model.compile(**compile_info)
    bidirect_model.summary()
    return bidirect_model


compile_info = {
    'optimizer': keras.optimizers.Adam(learning_rate=1e-3),
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy']
}

bidirect_model = create_model(compile_info)

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from functools import partial


def show_history(history: keras.callbacks.History):
    """
    Shows training history data stored by the History Keras callback

    :param history: History Keras callback
    """

    history_data = history.history
    print("Displaying the following history keys: ", history_data.keys())

    for key, value in history_data.items():
        if not key.startswith('val'):
            fig, ax = plt.subplots(1, 1)
            ax.set_title(key)
            ax.plot(value)
            if 'val_{}'.format(key) in history_data:
                ax.plot(history_data['val_{}'.format(key)])
            else:
                print("Couldn't find validation values for metric: ", key)

            ax.set_ylabel(key)
            ax.set_xlabel('epoch')
            ax.legend(['train', 'val'], loc='best')

    plt.show()


def train_model(model: keras.Model,
                x_train: np.ndarray,
                y_train: np.ndarray,
                x_val: np.ndarray,
                y_val: np.ndarray,
                training_info: dict):
    """
    Training routine for the Keras model.
    At the end of the training, retrieved History data is shown.

    :param model: Keras built model
    :param x_train: training data in np.ndarray format
    :param y_train: training labels in np.ndarray format
    :param x_val: validation data in np.ndarray format
    :param y_val: validation labels in np.ndarray format
    :param training_info: dictionary storing model fit() argument information

    :return
        model: trained Keras model
    """
    print("Start training! \nParameters: {}".format(training_info))
    history = model.fit(x=x_train, y=y_train,
                        validation_data=(x_val, y_val),
                        **training_info)
    print("Training completed! Showing history...")

    show_history(history)

    return model


def predict_data(model: keras.Model,
                 x: np.ndarray,
                 prediction_info: dict) -> np.ndarray:
    """
    Inference routine of a given input set of examples

    :param model: Keras built and possibly trained model
    :param x: input set of examples in np.ndarray format
    :param prediction_info: dictionary storing model predict() argument information

    :return
        predictions: predicted labels in np.ndarray format
    """

    print('Starting prediction: \n{}'.format(prediction_info))
    print('Predicting on {} samples'.format(x.shape[0]))

    predictions = model.predict(x, **prediction_info)
    return predictions


training_info = {
    'verbose': 1,
    'epochs': 10,
    'batch_size': 64,
    'callbacks': [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=10)]
}
model = train_model(model=bidirect_model, x_train=x_train, y_train=y_train,
                    x_val=x_val, y_val=y_val, training_info=training_info)

# t-SNE
'''
reduced_embedding_tSNE = reduce_tSNE(embedding_matrix)
visualize_embeddings(reduced_embedding_tSNE)
visualize_embeddings(reduced_embedding_tSNE,
                     ['good', 'love', 'beautiful'],
                     word_to_idx)

plt.show()
'''
