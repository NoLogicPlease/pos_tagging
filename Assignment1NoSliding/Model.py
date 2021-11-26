from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score, accuracy_score
from functools import partial
import matplotlib.pyplot as plt
import numpy as np


class Model(object):
    def __init__(self, model_type, compile_info, value_to_key, embedding_dim, max_seq_len, num_labels,
                 embedding_matrix):

        self.compile_info = compile_info
        self.value_to_key = value_to_key
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.num_labels = num_labels
        self.embedding_matrix = embedding_matrix

        if 'baseline' == model_type:
            self.model = self.create_LSTM()
        elif 'gru' == model_type:
            self.model = self.create_GRU()
        elif 'two_lstm' == model_type:
            self.model = self.create_two_LSTM()
        else:
            self.model = self.create_two_Dense()

    def create_LSTM(self) -> keras.Model:

        bidirect_model = keras.models.Sequential()
        bidirect_model.add(layers.Embedding(input_dim=len(self.value_to_key.keys()),
                                            output_dim=self.embedding_dim,
                                            input_length=self.max_seq_len,
                                            mask_zero=True,
                                            weights=[self.embedding_matrix],
                                            trainable=False
                                            ))
        bidirect_model.add(layers.Bidirectional(layers.LSTM(250, return_sequences=True)))
        bidirect_model.add(layers.TimeDistributed(layers.Dense(self.num_labels, activation="softmax")))

        bidirect_model.compile(**self.compile_info)
        bidirect_model.summary()
        return bidirect_model

    def create_GRU(self) -> keras.Model:
        gru = keras.models.Sequential()
        gru.add(layers.Embedding(input_dim=len(self.value_to_key.keys()),
                                 output_dim=self.embedding_dim,
                                 input_length=self.max_seq_len,
                                 mask_zero=True,
                                 weights=[self.embedding_matrix],
                                 trainable=False
                                 ))

        gru.add(layers.GRU(64, return_sequences=True))
        gru.add(layers.TimeDistributed(layers.Dense(self.num_labels, activation="softmax")))
        gru.compile(**self.compile_info)
        gru.summary()
        return gru

    def create_two_LSTM(self) -> keras.Model:
        lstm = keras.models.Sequential()
        lstm.add(layers.Embedding(input_dim=len(self.value_to_key.keys()),
                                  output_dim=self.embedding_dim,
                                  input_length=self.max_seq_len,
                                  mask_zero=True,
                                  weights=[self.embedding_matrix],
                                  trainable=False
                                  ))

        lstm.add(layers.Bidirectional(layers.LSTM(250, return_sequences=True)))
        lstm.add(layers.LSTM(64, return_sequences=True))
        lstm.add(layers.TimeDistributed(layers.Dense(self.num_labels, activation="softmax")))
        lstm.compile(**self.compile_info)
        lstm.summary()
        return lstm

    def create_two_Dense(self) -> keras.Model:
        lstm = keras.models.Sequential()
        lstm.add(layers.Embedding(input_dim=len(self.value_to_key.keys()),
                                  output_dim=self.embedding_dim,
                                  input_length=self.max_seq_len,
                                  mask_zero=True,
                                  weights=[self.embedding_matrix],
                                  trainable=False
                                  ))

        lstm.add(layers.Bidirectional(layers.LSTM(250, return_sequences=True)))
        lstm.add(layers.TimeDistributed(layers.Dense(128, activation="relu")))
        lstm.add(layers.TimeDistributed(layers.Dense(self.num_labels, activation="softmax")))
        lstm.compile(**self.compile_info)
        lstm.summary()
        return lstm

    def show_history(self, history: keras.callbacks.History):

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

    def train_model(self,
                    x_train: np.ndarray,
                    y_train: np.ndarray,
                    x_val: np.ndarray,
                    y_val: np.ndarray,
                    training_info: dict):

        print("Start training! \nParameters: {}".format(training_info))
        history = self.model.fit(x=x_train, y=y_train,
                                 validation_data=(x_val, y_val),
                                 **training_info)
        print("Training completed! Showing history...")

        self.show_history(history)

    def predict_data(self,
                     x: np.ndarray,
                     prediction_info: dict) -> np.ndarray:

        print('Starting prediction: \n{}'.format(prediction_info))
        print('Predicting on {} samples'.format(x.shape[0]))

        predictions = self.model.predict(x, **prediction_info)
        return predictions
