from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score, accuracy_score
from functools import partial
import matplotlib.pyplot as plt
import numpy as np


class Model(object):
    def __init__(self, model_type='gru'):
        self.model_type = model_type

    def create_LSTM(self, compile_info: dict, value_to_key, embedding_dim,
                    max_seq_len, num_labels, embedding_matrix) -> keras.Model:
        bidirect_model = keras.models.Sequential()
        bidirect_model.add(layers.Embedding(input_dim=len(value_to_key.keys()),
                                            output_dim=embedding_dim,
                                            input_length=max_seq_len,
                                            mask_zero=True,
                                            weights=[embedding_matrix],
                                            trainable=False
                                            ))
        bidirect_model.add(layers.Bidirectional(layers.LSTM(250, return_sequences=True)))
        bidirect_model.add(layers.TimeDistributed(layers.Dense(num_labels, activation="softmax")))

        bidirect_model.compile(**compile_info)
        bidirect_model.summary()
        return bidirect_model

    def create_GRU(self, compile_info: dict, value_to_key, embedding_dim,
                   max_seq_len, num_labels, embedding_matrix) -> keras.Model:
        gru = keras.models.Sequential()
        gru.add(layers.Embedding(input_dim=len(value_to_key.keys()),
                                 output_dim=embedding_dim,
                                 input_length=max_seq_len,
                                 mask_zero=True,
                                 weights=[embedding_matrix],
                                 trainable=False
                                 ))

        gru.add(layers.GRU(64, return_sequences=True))
        gru.add(layers.TimeDistributed(layers.Dense(num_labels, activation="softmax")))
        gru.compile(**compile_info)
        gru.summary()
        return gru

    def create_two_LSTM(self, compile_info: dict, value_to_key, embedding_dim,
                        max_seq_len, num_labels, embedding_matrix) -> keras.Model:
        lstm = keras.models.Sequential()
        lstm.add(layers.Embedding(input_dim=len(value_to_key.keys()),
                                  output_dim=embedding_dim,
                                  input_length=max_seq_len,
                                  mask_zero=True,
                                  weights=[embedding_matrix],
                                  trainable=False
                                  ))

        lstm.add(layers.Bidirectional(layers.LSTM(250, return_sequences=False)))
        lstm.add(layers.LSTM(64, return_sequences=True))
        lstm.add(layers.TimeDistributed(layers.Dense(num_labels, activation="softmax")))
        lstm.compile(**compile_info)
        lstm.summary()
        return lstm

    def create_two_Dense(self, compile_info: dict, value_to_key, embedding_dim,
                         max_seq_len, num_labels, embedding_matrix) -> keras.Model:
        lstm = keras.models.Sequential()
        lstm.add(layers.Embedding(input_dim=len(value_to_key.keys()),
                                  output_dim=embedding_dim,
                                  input_length=max_seq_len,
                                  mask_zero=True,
                                  weights=[embedding_matrix],
                                  trainable=False
                                  ))

        lstm.add(layers.Bidirectional(layers.LSTM(250, return_sequences=True)))
        lstm.add(layers.TimeDistributed(layers.Dense(128, activation="relu")))
        lstm.add(layers.TimeDistributed(layers.Dense(num_labels, activation="softmax")))
        lstm.compile(**compile_info)
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

    def train_model(self, model: keras.Model,
                    x_train: np.ndarray,
                    y_train: np.ndarray,
                    x_val: np.ndarray,
                    y_val: np.ndarray,
                    training_info: dict):

        print("Start training! \nParameters: {}".format(training_info))
        history = model.fit(x=x_train, y=y_train,
                            validation_data=(x_val, y_val),
                            **training_info)
        print("Training completed! Showing history...")

        self.show_history(history)

        return model

    def predict_data(self, model: keras.Model,
                     x: np.ndarray,
                     prediction_info: dict) -> np.ndarray:

        print('Starting prediction: \n{}'.format(prediction_info))
        print('Predicting on {} samples'.format(x.shape[0]))

        predictions = model.predict(x, **prediction_info)
        return predictions
