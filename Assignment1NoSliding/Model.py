from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score, accuracy_score
from functools import partial
import matplotlib.pyplot as plt
import numpy as np



def create_model(compile_info: dict, value_to_key, embedding_dim,
                 max_seq_len, num_labels, embedding_matrix) -> keras.Model:
    bidirect_model = keras.models.Sequential()
    bidirect_model.add(layers.Embedding(input_dim=len(value_to_key.keys()),
                                        output_dim=embedding_dim,
                                        input_length=max_seq_len,
                                        mask_zero=True,
                                        weights=embedding_matrix,
                                        trainable=False
                                        ))
    bidirect_model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
    bidirect_model.add(layers.TimeDistributed(layers.Dense(num_labels, activation="softmax")))

    bidirect_model.compile(**compile_info)
    bidirect_model.summary()
    return bidirect_model


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


