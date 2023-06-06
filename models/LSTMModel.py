import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential

import sys
sys.path.append('..')
from data.preparation.DataHandler import DataHandler
from models.Model import Model
from models.loss.LossClass import LossClass

class LSTMModel(Model):
    """
    A class representing an LSTM model for sequence prediction.

    The LSTMModel class extends the Model class and provides an implementation for training,
    evaluating, and predicting with an LSTM model using Keras.

    """

    def __init__(self, data_handler: DataHandler):
        """
        Initializes an LSTMModel object with the specified DataHandler.

        :param data_handler: The DataHandler object to handle data preparation.
        """
        self.data_handler = data_handler
        self.model = Sequential()

    def _build_model(self, input_shape: tuple):
        """
        Builds the LSTM model architecture.

        :param input_shape: The input shape of the model.
        :return: None
        """
        self.model.add(LSTM(64, input_shape=input_shape))
        self.model.add(Dense(1, activation='sigmoid'))

    def fit(self, loss: LossClass, epochs: int = 10, batch_size: int = 32):
        """
        Trains the LSTM model with the specified loss function, number of epochs, and batch size.

        :param loss: The LossClass object representing the loss function and optimizer.
        :param epochs: The number of epochs to train the model.
        :param batch_size: The batch size for training.
        :return: None
        """
        X_train, y_train = self.data_handler.get_train_data()
        input_shape = (X_train.shape[1], 1)

        self._build_model(input_shape)

        loss.compile_model(self.model)

        self.model.fit(X_train.to_numpy().reshape(X_train.shape[0], *input_shape), y_train, epochs=epochs, batch_size=batch_size)

    def evaluate(self) -> float:
        """
        Evaluates the LSTM model on the test data and returns the loss value.

        :return: The loss value.
        """
        X_test, y_test = self.data_handler.get_test_data()
        input_shape = (X_test.shape[1], 1)
        loss = self.model.evaluate(X_test.to_numpy().reshape(X_test.shape[0], *input_shape), y_test)
        return loss

    def predict(self) -> np.ndarray:
        """
        Makes predictions with the LSTM model on the test data and returns the predictions.

        :return: The predictions.
        """
        X_test = self.data_handler.get_test_data()[0]
        input_shape = (X_test.shape[1], 1)
        predictions = self.model.predict(X_test.to_numpy().reshape(X_test.shape[0], *input_shape))
        return predictions
