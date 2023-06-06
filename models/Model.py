from models.loss.LossClass import LossClass


class Model:
    """
    Base class for machine learning models.

    The Model class provides an interface for training, evaluating, and making predictions
    with machine learning models.

    """

    def fit(self, loss: LossClass, epochs=10, batch_size=32):
        """
        Trains the model with the specified loss function, number of epochs, and batch size.

        :param loss: The LossClass object representing the loss function and optimizer.
        :param epochs: The number of epochs to train the model.
        :param batch_size: The batch size for training.
        :return: None
        """
        raise NotImplementedError("The fit method must be implemented in the derived classes.")

    def evaluate(self):
        """
        Evaluates the model on the test data and returns the evaluation metric(s).

        :return: The evaluation metric(s).
        """
        raise NotImplementedError("The evaluate method must be implemented in the derived classes.")

    def predict(self):
        """
        Makes predictions with the model on new data and returns the predictions.

        :return: The predictions.
        """
        raise NotImplementedError("The predict method must be implemented in the derived classes.")
