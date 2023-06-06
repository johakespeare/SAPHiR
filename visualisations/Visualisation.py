import matplotlib.pyplot as plt


class Visualisation:
    """
    A class for visualizing model predictions compared to actual values.
    """

    def __init__(self, title: str = "Model Predictions vs Actual Values",
                 labels: tuple[str, str] = ("X_axis", "Y_axis")):
        """
        Initializes the Visualisation object.

        :param title: The title of the plot.
        :param labels: A tuple of labels for the x-axis and y-axis.
        """
        self.title = title
        self.labels = labels

    def plot(self, y_test, predictions):
        """
        Plots the actual values and predicted values.

        :param y_test: The actual values.
        :param predictions: The predicted values.
        """
        figure, axis = plt.subplots()
        axis.plot(y_test.values, label='Actual Values')
        axis.plot(predictions, label='Predictions')
        axis = self._set_labels(axis)
        axis = self._set_title(axis)
        axis.legend()
        plt.show()

    def _set_labels(self, axis: plt.Axes) -> plt.Axes:
        """
        Sets the labels for the x-axis and y-axis.

        :param axis: The axis to set the labels on.
        :return: The modified axis.
        """
        axis.set_xlabel(self.labels[0])
        axis.set_ylabel(self.labels[1])
        return axis

    def _set_title(self, axis: plt.Axes) -> plt.Axes:
        """
        Sets the title of the plot.

        :param axis: The axis to set the title on.
        :return: The modified axis.
        """
        axis.set_title(self.title)
        return axis



