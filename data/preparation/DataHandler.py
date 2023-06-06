import numpy as np
import pandas as pd

from data.preparation.splitters.DataSplitter import DataSplitter


class DataHandler:
    """
        Handles loading and splitting of data for machine learning tasks.
    """

    def __init__(self):
        """
        Initializes a new instance of the DataHandler class.
        """
        self.data = None
        self.target = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, data: pd.DataFrame, target: str) -> None:
        """
        Loads the data and target variable into the DataHandler instance.

        :param data: The input data as a pandas DataFrame.
        :param target: The name of the target variable column.
        :return: None
        """
        self._validate_input_data(data, target)
        self.data = data
        self.target = target

    def split_data(self, data_splitter: DataSplitter) -> None:
        """
        Splits the loaded data into training and test sets using the provided data_splitter.

        :param data_splitter: A DataSplitter Object with a  method that splits the data into X_train, X_test,
        y_train, y_test.
        :return: None
        """
        self._ensure_data_loaded()
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = data_splitter.split_data(self.X, self.y)
        except ValueError as e:
            raise ValueError(f"Error during data splitting: {str(e)}")

    def get_train_data(self) -> tuple:
        """
        Returns the training data.

        :return: A tuple containing the X_train and y_train.
        """
        self._ensure_data_split()
        return self.X_train.copy(), self.y_train.copy()

    def get_test_data(self) -> tuple:
        """
        Returns the test data.

        :return: A tuple containing the X_test and y_test.
        """
        self._ensure_data_split()
        return self.X_test.copy(), self.y_test.copy()

    def reset(self) -> None:
        """
        Resets the DataHandler instance by clearing all data and split information.

        :return: None
        """
        self.data = None
        self.target = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    @property
    def X(self) -> pd.DataFrame:
        """
        Returns a copy of the input features (X) of the loaded data.

        :return: The input features (X) as a pandas DataFrame.
        """
        self._ensure_data_loaded()
        return self.data.drop(self.target, axis=1).copy()

    @property
    def y(self) -> pd.Series:
        """
        Returns a copy of the target variable (y) of the loaded data.

        :return: The target variable (y) as a pandas Series.
        """
        self._ensure_data_loaded()
        return self.data[self.target].copy()

    def _validate_input_data(self, data: pd.DataFrame, target: str) -> None:
        """
        Validates the input data and target variable.

        :param data: The input data to be validated.
        :param target: The name of the target variable column.
        :return: None
        """
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise ValueError("data must be a pandas DataFrame or a NumPy array")

        if target not in data.columns:
            raise ValueError("The target column does not exist in the data")

    def _ensure_data_loaded(self) -> None:
        """
        Ensures that data has been loaded.

        :return: None
        """
        if self.data is None:
            raise ValueError("Data has not been loaded. Call load_data() first.")

    def _ensure_data_split(self) -> None:
        """
        Ensures that data has been split into training and test sets.

        :return: None
        """
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            raise ValueError("The data has not been split into training and test sets.")
