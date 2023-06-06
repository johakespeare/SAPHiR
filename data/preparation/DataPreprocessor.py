import numpy as np
import pandas as pd


from data.preparation.imputers import Imputer
from data.preparation.scalers import Scaler
from data.preparation.encoders import Encoder


class DataPreprocessor:
    """
    A class for preprocessing data by handling missing values, scaling numerical features, and encoding
    categorical variables.

    The DataPreprocessor class provides methods to handle common preprocessing tasks on input data,
    such as handling missing values, scaling numerical features, and encoding categorical variables.
    It utilizes popular libraries such as scikit-learn to perform these preprocessing operations.

    """
    def __init__(self):
        self.data = None
        self.numerical_features = None
        self.categorical_features = None

    def load_data(self, data: pd.DataFrame) -> None:
        """
        Loads the data into the DataPreprocessor instance.

        :param data: The input data as a pandas DataFrame.
        :return: None
        """
        self._validate_input_data(data)
        self.data = data.copy()
        self._detect_feature_types()

    def handle_missing_values(self, imputer: Imputer) -> None:
        """
        Handles missing values in the data using the specified imputer.

        :param imputer: An object implementing the Imputer interface.
        :return: None
        """
        self._ensure_data_loaded()
        self.data = imputer.handle_missing_values(self.data)

    def scale_features(self, scaler: Scaler) -> None:
        """
        Scales the numerical features in the data using the specified scaler.

        :param scaler: An object implementing the Scaler interface.
        :return: None
        """
        self._ensure_data_loaded()
        self.data = scaler.scale_features(self.data)

    def encode_categorical_variables(self, encoder: Encoder) -> None:
        """
        Encodes categorical variables in the data using the specified encoder.

        :param encoder: An object implementing the Encoder interface.
        :return: None
        """
        self._ensure_data_loaded()
        self.data = encoder.encode_features(self.data)

    def reset(self) -> None:
        """
        Resets the DataPreprocessor instance by clearing the data and feature information.

        :return: None
        """
        self.data = None
        self.numerical_features = None
        self.categorical_features = None

    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """
        Validates the input data.

        :param data: The input data to be validated.
        :return: None
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")

    def _ensure_data_loaded(self) -> None:
        """
        Ensures that data has been loaded.

        :return: None
        """
        if self.data is None:
            raise ValueError("Data has not been loaded. Call load_data() first.")

    def _detect_feature_types(self) -> None:
        """
        Detects the numerical and categorical feature columns in the data.

        :return: None
        """
        self.numerical_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
