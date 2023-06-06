import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data.preparation.scalers import Scaler


class MinMaxScalerWrapper:
    """
    A wrapper class for MinMaxScaler from scikit-learn.
    """

    def scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Scales the numerical features in the data using MinMaxScaler.

        :param data: The input data as a pandas DataFrame.
        :return: The data with scaled numerical features.
        """
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns)
        return scaled_data
