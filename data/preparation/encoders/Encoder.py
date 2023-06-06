import pandas as pd


class Encoder:
    """
    Base class for data encoding.
    """

    def encode_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to encode categorical features in data.

        :param data: The input data as a pandas DataFrame.
        :return: The data with encoded categorical features.
        """
        raise NotImplementedError("The encode_features method must be implemented in the derived classes.")
