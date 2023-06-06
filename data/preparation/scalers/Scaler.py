import pandas as pd


class Scaler:
    """
    Base class for data scaling.
    """

    def scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to scale numerical features in data.

        :param data: The input data as a pandas DataFrame.
        :return: The data with scaled numerical features.
        """
        raise NotImplementedError("The scale_features method must be implemented in the derived classes.")
