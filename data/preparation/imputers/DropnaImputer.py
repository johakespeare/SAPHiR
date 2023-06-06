import pandas as pd
from data.preparation.imputers import Imputer


class DropnaImputer():
    """
    A class that implements missing value imputation by dropping rows with missing values.
    """

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values by dropping rows with missing values.

        :param data: The input data as a pandas DataFrame.
        :return: The data with missing values handled.
        """
        return data.dropna()
