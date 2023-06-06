import pandas as pd


class Imputer:
    """
    Base class for data imputation.
    """

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to handle missing values in data.

        :param data: The input data as a pandas DataFrame.
        :return: The data with missing values handled.
        """
        raise NotImplementedError("The handle_missing_values method must be implemented in the derived classes.")
