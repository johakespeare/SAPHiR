import pandas as pd

from data.preparation.parsers.DataParser import DataParser

class CSVParser(DataParser):
    """
    A class for parsing CSV files.

    The CSVParser class extends the DataParser class and provides a specific implementation
    for parsing CSV files.

    """

    def parse(self, file_path, delimiter=";"):
        """
              Parses a CSV file and returns the parsed data as a pandas DataFrame.

              :param file_path: The path to the CSV file.
              :param delimiter: The delimiter used in the CSV file. Default is ";".
              :return: The parsed data as a pandas DataFrame.
              """

        file_path = "/Users/johannafericean/PycharmProjects/experimentation/data/database/"+file_path
        data = pd.read_csv(file_path, delimiter=delimiter)
        return data
