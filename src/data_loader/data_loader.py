"""
    data loader
"""
import json
import pandas as pd

__author__ = "Maryam Najafi"
__organization__ = "Religious ChatBot"
__license__ = "Public Domain"
__version__ = "1.0.0"
__email__ = "Maryam_Najafi73@yahoo.com"
__status__ = "Production"
__date__ = "07/27/2021"


def read_csv(path: str, columns: list = None, names: list = None) -> pd.DataFrame:
    """

    :param path:
    :param columns:
    :param names:
    :return:
    """
    dataframe = pd.read_csv(path, usecols=columns) if columns else pd.read_csv(path)  # , sep="\t"
    return dataframe.rename(columns={c: n for c, n in zip(columns, names)}) if names else dataframe


def write_json(path: str, data: dict) -> None:
    with open(path, "w") as outfile:
        json.dump(data, outfile, separators=(",", ":"), indent=4)
