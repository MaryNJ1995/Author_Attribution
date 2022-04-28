import json
from itertools import chain
from collections import Counter
from typing import List

__author__ = "Maryam Najafi🥰"
__organization__ = "Religious ChatBot"
__license__ = "Public Domain"
__version__ = "1.1.0"
__email__ = "Maryam_Najafi73@yahoo.com"
__status__ = "Production"
__date__ = "07/27/2021"


def item_counter(items: List[list]) -> dict:
    """

    :param items: ex: [["first", "sent"], ["second", "sent"]]
    :return: {"first": 1, "sent":2, "second": 1}
    """
    counter = Counter()
    for item in items:
        counter.update(item)

    return dict(counter)


def convert_words_to_chars(tokens: list) -> list:
    """

    :param tokens:
    :return:
    """
    return list(set(chain.from_iterable(tokens)))


def find_max_length_in_list(data: List[list]) -> int:
    """

    :param data:
    :return:
    """
    return max(len(sample) for sample in data)


def read_model_config(config_path):
    with open(config_path) as json_file:
        data = json.load(json_file)
    return data
