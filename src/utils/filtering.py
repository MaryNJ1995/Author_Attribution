from typing import List

from validation import is_in_range

__author__ = "Maryam Najafi🥰"
__organization__ = "Author Attribution"
__license__ = "Public Domain"
__version__ = "1.0.0"
__email__ = "Maryaminj1995@gmail.com"
__status__ = "Production"
__date__ = "07/27/2021"


def filter_by_length(data: List[list], lower_bound: int, upper_bound: int) -> List[list]:
    """

    :param data: [[word1, word2, word3], ..., [word1, word2]]
    :param lower_bound: 3
    :param upper_bound: 5
    :return: [[word1, word2, word3, ... ], ..., None]
    """
    for index in range(len(data)):
        if not is_in_range(len(data[index]), lower_bound, upper_bound):
            data[index] = None
    return data


def filter_by_value(first_data: List[list], second_data: List[list], third_list: list,
                    value: str = None) -> [List[list], List[list], List[list]]:
    """

    :param first_data:
    :param second_data:
    :param third_list:
    :param value:
    :return:
    """
    filtered_data = [(i, j, k) for i, j, k in zip(first_data, second_data, third_list)
                     if (i is not value) and (j is not value)]
    return [item[0] for item in filtered_data], \
           [item[1] for item in filtered_data], \
           [item[2] for item in filtered_data]


def filter_by_count(item2count: dict, lower_range: int, upper_range: int) -> dict:
    """

    :param item2count:
    :param lower_range:
    :param upper_range:
    :return:
    """
    filtered_item2count = dict()
    for key, value in item2count.items():
        if is_in_range(value, lower_range, upper_range):
            filtered_item2count[key] = value
    return filtered_item2count
