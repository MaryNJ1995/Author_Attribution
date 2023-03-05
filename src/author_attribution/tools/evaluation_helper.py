#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable-msg=no-member

"""
evaluation_helper.py is a evaluation file for writing evaluation methods
"""

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def binary_accuracy(preds, target):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == target).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def categorical_accuracy(preds, target):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    correct = max_preds.squeeze(1).eq(target)
    return correct.sum().to(DEVICE) / torch.FloatTensor([target.shape[0]]).to(DEVICE)


def top_n_accuracy(preds, target, num=5):
    """
    get top n accuracy in model result
    """
    score = 0
    for i, pred in enumerate(preds):
        arr = np.array(pred)
        arr = arr.argsort()[-num:][::-1]
        if target[i] in arr:
            score += 1
    return score / len(preds)
