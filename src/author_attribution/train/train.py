#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=import-error

"""
train.py is written for train model
"""

import logging
import itertools
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score
from author_attribution.tools.evaluation_helper import categorical_accuracy, top_n_accuracy

__author__ = "Maryam Najafi"
__organization__ = "AuthorShip Attribution"
__license__ = "Public Domain"
__version__ = "1.0.0"
__email__ = "Maryam_Najafi73@yahoo.com"
__status__ = "Production"
__date__ = "06/09/2021"
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


def train(model, iterator, optimizer, criterion):
    """
    train method is written for train model
    :param model: your creation model
    :param iterator: train iterator
    :param optimizer: your optimizer
    :param criterion: your criterion
    """

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    iter_len = len(iterator)
    n_batch = 0

    # start training model
    for batch in iterator:
        n_batch += 1
        optimizer.zero_grad()
        text, _ = batch.text
        label = batch.label
        predictions = model(text)

        # calculate loss
        loss = criterion(predictions, label)

        # calculate accuracy
        acc = categorical_accuracy(predictions, label)

        # back-propagate loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if (n_batch % (iter_len // 5)) == 0:
            logging.info(f"\t train on: {(n_batch / iter_len) * 100:.2f}% of samples")
            logging.info(f"\t accuracy : {(epoch_acc / n_batch) * 100 :.2f}%")
            logging.info(f"\t loss : {(epoch_loss / n_batch):.4f}".format())
            logging.info("________________________________________________\n")


def evaluate(model, iterator, criterion):
    """
    evaluate method is written for for evaluate model
    :param model: your creation model
    :param iterator: your iterator
    :param criterion: your criterion
    :return:
        loss: loss of all  data
        acc: accuracy of all  data
        precision: precision for each class of data
        recall: recall for each class of data
        f-score: F1-score for each class of data
        total_fscore: F1-score of all  data
    """
    # define evaluate_parameters_dict to save output result
    evaluate_parameters_dict = {"loss": 0, "acc": 0, "top_3_acc": 0,
                                "top_5_acc": 0, "precision": 0, "recall": 0,
                                "f-score": 0, "total_fscore": 0}
    total_predict = []
    total_label = []
    # put model in evaluate model
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            # predict input data
            text, _ = batch.text
            predictions = model(text)

            total_predict.append(predictions.cpu().numpy())
            total_label.append(batch.label.cpu().numpy())

            # calculate loss
            loss = criterion(predictions, batch.label)

            # calculate accuracy
            acc = categorical_accuracy(predictions, batch.label)
            top_3_acc = top_n_accuracy(predictions.cpu().numpy(), batch.label.cpu().numpy(), num=3)
            top_5_acc = top_n_accuracy(predictions.cpu().numpy(), batch.label.cpu().numpy(), num=5)

            # # save model result
            evaluate_parameters_dict["loss"] += loss.item()
            evaluate_parameters_dict["acc"] += acc.item()
            evaluate_parameters_dict["top_3_acc"] += top_3_acc
            evaluate_parameters_dict["top_5_acc"] += top_5_acc
    total_predict = list(itertools.chain.from_iterable(total_predict))
    total_predict = list(np.argmax(total_predict, axis=1))
    total_label = list(itertools.chain.from_iterable(total_label))

    # calculate precision, recall and f_score
    evaluate_parameters_dict["precision"], evaluate_parameters_dict["recall"], \
    evaluate_parameters_dict["f-score"], _ = \
        precision_recall_fscore_support(y_true=total_label,
                                        y_pred=total_predict)

    # calculate total f-score of all data
    evaluate_parameters_dict["total_fscore"] = f1_score(y_true=total_label,
                                                        y_pred=total_predict,
                                                        average="weighted")

    evaluate_parameters_dict["loss"] = \
        evaluate_parameters_dict["loss"] / len(iterator)

    evaluate_parameters_dict["acc"] = \
        evaluate_parameters_dict["acc"] / len(iterator)
    evaluate_parameters_dict["top_3_acc"] = \
        evaluate_parameters_dict["top_3_acc"] / len(iterator)

    evaluate_parameters_dict["top_5_acc"] = \
        evaluate_parameters_dict["top_5_acc"] / len(iterator)

    return evaluate_parameters_dict
