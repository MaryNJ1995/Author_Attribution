#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable-msg=import-error
# pylint: disable-msg=no-member
# pylint: disable-msg=not-callable

import time
import logging
import torch
from torch import optim
from torch import nn
from importlib import import_module
from author_attribution.utils.data_util import DataSet, init_weights, read_model_config, \
    initialize_weights_xavier_uniform_
from author_attribution.train.train import train, evaluate
from author_attribution.tools.log_helper import count_parameters, process_time, \
    model_result_log, model_result_save
from author_attribution.config.config import get_config
import matplotlib.pyplot as plt

__author__ = "Maryam Najafi"
__organization__ = "AuthorShip Attribution"
__license__ = "Public Domain"
__version__ = "1.0.0"
__email__ = "Maryam_Najafi73@yahoo.com"
__status__ = "Production"
__date__ = "11/14/2021"
args = get_config()

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


class RunModel:
    """
    In this class we start training and testing model
    """

    def __init__(self):
        # open log file
        self.log_file = open(args.log_path, "w")

    @staticmethod
    def load_data_set():
        """
        load_data_set method is written for load input data and iterators
        :return:
            data_set: data_set
        """
        # load data from input file

        data_set = DataSet(train_data_path=args.train_data,
                           test_data_path=args.test_data,
                           valid_data_path=args.valid_data,
                           embedding_path=args.embedding_path)

        data_set.load_data(text_field_path=args.text_field_path, label_field_path=args.label_field_path,
                           device=args.device, batch_size=args.batch_size,
                           min_freq=args.min_freq)
        print(data_set.embedding_dict["vocab_embedding_vectors"])
        return data_set

    @staticmethod
    def draw_curves(**kwargs):
        """
        draw_curves method is written for drawing loss and accuracy curve
        """
        # plot loss curves
        plt.figure()
        plt.plot(kwargs["train_loss"], "r", label="train_loss")
        plt.plot(kwargs["validation_loss"], "b", label="validation_loss")
        plt.plot(kwargs["test_loss"], "g", label="test_loss")
        plt.legend()
        plt.xlabel("number of epochs")
        plt.ylabel("loss value")
        plt.show()
        plt.savefig(args.loss_curve_path)

        # clear figure command
        plt.clf()

        # plot accuracy curves
        plt.figure()
        plt.plot(kwargs["train_acc"], "r", label="train_acc")
        plt.plot(kwargs["validation_acc"], "b", label="validation_acc")
        plt.plot(kwargs["test_acc"], "g", label="test_acc")
        plt.legend()
        plt.xlabel("number of epochs")
        plt.ylabel("accuracy value")
        plt.show()
        plt.savefig(args.accuracy_curve_path)

    def run(self):
        """
        run method is written for running model
        """
        logging.info(args)
        data_set = self.load_data_set()
        model, criterion, optimizer = init_model(data_set)

        best_validation_loss = float("inf")
        best_test_f_score = 0.0
        losses_dict = dict()
        acc_dict = dict()
        losses_dict["train_loss"] = []
        losses_dict["validation_loss"] = []
        losses_dict["test_loss"] = []
        acc_dict["train_acc"] = []
        acc_dict["validation_acc"] = []
        acc_dict["test_acc"] = []
        # start training model
        for epoch in range(args.n_epochs):
            start_time = time.time()

            # train model on train data
            train(model=model, iterator=data_set.iterator_dict["train_iterator"],
                  optimizer=optimizer, criterion=criterion)

            # compute model result on train data
            train_log_dict = evaluate(model=model, iterator=data_set.iterator_dict["train_iterator"],
                                      criterion=criterion)

            losses_dict["train_loss"].append(train_log_dict["loss"])
            acc_dict["train_acc"].append(train_log_dict["acc"])

            # compute model result on validation data
            valid_log_dict = evaluate(model=model, iterator=data_set.iterator_dict["valid_iterator"],
                                      criterion=criterion)

            losses_dict["validation_loss"].append(valid_log_dict["loss"])
            acc_dict["validation_acc"].append(valid_log_dict["acc"])

            # compute model result on test data
            test_log_dict = evaluate(model=model, iterator=data_set.iterator_dict["test_iterator"],
                                     criterion=criterion)
            losses_dict["test_loss"].append(test_log_dict["loss"])
            acc_dict["test_acc"].append(test_log_dict["acc"])

            end_time = time.time()

            # calculate epoch time
            epoch_mins, epoch_secs = process_time(start_time, end_time)

            # save model when loss in validation data is decrease
            if valid_log_dict["loss"] < best_validation_loss:
                best_validation_loss = valid_log_dict["loss"]
                torch.save(model.state_dict(),
                           args.model_path + f"model_epoch{epoch + 1}_loss_"
                                             f"{valid_log_dict['loss']}.pt")

            # save model when fscore in test data is increase
            if test_log_dict["total_fscore"] > best_test_f_score:
                best_test_f_score = test_log_dict["total_fscore"]
                torch.save(model.state_dict(),
                           args.model_path + f"model_epoch{epoch + 1}"
                                             f"_fscore_{test_log_dict['total_fscore']}.pt")

            # show model result
            logging.info(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            model_result_log(train_log_dict, valid_log_dict, test_log_dict)

            # save model result in log file
            self.log_file.write(f"Epoch: {epoch + 1:02} | Epoch Time: "
                                f"{epoch_mins}m {epoch_secs}s\n")
            model_result_save(self.log_file, train_log_dict, valid_log_dict, test_log_dict)

        # save final model
        torch.save(model.state_dict(), args.model_path + "final_model.pt")

        # plot curve
        self.draw_curves(train_acc=acc_dict["train_acc"], validation_acc=acc_dict["validation_acc"],
                         test_acc=acc_dict["test_acc"], train_loss=losses_dict["train_loss"],
                         validation_loss=losses_dict["validation_loss"],
                         test_loss=losses_dict["test_loss"])


def init_model(data_set):
    """
    init_model method is written for loading model and
    define loss function and optimizer
    :param data_set: data_set class
    :return:
        model: AI model
        criterion: loss function
        optimizer: optimizer function
    """
    # create model
    config = {
        "vocab_size": data_set.num_vocab_dict["num_token"],
        "pad_idx": data_set.pad_idx_dict["token_pad_idx"],
        "output_size": data_set.num_vocab_dict["num_label"],
        "pos_embedding_dim": args.pos_embeddings,
        "embedding_dim": args.embedding_dim,
        "char_embedding_dim": data_set.num_vocab_dict["num_token"],
        "depth": 9,
        "shortcut": False,
        "lstm_layers": args.lstm_layers,
        "n_fc_neurons": args.linear_units,
        "lstm_units": args.lstm_units,
        "start_dropout": args.start_dropout,
        "middle_dropout": args.middle_dropout,
        "bidirectional": args.bidirectional,
        "n_filters": args.n_filters,
        "filter_sizes": args.filter_size,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "out_units": 2000,

        "hidden_dim": args.hidden_dim,
        "encoder_layers": args.encoder_layers,
        "num_heads": args.num_heads,
        "encoder_pf_dim": args.encoder_pf_dim,
        "max_len": args.max_len,
        "device": args.device,
        "model_name": "DCBiLstmAttentions",
    }
    models_dict = read_model_config(args.model_name_config_path)

    logging.info("model name is:{0}".format(config.get("model_name")))
    ai_model = getattr(import_module(models_dict[config.get("model_name")]['path']),
                       models_dict[config.get("model_name")]['object'])
    model = ai_model(**config)

    # initializing model parameters
    model.apply(init_weights)
    logging.info("create model.")

    # copy word embedding vectors to embedding layer
    model.embeddings.weight.data.copy_(data_set.embedding_dict["vocab_embedding_vectors"])
    model.embeddings.weight.data[data_set.pad_idx_dict["token_pad_idx"]] = \
        torch.zeros(args.embedding_dim)
    model.embeddings.weight.requires_grad = True
    if args.use_pos:
        nn.init.uniform_(model.pos_embeddings.weight, -1.0, 1.0)
        model.pos_embeddings.weight.data[data_set.pad_idx_dict["token_pad_idx"]] = torch.zeros(
            args.pos_embeddings)
        model.pos_embeddings.weight.requires_grad = True

    logging.info(f"The model has {count_parameters(model):,} trainable parameters")

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    # define loss function
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(data_set.class_weight))

    # load model into GPU
    model = model.to(args.device)
    criterion = criterion.to(args.device)
    return model, criterion, optimizer


if __name__ == "__main__":
    MYCLASS = RunModel()
    MYCLASS.run()
