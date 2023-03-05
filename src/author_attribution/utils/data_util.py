#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=import-error
# pylint: disable=too-many-instance-attributes

"""
data_utils.py is writen for creating iterator and save field
"""
import json
import logging
import hazm
import torch
import numpy as np
import pandas as pd
from torchtext.legacy import data
from torchtext.vocab import Vectors
from sklearn.utils import class_weight
from author_attribution.config.config import get_config

__author__ = "Maryam Najafi"
__organization__ = "AuthorShip Attribution"
__license__ = "Public Domain"
__version__ = "1.0.0"
__email__ = "Maryam_Najafi73@yahoo.com"
__status__ = "Production"
__date__ = "06/09/2021"
ARGS = get_config()
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


class DataSet:
    """
    DataSet Class use for preparing data
    and iterator for training model.
    """

    def __init__(self, **kwargs):
        self.files_address = {
            "train_data_path": kwargs["train_data_path"],
            "test_data_path": kwargs["test_data_path"],
            "valid_data_path": kwargs["valid_data_path"],
            "embedding_path": kwargs["embedding_path"],
        }

        self.class_weight = None
        self.text_field = None

        self.iterator_dict = dict()
        self.embedding_dict = dict()
        self.unk_idx_dict = dict()
        self.pad_idx_dict = dict()
        self.num_vocab_dict = dict()
        self.embedding_dict = {'vocab_embedding_vector': None}

    @staticmethod
    def read_csv_file(input_path):
        """
        read_csv_file method is written for reading input csv file
        :param input_path: csv file path
        :return:
            input_df: dataFrame of input data
        """
        input_df = pd.read_csv(input_path)
        input_df = input_df.astype({"text": "str"})
        input_df = input_df.astype({"label": "str"})

        return input_df

    @staticmethod
    def char_tokenizer(sent):
        """
        a method for tokenizing in character level
        """
        return list(sent[::1])

    def create_fields(self):
        """
        This method is writen for creating torchtext fields
        :return:
            dictionary_fields: dictionary of data fields
            data_fields: list of data fields
        """
        if ARGS.use_char:
            text_field = data.Field(tokenize=self.char_tokenizer, batch_first=True)
        else:
            text_field = data.Field(sequential=True, tokenize=hazm.word_tokenize, batch_first=True,
                                    include_lengths=True, fix_length=ARGS.max_len)
        label_field = data.LabelField()
        dictionary_fields = {
            "text_field": text_field,
            "label_field": label_field
        }

        # create list of data fields
        data_fields = [("text", text_field), ("label", label_field)]

        return dictionary_fields, data_fields

    def load_data(self, text_field_path, label_field_path, device, batch_size, min_freq):
        """
        load_data method is written for creating iterator for train and test data
        :param text_field_path: path for text_field
        :param label_field_path: path for label_field
        :param min_freq: min_freq for data occuration
        :param device: gpu or cpu
        :param batch_size: number of sample in batch
        """
        # create fields
        logging.info("Start creating fields.")
        dictionary_fields, data_fields = self.create_fields()

        # Load data from pd.DataFrame into torchtext.data.Dataset
        logging.info("Start creating train example.")
        train_examples = [data.Example.fromlist(row, data_fields) for row in
                          self.read_csv_file(self.files_address["train_data_path"]).values.tolist()]

        train_data = data.Dataset(train_examples, data_fields)

        test_examples = [data.Example.fromlist(row, data_fields) for row in
                         self.read_csv_file(self.files_address["test_data_path"]).values.tolist()]
        test_data = data.Dataset(test_examples, data_fields)
        valid_examples = [data.Example.fromlist(row, data_fields) for row in
                          self.read_csv_file(self.files_address["valid_data_path"]).values.tolist()]
        valid_data = data.Dataset(valid_examples, data_fields)

        # all_data = data.Dataset(all_examples, data_fields)
        # train_data, test_data = all_data.split(split_ratio=0.8, random_state=random.seed(1234))
        # train_data, valid_data = train_data.split(split_ratio=0.8, random_state=random.seed(1234))
        # build vocab in all fields
        logging.info("Start creating text_field vocabs.")
        if ARGS.use_char:
            dictionary_fields["text_field"].build_vocab(train_data)
            if ARGS.use_onehot:
                embedding_mat = self.get_embedding_matrix(list(
                    dictionary_fields["text_field"].vocab.stoi.keys()))

                dictionary_fields["text_field"].vocab.set_vectors(
                    dictionary_fields["text_field"].vocab.stoi,
                    torch.FloatTensor(embedding_mat),
                    len(dictionary_fields["text_field"].vocab.stoi))

            self.embedding_dict["vocab_embedding_vectors"] = \
                dictionary_fields["text_field"].vocab.vectors
        else:
            dictionary_fields["text_field"].build_vocab(train_data,
                                                        min_freq=min_freq,
                                                        unk_init=torch.Tensor.normal_,
                                                        vectors=Vectors(
                                                            self.files_address["embedding_path"]
                                                        ))
        self.text_field = dictionary_fields["text_field"]

        # self.pos_tags = dictionary_fields["pos_field"].vocab
        # self.pos_pad_idx ===>
        # dictionary_fields["pos_field"].vocab.stoi[dictionary_fields["pos_field"].pad_token]
        # self.pos_unk_idx ===>
        # dictionary_fields["pos_field"].vocab.stoi[dictionary_fields["pos_field"].unk_token]
        self.embedding_dict["vocab_embedding_vectors"] = \
            dictionary_fields["text_field"].vocab.vectors

        logging.info("Start creating label_field vocabs.")
        dictionary_fields["label_field"].build_vocab(train_data)

        # count number of unique vocab in all fields
        self.num_vocab_dict = self.calculate_num_vocabs(dictionary_fields)

        # get pad index in all fields
        self.pad_idx_dict = self.find_pad_index(dictionary_fields)

        # get unk index in all fields
        self.unk_idx_dict = self.find_unk_index(dictionary_fields)

        # calculate class weight for handling imbalanced data
        self.class_weight = self.calculate_class_weight(dictionary_fields)
        self.save_fields(dictionary_fields, text_field_path, label_field_path)

        # creating iterators for training model
        logging.info("Start creating iterator.")
        self.iterator_dict = self.creating_iterator(train_data=train_data,
                                                    valid_data=valid_data,
                                                    test_data=test_data,
                                                    batch_size=batch_size,
                                                    device=device)

        logging.info("Loaded %d train examples", len(train_data))
        logging.info("Loaded %d valid examples", len(valid_data))
        logging.info("Loaded %d test examples", len(test_data))

    @staticmethod
    def get_embedding_matrix(vocab_chars):
        """
        get_embedding_matrix method is written to
        create one-hot embedding for characters
        :param vocab_chars: all character in data
        :return:
            onehot_matrix: one-hot matrix for character embedding
        """
        # one hot embedding plus all-zero vector
        vocabulary_size = len(vocab_chars)
        onehot_matrix = np.eye(vocabulary_size, vocabulary_size)
        return onehot_matrix

    @staticmethod
    def save_fields(dictionary_fields, text_field_path, label_field_path):
        """
        save_fields method is writen for saving fields
        :param dictionary_fields: dictionary of fields
        :param text_field_path: path for text_field
        :param pos_field_path: path for pos_field
        """
        logging.info("Start saving fields...")
        # save text_field
        torch.save(dictionary_fields["text_field"], text_field_path)
        logging.info("text_field is saved.")

        # save label_field
        torch.save(dictionary_fields["label_field"], label_field_path)
        logging.info("label_field is saved.")

    @staticmethod
    def calculate_class_weight(dictionary_fields):
        """
        calculate_class_weight method is written for calculate class weight
        :param dictionary_fields: dictionary of fields
        :return:
            class_weights: calculated class weight
        """
        label_list = []
        print(dictionary_fields["label_field"].vocab.stoi.items())
        for label, idx in dictionary_fields["label_field"].vocab.stoi.items():

            for _ in range(dictionary_fields["label_field"].vocab.freqs[label]):
                label_list.append(idx)
        class_weights = class_weight.compute_class_weight(class_weight="balanced",
                                                          classes=np.unique(label_list),
                                                          y=label_list).astype(np.float32)
        return class_weights

    @staticmethod
    def creating_iterator(**kwargs):
        """
        creating_iterator method is written for create iterator for training model
        :param kwargs:
            train_data: train dataSet
            valid_data: validation dataSet
            test_data: test dataSet
            batch_size: number of sample in batch
            device: gpu or cpu

        :return:
            iterator_dict: dictionary of iterators
        """
        iterator_dict = {
            "train_iterator": data.BucketIterator(kwargs["train_data"],
                                                  batch_size=kwargs["batch_size"],
                                                  sort=False,
                                                  shuffle=True,
                                                  device=kwargs["device"]),
            "valid_iterator": data.BucketIterator(kwargs["valid_data"],
                                                  batch_size=kwargs["batch_size"],
                                                  sort=False,
                                                  shuffle=True,
                                                  device=kwargs["device"]),
            "test_iterator": data.BucketIterator(kwargs["test_data"],
                                                 batch_size=kwargs["batch_size"],
                                                 sort=False,
                                                 shuffle=True,
                                                 device=kwargs["device"])
        }
        return iterator_dict

    @staticmethod
    def calculate_num_vocabs(dictionary_fields):
        """
        calculate_num_vocabs method is written for calculate vocab counts in each field
        :param dictionary_fields: dictionary of fields
        :return:
            num_vocab_dict:  dictionary of vocab counts in each field
        """
        num_vocab_dict = dict()
        num_vocab_dict["num_token"] = len(dictionary_fields["text_field"].vocab)
        num_vocab_dict["num_label"] = len(dictionary_fields["label_field"].vocab)
        return num_vocab_dict

    @staticmethod
    def find_pad_index(dictionary_fields):
        """
        find_pad_index method is written for find pad index in each field
        :param dictionary_fields: dictionary of fields
        :return:
            pad_idx_dict: dictionary of pad index in each field
        """
        pad_idx_dict = dict()
        pad_idx_dict["token_pad_idx"] = dictionary_fields["text_field"] \
            .vocab.stoi[dictionary_fields["text_field"].pad_token]
        return pad_idx_dict

    @staticmethod
    def find_unk_index(dictionary_fields):
        """
        find_unk_index method is written for find unk index in each field
        :param dictionary_fields: dictionary of fields
        :return:
            unk_idx_dict: dictionary of unk index in each field
        """
        unk_idx_dict = dict()
        unk_idx_dict["token_unk_idx"] = dictionary_fields["text_field"] \
            .vocab.stoi[dictionary_fields["text_field"].unk_token]
        return unk_idx_dict


def init_weights(model):
    """
    init_weights method is written for initialize model parameters
    :param model: input model
    """
    for _, param in model.named_parameters():
        torch.nn.init.normal_(param.data, mean=0, std=0.1)


def initialize_weights_xavier_uniform_(model):
    """
    init_weights method is written for initialize model parameters
    :param model: input model
    """
    if hasattr(model, "weight") and model.weight.dim() > 1:
        torch.nn.init.xavier_uniform_(model.weight.data)


def read_model_config(config_path):
    """
    read model name by model path
    """
    with open(config_path) as json_file:
        json_data = json.load(json_file)
    return json_data
