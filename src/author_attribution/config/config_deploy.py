"""config.py is a configuration on deep method"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable-msg=no-member
import argparse
import torch

__author__ = "Maryam Najafi"
__organization__ = "AuthorShip Attribution"
__license__ = "Public Domain"
__version__ = "1.0.0"
__email__ = "Maryam_Najafi73@yahoo.com"
__status__ = "Production"
__date__ = "06/09/2021"


def get_config():
    """
    is a configuration on deep method
    """

    parser = argparse.ArgumentParser()
    # FollowerTweet
    # NovelBook
    # ----------------------------------- Data -----------------------------------
    parser.add_argument("--data_root", default='home/data/NewsPost/', type=str)

    parser.add_argument("--train_data",
                        default='home/data/NewsPost/news_chunk_part_train_unnorm.csv', type=str)
    parser.add_argument("--valid_data",
                        default='home/data/NewsPost/news_chunk_part_dev_unnorm.csv', type=str)
    parser.add_argument("--test_data",
                        default='home/data/NewsPost/news_chunk_part_test_unnorm.csv', type=str)
    # parser.add_argument("--embedding_path",
    #                     default='E:\DATASET\Embedding\cbow_news_300d_30e.txt', type=str)
    parser.add_argument("--embedding_path",
                        default="/home/maryam.najafi/Embeddings/cbow_news_300d_30e.txt", type=str)

    parser.add_argument("--model_name_config_path",
                        default='home/data/DataUsage/model_name_config.json', type=str)

    # ----------------------------------- Model -----------------------------------
    parser.add_argument("--model_root", default='', type=str)
    parser.add_argument("--loss_curve_path",
                        default='home/models/ID_x/Curves/loss_curve.png', type=str)
    parser.add_argument("--accuracy_curve_path",
                        default='home/models/ID_x/Curves/accuracy_curve.png', type=str)
    parser.add_argument("--text_field_path", default="home/models/ID_x/Fields/text_field", type=str)
    parser.add_argument("--label_field_path", default="home/models/ID_x/Fields/label_field", type=str)
    parser.add_argument("--pos_field_path", default="home/models/ID_x/Fields/pos_field", type=str)
    parser.add_argument("--lemma_field_path", default="home/models/ID_x/Fields/lemma_field", type=str)
    parser.add_argument("--cm_path", default='home/models/ID_x/Curves/cm.png', type=str)
    parser.add_argument("--log_path", default="home/models/ID_x/Logs/log.txt", type=str)
    parser.add_argument("--model_path", default="home/models/ID_x/", type=str)
    parser.add_argument("--final_model_path", default="home/models/ID_x/final_model.pt", type=str)
    parser.add_argument("--model_name", default="", type=str)

    # ----------------------------------- GeNeral -----------------------------------
    parser.add_argument("--device",
                        default=torch.device("cuda"
                                             if torch.cuda.is_available()
                                             else "cpu"), type=str)
    parser.add_argument("--random_seed", default=1234, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--n_epochs", default=15, type=int)
    parser.add_argument("--embedding_dim", default=300, type=int)
    parser.add_argument("--char_embedding_dim", default=16, type=int)
    parser.add_argument("--pos_embeddings", default=30, type=int)
    parser.add_argument("--min_freq", default=10, type=int)
    parser.add_argument("--max_len", default=150, type=int)
    parser.add_argument("--linear_units", default=1256, type=int)

    # ----------------------------------- Transformer -----------------------------------
    parser.add_argument("--encoder_layers", default=3, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--num_heads", default=8, type=int)
    parser.add_argument("--encoder_pf_dim", default=512, type=int)
    parser.add_argument("--fix_len", default=70, type=int)

    # ----------------------------------- DropOut -----------------------------------
    parser.add_argument("--start_dropout", default=0.3, type=float)
    parser.add_argument("--middle_dropout", default=0.35, type=float)
    parser.add_argument("--last_dropout", default=0.45, type=float)
    parser.add_argument("--dropout", default=0.3, type=float)

    # ----------------------------------- Features-----------------------------------
    parser.add_argument("--use_pos", default=False, type=bool)
    parser.add_argument("--use_char", default=False, type=bool)
    parser.add_argument("--use_onehot", default=False, type=bool)
    parser.add_argument("--use_lemma", default=False, type=bool)
    parser.add_argument("--use_transformer", default=False, type=bool)
    parser.add_argument("--use_elmo", default=False, type=bool)
    parser.add_argument("--use_stopword", default=False, type=bool)
    parser.add_argument("--use_bert", default=False, type=bool)

    # ----------------------------------- LSTM-----------------------------------
    parser.add_argument("--lstm_layers", default=1, type=int)
    parser.add_argument("--lstm_units", default=32, type=int)
    parser.add_argument("--bidirectional", default=True, type=bool)

    # ----------------------------------- CNN-----------------------------------
    parser.add_argument("--filter_size", default=[2, 3], type=list)
    parser.add_argument("--n_filters", default=16, type=int)

    args = parser.parse_args()
    return args
