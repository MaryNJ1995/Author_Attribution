#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=no-member
# pylint: disable-msg=arguments-differ
# pylint: disable-msg=import-error

"""
DenseAA.py is written for test AA model
"""
import time
import hazm
import torch
from torch import nn
import numpy as np
import pandas as pd
from author_attribution.config.config import get_config
from author_attribution.methods.model import DCBiLstmAttentions

__author__ = "Maryam Najafi"
__organization__ = "AuthorShip Attribution"
__license__ = "Public Domain"
__version__ = "1.0.0"
__email__ = "Maryam_Najafi73@yahoo.com"
__status__ = "Production"
__date__ = "11/14/2021"

np.seterr(divide='ignore', invalid='ignore')

ARGS = get_config()


class TestModel:
    """
    A class yo test the model quality in case of accuracy and timing
    """

    def __init__(self):

        self.model_path = ARGS.final_model_path
        self.normalizer = hazm.Normalizer()
        self.text_field, self.label_field = self.load_fields(
            ARGS.text_field_path, ARGS.label_field_path)
        self.model = self.load_model(self.text_field, self.label_field)
        self.dataframe = pd.read_csv(ARGS.test_data)

    def load_model(self, text_field, label_field):
        """
        low model with prposed hyperparameters
        """
        model = DCBiLstmAttentions(vocab_size=len(text_field.vocab),
                                   embedding_dim=ARGS.embedding_dim,
                                   lstm_units=ARGS.lstm_units,
                                   output_size=len(label_field.vocab),
                                   lstm_layers=ARGS.lstm_layers,
                                   bidirectional=ARGS.bidirectional,
                                   start_dropout=ARGS.start_dropout,
                                   middle_dropout=ARGS.middle_dropout,
                                   pad_idx=text_field.vocab.stoi[text_field.pad_token],
                                   final_dropout=ARGS.last_dropout,
                                   dropout=ARGS.dropout)
        model.load_state_dict(torch.load(self.model_path, map_location=torch.device("cpu")))
        return model

    @staticmethod
    def load_fields(text_field_path, label_field_path):
        """
        load the text and label filed pickles
        """
        text_field = torch.load(text_field_path)
        label_field = torch.load(label_field_path)
        return text_field, label_field

    @staticmethod
    def tokenizing_data(vocab, min_len=5):
        """
        word tokenize the input sentence
        """
        tokenized_text = hazm.word_tokenize(vocab)
        if len(tokenized_text) < min_len:
            for i in range(min_len - len(tokenized_text)):
                tokenized_text.append("<pad>")
        return tokenized_text

    def indexing_data(self, vocab):
        """
        index the input text to be verified
        """
        return [self.text_field.vocab.stoi[t] for t in vocab]

    def run_flask(self, input_dict):
        input_text = input_dict["text"]
        tokenized_text = self.tokenizing_data(input_text)
        # the input text should be indexed into numbers
        indexed_text = self.indexing_data(tokenized_text)
        # the indexes should be changed into tensors
        tensor_text = torch.LongTensor(indexed_text)
        tensor_text = tensor_text.unsqueeze(0)
        self.model.eval()
        softmax = nn.Softmax(-1)
        with torch.no_grad():
            predict = softmax(self.model(tensor_text)).tolist()
        pred = int(np.argmax(predict, axis=1))
        author_name = self.label_field.vocab.itos[pred]
        output_dict = dict()
        output_dict["author_name"] = author_name

        return output_dict


if __name__ == "__main__":
    INSTANCE = TestModel()
    # ins.run_prediction()
    START_TIME = time.time()
    RESULT = INSTANCE.run_flask(
        input_dict={
            "data": "برخورد با رو دربایستی‌های بی‌جا و تعارف‌های غیرمعمول رواج چنین فرهنگی که اصل "
                    "بر روشن‌بودن و یا روشن‌شدن پدیدآور و یا راوی خبر است موجب خواهد شد تا بازار تعارفات "
                    "غیر معمول و رودربایستی‌های نامیمون از رونق بیفتد. تقویت روحیه شجاعت و صراحت لهجه در"
                    " جامعه پدیدآورمحوری سبب خواهد شد تا جامعه به شجاعت خو کند. به طور مثال اگر می‌خواهد"
                    " حرفی را بزند، خود، با شهامت آن‌را ابراز کند. بدیهی است صراحت لهجه به ویژه در اموری "
                    "که به تصمیم‌گیری منجر می‌شود بسیار اهمیت دارد. پرهیز از پرگویی، کنایه‌گویی، استعاره و"
                    " امثال آن به ویژه به هنگام تصمیم‌گیری زیادگفتن خود موجب لغزش است و به قول شاعر: کم بگو "
                    "اما که پر معنا بگو (فدائی کزازی، ۱۳۹۲). هم¬چنین استفاده از کنایه، استعاره و امثال آن"
                    " در جایی است که ترس، که حکایت از ضعف دارد، وجود داشته¬باشد. ترس، زمانی اتفاق می‌افتد"
                    " که فرد بیم آن دارد که چیزی را از دست‌ بدهد، و یا امید دارد چیزی را به‌دست‌ آورد. "
        })
    END_TIME = time.time()
    print(RESULT)
    print(END_TIME - START_TIME)

#     test_set = [
#         "@delnaha منم درباره شرایط ایده‌آل حرف نمی‌زنم، درباره رعایت همین قانون"
#         " - پر از اشکال - جمهوری اسلامی حرف می‌زنم."
#         " ضمن این‌که اگر قرار به آوردن صرف ۲-۳ مصداق باشه، می‌شه از لاجوردی هم فرشته عدالت ساخت",
#         "@FarnazML64 سرکار خانم این نشان دهنده وطن دوستی شماست . پاینده باشید سپاس از شما",
#         "@mansoori66 /ولی اگه بعد بفهمم که تو کار پایان‌نامه‌ت پیش اون بابا گیره"
#         "یا یه پولی/موقعیتی گرفته‌ای که سنگش رو به سینه بزنی، حس دورخوردن دارم.",
#         "@Ehsanism البته بعد از زدن ریش. چون خدای نکرپه مجاهدین خودتو میبرن نذری میدن تیرنا!"
#     ]
#     true_list = ["ArashBahmani", "najafi_tehrani", "1hoseim", "h0d3r_fa"]
#     ins = TestModel()
#     for item in test_set:
#         s_time = time.time()
#         # ins.run_prediction()
#         output = ins.__run_flask__(input_dict={"text": item})
#         print(output)
#         time.sleep(1000)
#     print((time.time() - s_time) / 10)
