import argparse
from pathlib import Path

__author__ = "Maryam Najafi"
__organization__ = "Religious ChatBot"
__license__ = "Public Domain"
__version__ = "1.0.0"
__email__ = "Maryam_Najafi73@yahoo.com"
__status__ = "Production"
__date__ = "07/27/2021"


class BaseConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--raw_data_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/FollowerTweet/")
        self.parser.add_argument("--log_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/results/")
        self.parser.add_argument("--assets_dir", type=str,
                                 default="/home/maryam.najafi/Project_FineTune_LM_TextMatching/assets/")
        self.parser.add_argument("--bert_model_path", type=str,
                                 default="/home/maryam.najafi/Project_FineTune_LM_TextMatching/"
                                         "assets/pretrained_models/MBert/model/",
                                 help="Path of the multilingual bert model dir")
        self.parser.add_argument("--bert_tokenizer_path", type=str,
                                 default=
                                 "pretrained_models/MBert/tokenizer/",
                                 help="Path of the multilingual bert tokenizer dir")
        self.parser.add_argument("--lm_model_path", type=str,
                                 default="/home/maryam.najafi/LanguageModels/mt5_en_large",
                                 help="Path of the multilingual MBert tokenizer dir")
        self.parser.add_argument("--train_file", type=str, default="train_data.csv")
        self.parser.add_argument("--val_file", type=str, default="dev_data.csv")
        self.parser.add_argument("--test_file", type=str, default="test_data.csv")

        self.parser.add_argument("--data_headers", type=list, default=["text", "label"])
        self.parser.add_argument("--customized_headers", type=list, default=["text", "label"])

        self.parser.add_argument("--num_of_gpu", type=int, default=1,
                                 help="numbers of gpus")
        self.parser.add_argument("--lr", type=float, default=2e-5,
                                 help="Learning Rate")
        self.parser.add_argument("--batch_size", type=int, default=16,
                                 help="Number Of batches")

        self.parser.add_argument("--max_length", type=int,
                                 default=150,
                                 help="...")
        self.parser.add_argument("--input_size", type=int,
                                 default=768,
                                 help="...")
        self.parser.add_argument("--hidden_size", type=int,
                                 default=128,
                                 help="...")

        self.parser.add_argument("--num_layers", type=int,
                                 default=1,
                                 help="...")
        self.parser.add_argument("--filter_sizes", type=list,
                                 default=[3, 4, 5],
                                 help="...")
        self.parser.add_argument("--n_filters", type=int,
                                 default=64,
                                 help="...")

        self.parser.add_argument("--num_workers", type=int,
                                 default=10,
                                 help="...")
        self.parser.add_argument("--save_top_k", type=int, default=1,
                                 help="number of best models should be saved(pytorch_lightening_config)")
        self.parser.add_argument("--n_epochs", type=int, default=100,
                                 help="Number Of Epochs")

    def get_config(self):
        return self.parser.parse_args()
