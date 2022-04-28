import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

__author__ = "Maryam Najafi🥰"
__organization__ = "Author Attribution"
__license__ = "Public Domain"
__version__ = "1.0.0"
__email__ = "Maryaminj1995@gmail.com"
__status__ = "Production"
__date__ = "07/27/2021"


class CustomDataset(Dataset):
    """
    a class to make instance from dataframe
    """

    def __init__(self, texts: list, labels: list, max_len, tokenizer):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item_index):
        # first_text = self.first_texts[item_index]
        # first_text = " ".join(first_text.split()[:120])
        #
        # second_text = self.second_texts[item_index]
        # second_text = " ".join(second_text.split()[:50])
        #
        # label = self.targets[item_index]
        texts = self.texts[item_index]
        label = self.labels[item_index]
        inputs = self.tokenizer.encode_plus(
            text=texts,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].flatten()
        attn_mask = inputs["attention_mask"].flatten()
        token_type_ids = inputs["token_type_ids"].flatten()

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "token_type_ids": token_type_ids,
            "label": torch.tensor(label)
        }


class DataModule(pl.LightningDataModule):
    """
    class for setup train,test,validation batch from Dataloader
    """

    def __init__(self, num_workers, train_data, val_data, test_data,
                 tokenizer, batch_size, max_token_len):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.num_workers = num_workers
        self.test_data = test_data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def setup(self):
        """
        setup a CustomDataset for train,test,validation data
        """
        self.train_dataset = CustomDataset(texts=self.train_data[0],
                                           labels=self.train_data[1],
                                           tokenizer=self.tokenizer,
                                           max_len=self.max_token_len)

        self.val_dataset = CustomDataset(texts=self.val_data[0],
                                         labels=self.val_data[1],
                                         tokenizer=self.tokenizer,
                                         max_len=self.max_token_len)

        self.test_dataset = CustomDataset(texts=self.test_data[0],
                                          labels=self.test_data[1],
                                          tokenizer=self.tokenizer,
                                          max_len=self.max_token_len)

    def train_dataloader(self):
        """
        a DataLoader to Load train dataset with specific batch_sizes
        """
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        """
        a DataLoader to Load val dataset with specific batch_sizes
        """
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        """
        a DataLoader to Load test dataset with specific batch_sizes
        """
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size, num_workers=self.num_workers)
