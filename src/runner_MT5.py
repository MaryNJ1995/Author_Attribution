import os
import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping
import time
from transformers import MT5Tokenizer
from configuration import BaseConfig
from data_loader import read_csv, write_json
from data_preparation import Indexer
from model import DataModule, Classifier, build_checkpoint_callback

__author__ = "Maryam Najafi"
__organization__ = "Religious ChatBot"
__license__ = "Public Domain"
__version__ = "1.1.0"
__email__ = "Maryam_Najafi73@yahoo.com"
__status__ = "Production"
__date__ = "07/27/2021"
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()
    TOKENIZER = MT5Tokenizer.from_pretrained(CONFIG.lm_model_path)
    LOGGER = CSVLogger(CONFIG.log_dir, name="Project_FineTune_LM_TextClassification_Log")

    # load raw data
    RAW_TRAIN_DATA = read_csv(path=os.path.join(CONFIG.raw_data_dir, CONFIG.train_file),
                              columns=CONFIG.data_headers,
                              names=CONFIG.customized_headers).dropna()
    RAW_VAL_DATA = read_csv(path=os.path.join(CONFIG.raw_data_dir, CONFIG.val_file),
                            columns=CONFIG.data_headers,
                            names=CONFIG.customized_headers).dropna()
    RAW_TEST_DATA = read_csv(path=os.path.join(CONFIG.raw_data_dir, CONFIG.test_file),
                             columns=CONFIG.data_headers,
                             names=CONFIG.customized_headers).dropna()
    logging.debug(RAW_TRAIN_DATA.head(), RAW_VAL_DATA.head(), RAW_TEST_DATA.head())
    logging.debug("length of Train data is: {}".format(len(RAW_TRAIN_DATA)))
    logging.debug("length of Val data is: {}".format(len(RAW_VAL_DATA)))
    logging.debug("length of Test data is: {}".format(len(RAW_TEST_DATA)))

    TARGET_INDEXER = Indexer(RAW_TRAIN_DATA[CONFIG.customized_headers[1]])
    TARGET_INDEXER.build_vocab2idx()

    TRAIN_TARGETS = [[target] for target in RAW_TRAIN_DATA[CONFIG.customized_headers[1]]]
    TRAIN_TARGETS = TARGET_INDEXER.convert_samples_to_indexes(TRAIN_TARGETS)

    VAL_TARGETS = [[target] for target in RAW_VAL_DATA[CONFIG.customized_headers[1]]]
    VAL_TARGETS = TARGET_INDEXER.convert_samples_to_indexes(VAL_TARGETS)

    TEST_TARGETS = [[target] for target in RAW_TEST_DATA[CONFIG.customized_headers[1]]]
    TEST_TARGETS = TARGET_INDEXER.convert_samples_to_indexes(TEST_TARGETS)

    logging.debug("Maximum length is: {}".format(CONFIG.max_length))

    TRAIN_DATA = [list(RAW_TRAIN_DATA[CONFIG.customized_headers[0]]),
                  TRAIN_TARGETS]

    VAL_DATA = [list(RAW_VAL_DATA[CONFIG.customized_headers[0]]),
                VAL_TARGETS]

    TEST_DATA = [list(RAW_TEST_DATA[CONFIG.customized_headers[0]]),
                 TEST_TARGETS]

    DATA_MODULE = DataModule(train_data=TRAIN_DATA, val_data=VAL_DATA, test_data=TEST_DATA,
                             tokenizer=TOKENIZER, batch_size=CONFIG.batch_size,
                             max_token_len=CONFIG.max_length, num_workers=CONFIG.num_workers)

    DATA_MODULE.setup()

    CHECKPOINT_CALLBACK = build_checkpoint_callback(CONFIG.save_top_k)
    EARLY_STOPPING_CALLBACK = EarlyStopping(monitor="val_loss", patience=5)

    # Instantiate the Model Trainer
    TRAINER = pl.Trainer(max_epochs=CONFIG.n_epochs, gpus=[0],  # CONFIG.num_of_gpu,
                         callbacks=[CHECKPOINT_CALLBACK, EARLY_STOPPING_CALLBACK], progress_bar_refresh_rate=60,
                         logger=LOGGER)  # min_epochs=10
    # Train the Classifier Model
    STEPS_PER_EPOCH = len(TRAIN_DATA) // CONFIG.batch_size
    N_CLASSES = len(TARGET_INDEXER.vocabs)
    logging.debug("number of class is: {}".format(N_CLASSES))
    MODEL = Classifier(CONFIG, steps_per_epoch=STEPS_PER_EPOCH, n_classes=N_CLASSES)
    MODEL.train()
    TRAINER.fit(MODEL, DATA_MODULE)
    START_TIME = time.time()
    TRAINER.test(ckpt_path='best', datamodule=DATA_MODULE)
    print('test time is :', (time.time() - START_TIME) / len(DATA_MODULE.test_dataloader().dataset))

    # save best mt5_model_en path
    write_json(path=CONFIG.log_dir + 'logs/best_model_path.json',
               data={'best_model_path': CHECKPOINT_CALLBACK.best_model_path})

    # BEST_MODEL = MODEL.load_from_checkpoint(
    #     "./logs/my_exp_name/version_10/checkpoints/QTag-epoch=01-val_loss=0.32.ckpt",
    #     map_location="cpu",
    # )
    # BEST_MODEL.eval()
    # BEST_MODEL.freeze()
    # # TRAINER_BEST_EPOCH = pl.Trainer(weights_summary=None)
    # TRAINER.test(model=BEST_MODEL,
    #              datamodule=DATA_MODULE)

    # write best model path
    # write_json(path=CONFIG.assets_dir + "logs/b_model_path.json",
    #            data={"best_model_path": CHECKPOINT_CALLBACK.best_model_path})
