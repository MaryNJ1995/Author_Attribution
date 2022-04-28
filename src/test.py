import os
import hazm
import logging
import copy
import pytorch_lightning as pl

from configuration import BaseConfig
from data_loader import read_csv, write_json
from data_preparation import word_tokenizer, Indexer, TokenIndexer, build_exact_match
from embedder import Embed
from utils import filter_by_length, filter_by_value, item_counter, filter_by_count, convert_words_to_chars
from model import DataModule, Classifier, build_checkpoint_callback

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    # load raw data
    TRAIN_DATA = read_csv(path=os.path.join(ARGS.raw_data_dir, ARGS.train_file), columns=ARGS.data_headers,
                          names=ARGS.customized_headers)[:100]
    TEST_DATA = read_csv(path=os.path.join(ARGS.raw_data_dir, ARGS.train_file), columns=ARGS.data_headers,
                         names=ARGS.customized_headers)[:100]
    VAL_DATA = read_csv(path=os.path.join(ARGS.raw_data_dir, ARGS.train_file), columns=ARGS.data_headers,
                        names=ARGS.customized_headers)[:100]

    # tokenizing data
    TRAIN_TOKENIZED_FIRST_TEXT = word_tokenizer(list(TRAIN_DATA["first_text"]), hazm.word_tokenize)
    TRAIN_TOKENIZED_SECOND_TEXT = word_tokenizer(list(TRAIN_DATA["second_text"]), hazm.word_tokenize)
    TEST_TOKENIZED_FIRST_TEXT = word_tokenizer(list(TEST_DATA["first_text"]), hazm.word_tokenize)
    TEST_TOKENIZED_SECOND_TEXT = word_tokenizer(list(TEST_DATA["second_text"]), hazm.word_tokenize)
    VAL_TOKENIZED_FIRST_TEXT = word_tokenizer(list(VAL_DATA["first_text"]), hazm.word_tokenize)
    VAL_TOKENIZED_SECOND_TEXT = word_tokenizer(list(VAL_DATA["second_text"]), hazm.word_tokenize)

    # filter sentences by length
    FILTERED_FIRST_TEXT = filter_by_length(TRAIN_TOKENIZED_FIRST_TEXT, lower_bound=ARGS.min_length,
                                           upper_bound=ARGS.max_length)
    FILTERED_SECOND_TEXT = filter_by_length(TRAIN_TOKENIZED_SECOND_TEXT, lower_bound=ARGS.min_length,
                                            upper_bound=ARGS.max_length)

    assert len(FILTERED_FIRST_TEXT) == len(FILTERED_SECOND_TEXT), "bad filtering"

    FILTERED_FIRST_TEXT, FILTERED_SECOND_TEXT, FILTERED_TARGETS = filter_by_value(FILTERED_FIRST_TEXT,
                                                                                  FILTERED_SECOND_TEXT,
                                                                                  list(TRAIN_DATA["targets"]))
    COPY_FILTERED_FIRST_TEXT = copy.deepcopy(FILTERED_FIRST_TEXT)
    COPY_FILTERED_SECOND_TEXT = copy.deepcopy(FILTERED_SECOND_TEXT)
    assert len(FILTERED_FIRST_TEXT) == len(FILTERED_SECOND_TEXT), "bad filtering"
    print(len(FILTERED_FIRST_TEXT))
    # token counter
    VOCAB2COUNT = item_counter(FILTERED_FIRST_TEXT + FILTERED_SECOND_TEXT)

    # filter by count
    FILTERED_VOCAB2COUNT = filter_by_count(VOCAB2COUNT, ARGS.min_count, ARGS.max_count)

    # indexing targets
    TARGET_INDEXER = Indexer(vocabs=FILTERED_TARGETS)
    TARGET_INDEXER.build_vocab2idx()

    TRAIN_TARGETS_CONVENTIONAL = [[target] for target in list(TRAIN_DATA.labels)]
    TRAIN_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(TRAIN_TARGETS_CONVENTIONAL)

    TEST_TARGETS_CONVENTIONAL = [[target] for target in list(TEST_DATA.labels)]
    TEST_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(TEST_TARGETS_CONVENTIONAL)

    VAL_TARGETS_CONVENTIONAL = [[target] for target in list(VAL_DATA.labels)]
    VAL_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(VAL_TARGETS_CONVENTIONAL)

    # indexing tokens
    TOKEN_INDEXER = TokenIndexer(vocabs=list(FILTERED_VOCAB2COUNT.keys()))
    TOKEN_INDEXER.build_vocab2idx()

    TRAIN_INDEXED_FIRST_TEXT = TOKEN_INDEXER.convert_samples_to_indexes(FILTERED_FIRST_TEXT)
    TRAIN_INDEXED_SECOND_TEXT = TOKEN_INDEXER.convert_samples_to_indexes(FILTERED_SECOND_TEXT)
    TEST_INDEXED_FIRST_TEXT = TOKEN_INDEXER.convert_samples_to_indexes(TEST_TOKENIZED_FIRST_TEXT)
    TEST_INDEXED_SECOND_TEXT = TOKEN_INDEXER.convert_samples_to_indexes(TEST_TOKENIZED_SECOND_TEXT)
    VAL_INDEXED_FIRST_TEXT = TOKEN_INDEXER.convert_samples_to_indexes(VAL_TOKENIZED_FIRST_TEXT)
    VAL_INDEXED_SECOND_TEXT = TOKEN_INDEXER.convert_samples_to_indexes(VAL_TOKENIZED_SECOND_TEXT)

    # indexing chars
    CHARS_INDEXER = TokenIndexer(vocabs=convert_words_to_chars(list(FILTERED_VOCAB2COUNT.keys())))
    CHARS_INDEXER.build_vocab2idx()
    CHAR_INDEXED_FIRST_TEXT = CHARS_INDEXER.convert_samples_to_char_indexes(COPY_FILTERED_FIRST_TEXT)
    CHAR_INDEXED_SECOND_TEXT = CHARS_INDEXER.convert_samples_to_char_indexes(COPY_FILTERED_SECOND_TEXT)

    # create exact_match
    TRAIN_EXACT_MATCH_FIRST_TEXT = build_exact_match(source_data=TRAIN_INDEXED_FIRST_TEXT,
                                                     compare_data=TRAIN_INDEXED_SECOND_TEXT,
                                                     skip_item=TOKEN_INDEXER.unk_index)
    TRAIN_EXACT_MATCH_SECOND_TEXT = build_exact_match(source_data=TRAIN_INDEXED_FIRST_TEXT,
                                                      compare_data=TRAIN_INDEXED_SECOND_TEXT,
                                                      skip_item=TOKEN_INDEXER.unk_index)

    assert len(TRAIN_INDEXED_FIRST_TEXT) == len(TRAIN_EXACT_MATCH_FIRST_TEXT), \
        "bad sentence exact match"
    assert len(TRAIN_INDEXED_FIRST_TEXT) == len(TRAIN_INDEXED_SECOND_TEXT), \
        "bad sentence indexing"
    assert len(TRAIN_INDEXED_FIRST_TEXT) == len(TRAIN_INDEXED_TARGET), \
        "bad target indexing"
    assert len(TRAIN_INDEXED_FIRST_TEXT) == len(CHAR_INDEXED_FIRST_TEXT), \
        "bad character indexing"

    # load embedding
    EMBED = Embed(path=os.path.join(ARGS.embedding_dir, ARGS.embedding_file))

    # build embedding matrix
    EMBED.build_embedding_matrix(TOKEN_INDEXER.get_vocab2idx())

    # dataset dicts
    TRAIN_PREMISES = {'words': TRAIN_INDEXED_FIRST_TEXT, 'pos': TRAIN_INDEXED_FIRST_TEXT,
                      'exact': TRAIN_EXACT_MATCH_FIRST_TEXT, 'char': CHAR_INDEXED_FIRST_TEXT}
    TRAIN_HYPOTHESES = {'words': TRAIN_INDEXED_SECOND_TEXT, 'pos': TRAIN_INDEXED_SECOND_TEXT,
                        'exact': TRAIN_EXACT_MATCH_SECOND_TEXT, 'char': CHAR_INDEXED_SECOND_TEXT}
    TEST_PREMISES = {'words': TEST_INDEXED_FIRST_TEXT, 'pos': TEST_INDEXED_FIRST_TEXT,
                     'exact': TRAIN_EXACT_MATCH_FIRST_TEXT, 'char': CHAR_INDEXED_FIRST_TEXT}
    TEST_HYPOTHESES = {'words': TEST_INDEXED_SECOND_TEXT, 'pos': TEST_INDEXED_SECOND_TEXT,
                       'exact': TRAIN_EXACT_MATCH_SECOND_TEXT, 'char': CHAR_INDEXED_SECOND_TEXT}
    VAL_PREMISES = {'words': VAL_INDEXED_FIRST_TEXT, 'pos': VAL_INDEXED_FIRST_TEXT,
                    'exact': TRAIN_EXACT_MATCH_FIRST_TEXT, 'char': CHAR_INDEXED_FIRST_TEXT}
    VAL_HYPOTHESES = {'words': VAL_INDEXED_SECOND_TEXT, 'pos': VAL_INDEXED_SECOND_TEXT,
                      'exact': TRAIN_EXACT_MATCH_SECOND_TEXT, 'char': CHAR_INDEXED_SECOND_TEXT}

    # Create Data Module
    DATA_MODULE = DataModule(train_first_cul=TRAIN_PREMISES, train_second_cul=TRAIN_HYPOTHESES,
                             train_target=TRAIN_INDEXED_TARGET,
                             val_first_cul=VAL_PREMISES, val_second_cul=VAL_HYPOTHESES,
                             val_target=VAL_INDEXED_TARGET,
                             test_first_cul=TEST_PREMISES, test_second_cul=TEST_HYPOTHESES,
                             test_target=TEST_INDEXED_TARGET, batch_size=ARGS.batch_size, num_workers=ARGS.num_workers,
                             pad_idx=TOKEN_INDEXER.pad_index)
    DATA_MODULE.setup()
    CHECKPOINT_CALLBACK = build_checkpoint_callback(save_top_k=ARGS.save_top_k)

    # Instantiate the Model Trainer
    TRAINER = pl.Trainer(max_epochs=ARGS.n_epoch, gpus=0, callbacks=[CHECKPOINT_CALLBACK], progress_bar_refresh_rate=30)

    # Create Model
    MODEL = Classifier(vocab_size=len(TOKEN_INDEXER.get_vocab2idx()), embedding_dim=ARGS.embedding_dim,
                       embedding_weights=EMBED.get_embedding_matrix(), pad_idx=TOKEN_INDEXER.pad_index, dense_size=100,
                       num_classes=len(TARGET_INDEXER.get_vocab2idx()))

    TRAINER.fit(MODEL, datamodule=DATA_MODULE)
    TRAINER.test(MODEL, datamodule=DATA_MODULE)

    # save best model path
    write_json(path=ARGS.assets_dir + 'logs/b_model_path.json',
               data={'best_model_path': CHECKPOINT_CALLBACK.best_model_path})
