import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader
from util.logger import get_logger
from torch.utils.data import Dataset

logger = get_logger(__name__)


class CustomTranslationDataset(Dataset):
    def __init__(self, dataset) -> None:
        """
        Custom dataset wrapper for translation datasets that implements the __len__ method.

        Args:
            dataset: The original dataset (e.g., Multi30k) that does not implement __len__.
        """
        self.dataset = list(dataset)
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.dataset[index]


class TranslationDataLoader:

    def __init__(self, language_pair, tokenize_en, tokenize_de):
        self.language_pair = language_pair
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = "<sos>"
        self.eos_token = "<eos>"
        self.tokenizer_source = None
        self.tokenizer_target = None
        self.source_vocab = None
        self.target_vocab = None
        logger.info("Dataset initializing done")

    def make_dataset(self):
        logger.info("Making dataset process initiated...")
        logger.info(
            f"The source language is {self.language_pair[0]} and the target language is {self.language_pair[1]}"
        )
        if self.language_pair == ("de", "en"):
            self.tokenizer_source = get_tokenizer(
                self.tokenize_de, language="de_core_news_sm"
            )
            self.tokenizer_target = get_tokenizer(
                self.tokenize_en, language="en_core_web_sm"
            )
        elif self.language_pair == ("en", "de"):
            self.tokenizer_source = get_tokenizer(
                self.tokenize_en, language="en_core_web_sm"
            )
            self.tokenizer_target = get_tokenizer(
                self.tokenize_de, language="de_core_news_sm"
            )

        train_data = Multi30k(split="train", language_pair=self.language_pair)
        train_data = CustomTranslationDataset(train_data)
        val_data = Multi30k(split="valid", language_pair=self.language_pair)
        val_data = CustomTranslationDataset(val_data)
        test_data = Multi30k(split="test", language_pair=self.language_pair)
        # test_data = CustomTranslationDataset(test_data)
        logger.info("Making dataset process completed")
        return train_data, val_data, test_data

    def build_vocab(self, train_data):
        """
        Builds the source and target vocabularies for the given train_data.

        Parameters:
            train_data (iterable): An iterable containing the training data.
            Each element of the iterable should be a tuple of two strings
            representing the source and target data.

        Returns:
            None
        """

        def yield_tokens(data_iter, tokenizer):
            for data in data_iter:
                yield tokenizer(data[0])
                yield tokenizer(data[1])

        self.source_vocab = build_vocab_from_iterator(
            yield_tokens(train_data, self.tokenizer_source),
            specials=["<unk>", "<pad>", self.init_token, self.eos_token],
        )
        self.source_vocab.set_default_index(self.source_vocab["<unk>"])

        self.target_vocab = build_vocab_from_iterator(
            yield_tokens(train_data, self.tokenizer_target),
            specials=["<unk>", "<pad>", self.init_token, self.eos_token],
        )
        self.target_vocab.set_default_index(self.target_vocab["<unk>"])

    def collate_batch(self, batch):
        """
        Collates a batch of data into source and target batches.

        Args:
            batch: A list of tuples containing the source and target data.

        Returns:
            A tuple of two tensors: the source batch and the target batch.
            The source batch and target batch are padded to the same length.
        """
        source_pipeline = lambda x: [
            self.source_vocab[token] for token in self.tokenizer_source(x)
        ]
        target_pipeline = lambda x: [
            self.target_vocab[token] for token in self.tokenizer_target(x)
        ]

        source_batch, target_batch = [], []
        for src, tgt in batch:
            source_batch.append(
                torch.tensor(
                    [self.source_vocab[self.init_token]]
                    + source_pipeline(src)
                    + [self.source_vocab[self.eos_token]],
                    dtype=torch.int64,
                )
            )
            target_batch.append(
                torch.tensor(
                    [self.target_vocab[self.init_token]]
                    + target_pipeline(tgt)
                    + [self.target_vocab[self.eos_token]],
                    dtype=torch.int64,
                )
            )

        # Find the maximum length in both source and target batches
        max_source_len = max(len(src) for src in source_batch)
        max_target_len = max(len(tgt) for tgt in target_batch)
        max_len = max(max_source_len, max_target_len)

        # Pad both source and target batches to the maximum length
        source_batch = torch.nn.utils.rnn.pad_sequence(
            source_batch, padding_value=self.source_vocab["<pad>"], batch_first=True
        )
        target_batch = torch.nn.utils.rnn.pad_sequence(
            target_batch, padding_value=self.target_vocab["<pad>"], batch_first=True
        )

        # Pad both source and target to the same length
        if source_batch.size(1) < max_len:
            padding = torch.full(
                (source_batch.size(0), max_len - source_batch.size(1)),
                self.source_vocab["<pad>"],
                dtype=torch.int64,
            )
            source_batch = torch.cat([source_batch, padding], dim=1)

        if target_batch.size(1) < max_len:
            padding = torch.full(
                (target_batch.size(0), max_len - target_batch.size(1)),
                self.target_vocab["<pad>"],
                dtype=torch.int64,
            )
            target_batch = torch.cat([target_batch, padding], dim=1)

        return source_batch, target_batch

    def make_train_iter(self, train_data, batch_size):
        train_loader = DataLoader(
            train_data, batch_size=batch_size, collate_fn=self.collate_batch
        )
        return train_loader

    def make_val_iter(self, val_data, batch_size):
        val_loader = DataLoader(
            val_data, batch_size=batch_size, collate_fn=self.collate_batch
        )
        return val_loader

    def make_test_iter(self, test_data, batch_size):
        test_loader = DataLoader(
            test_data, batch_size=batch_size, collate_fn=self.collate_batch
        )
        return test_loader


if __name__ == "__main__":
    language_pair = ("de", "en")
    tokenize_en = "spacy"
    tokenize_de = "spacy"
    batch_size = 32

    translation_data_loader = TranslationDataLoader(
        language_pair, tokenize_en, tokenize_de
    )
    _train_data, _, _ = translation_data_loader.make_dataset()
    translation_data_loader.build_vocab(_train_data)
    _train_loader = translation_data_loader.make_train_iter(_train_data, batch_size)

    counter = 0
    for src_batch, tgt_batch in _train_loader:
        logger.info(f"source shape: {src_batch.shape}, target shape: {tgt_batch.shape}")
        counter += 1
        if counter > 5:
            break
