from util.data_loader import TranslationDataLoader
from util.tokenizer import Tokenizer
from omegaconf import DictConfig

tokenizer = Tokenizer()
loader = TranslationDataLoader(
    language_pair=("en", "de"),
    tokenize_en=tokenizer.tokenize_en,
    tokenize_de=tokenizer.tokenize_de,
)
train_data, val_data, test_data = loader.make_dataset()
loader.build_vocab(train_data=train_data)


def get_data_iter(cfg: DictConfig):
    train_iter = loader.make_train_iter(
        train_data, batch_size=cfg.dataset.batch_size, drop_last=cfg.dataset.drop_last
    )
    val_iter = loader.make_val_iter(
        val_data, batch_size=cfg.dataset.batch_size, drop_last=cfg.dataset.drop_last
    )
    test_iter = loader.make_test_iter(
        test_data, batch_size=cfg.dataset.batch_size, drop_last=cfg.dataset.drop_last
    )
    return train_iter, val_iter, test_iter
