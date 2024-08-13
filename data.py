from util.data_loader import TranslationDataLoader
from util.tokenizer import Tokenizer
from hydra import compose, initialize

with initialize(config_path="config", version_base="1.3"):
    cfg = compose(config_name="config")

tokenizer = Tokenizer()
loader = TranslationDataLoader(language_pair=('en', 'de'),
                               tokenize_en=tokenizer.tokenize_en,
                               tokenize_de=tokenizer.tokenize_de)

train_data, val_data, test_data = loader.make_dataset()
loader.build_vocab(train_data=train_data)
train_iter = loader.make_test_iter(train_data, batch_size=cfg.dataset.batch_size)
valid_iter = loader.make_test_iter(val_data, batch_size=cfg.dataset.batch_size)
test_iter = loader.make_test_iter(test_data, batch_size=cfg.dataset.batch_size)

src_pad_idx = loader.source_vocab['<pad>']
trg_pad_idx = loader.target_vocab['<pad>']
trg_sos_idx = loader.target_vocab['<sos>']

enc_voc_size = len(loader.target_vocab)
dec_voc_size = len(loader.target_vocab)
