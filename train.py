import math
import time
import os

from torch import nn, optim
from torch.optim import Adam
from omegaconf import DictConfig

import torch
from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time
from util.logger import get_logger

logger = get_logger(__name__)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)


def train_data(cfg, model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch[0]
        trg = batch[1]

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.clip)
        optimizer.step()

        epoch_loss += loss.item()
        logger.info(
            f"step: {round((i / len(iterator)) * 100, 2)} , loss: {loss.item()}"
        )
        if cfg.runner.dry_run:
            break

    return epoch_loss / len(iterator)


def evaluate(cfg, model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0]
            trg = batch[1]
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(cfg.dataset.batch_size):
                try:
                    trg_words = idx_to_word(batch[1][j], loader.target_vocab.vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target_vocab.vocab)
                    bleu = get_bleu(
                        hypotheses=output_words.split(), reference=trg_words.split()
                    )
                    total_bleu.append(bleu)
                except Exception as e:
                    logger.error(f"During the bleu calculation, an error occurred: {e}")

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

            if cfg.runner.dry_run:
                break

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu


def run(cfg: DictConfig):
    model = Transformer(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        trg_sos_idx=trg_sos_idx,
        d_model=cfg.model.d_model,
        enc_voc_size=enc_voc_size,
        dec_voc_size=dec_voc_size,
        max_len=cfg.model.max_len,
        ffn_hidden=cfg.model.ffn_hidden,
        num_head=cfg.model.num_head,
        num_layers=cfg.model.num_layers,
        drop_prob=cfg.model.drop_prob,
        device=torch.device(cfg.runner.device),
    ).to(torch.device(cfg.runner.device))
    best_loss = float("inf")

    logger.info(f"The model has {count_parameters(model):,} trainable parameters")
    model.apply(initialize_weights)
    optimizer = Adam(
        params=model.parameters(),
        lr=cfg.optimizer.init_lr,
        weight_decay=cfg.optimizer.weight_decay,
        eps=cfg.optimizer.adam_eps,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=cfg.scheduler.factor,
        patience=cfg.scheduler.patience,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

    if not os.path.exists("result"):
        logger.info("Creating result directory...")
        os.makedirs("result")
    if not os.path.exists("saved"):
        logger.info("Creating saved directory...")
        os.makedirs("saved")

    train_losses, test_losses, bleus = [], [], []
    for step in range(cfg.runner.epoch):
        start_time = time.time()
        train_loss = train_data(cfg, model, train_iter, optimizer, criterion)
        valid_loss, bleu = evaluate(cfg, model, valid_iter, criterion)
        end_time = time.time()

        if step > cfg.runner.warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), "saved/model-{0}.pt".format(valid_loss))

        f = open("result/train_loss.txt", "w")
        f.write(str(train_losses))
        f.close()

        f = open("result/bleu.txt", "w")
        f.write(str(bleus))
        f.close()

        f = open("result/test_loss.txt", "w")
        f.write(str(test_losses))
        f.close()

        logger.info(f"Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s")
        logger.info(
            f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
        )
        logger.info(
            f"\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}"
        )
        logger.info(f"\tBLEU Score: {bleu:.3f}")

        if cfg.runner.dry_run:
            break

    logger.info("Training finished.")
