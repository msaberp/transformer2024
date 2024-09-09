import math
import time
import os
from tqdm import tqdm
import shutil

from torch import nn, optim
from torch.optim import Adam
from omegaconf import DictConfig

import torch
from data import loader, get_data_iter
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time
from util.logger import get_logger
from torch.utils.tensorboard import SummaryWriter

tensorboard_writer = SummaryWriter()
logger = get_logger(__name__)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)


def save_results(train_losses, test_losses, bleus):
    os.makedirs("result", exist_ok=True)
    with open("result/train_loss.txt", "w") as f:
        f.write(str(train_losses))
    with open("result/bleu.txt", "w") as f:
        f.write(str(bleus))
    with open("result/test_loss.txt", "w") as f:
        f.write(str(test_losses))


def train_one_epoch(cfg, model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0

    # Get terminal width and set tqdm bar to 1/3rd of it
    terminal_width = shutil.get_terminal_size().columns
    tqdm_width = terminal_width // 3

    # Simplified bar format to include loss
    progress_bar = tqdm(iterator, desc="Training", ncols=tqdm_width)
    progress_bar.set_postfix({"loss": "0.0"})

    for i, batch in enumerate(progress_bar):
        src, trg = batch[0], batch[1]
        optimizer.zero_grad()

        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.clip)
        optimizer.step()

        epoch_loss += loss.item()

        # Update tqdm bar with live loss value
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        if cfg.runner.dry_run:
            break

    return epoch_loss / len(iterator)

def evaluate_one_epoch(cfg, model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []

    terminal_width = shutil.get_terminal_size().columns
    tqdm_width = terminal_width // 3

    progress_bar = tqdm(iterator, desc="Evaluating", ncols=tqdm_width)
    progress_bar.set_postfix({"loss": "0.0"})

    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            src, trg = batch[0], batch[1]
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            # Update tqdm bar with live loss value
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            total_bleu = []
            if not cfg.dataset.drop_last:
                batch_length = len(batch[0])
            else:
                batch_length = cfg.dataset.batch_size

            for j in range(batch_length):
                try:
                    trg_words = idx_to_word(batch[1][j], loader.target_vocab.vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target_vocab.vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except Exception as e:
                    logger.error(f"BLEU calculation error: {e}")

            if total_bleu:
                batch_bleu.append(sum(total_bleu) / len(total_bleu))

            if cfg.runner.dry_run:
                break

    avg_bleu = sum(batch_bleu) / len(batch_bleu) if batch_bleu else 0
    return epoch_loss / len(iterator), avg_bleu


def run(cfg: DictConfig):
    src_pad_idx = loader.source_vocab['<pad>']
    trg_pad_idx = loader.target_vocab['<pad>']
    trg_sos_idx = loader.target_vocab['<sos>']
    enc_voc_size = len(loader.target_vocab)
    dec_voc_size = len(loader.target_vocab)
    train_iter, val_iter, _ = get_data_iter(cfg)

    device = torch.device(cfg.runner.device)
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
        device=device,
    ).to(device)

    logger.info(f"Model has {count_parameters(model):,} trainable parameters")
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

    best_loss = float("inf")

    for epoch in range(cfg.runner.epoch):
        train_loss = train_one_epoch(cfg, model, train_iter, optimizer, criterion)
        tensorboard_writer.add_scalar("Loss/train", train_loss, epoch)
        val_loss, bleu = evaluate_one_epoch(cfg, model, val_iter, criterion)
        tensorboard_writer.add_scalar("Loss/val", val_loss, epoch)
        tensorboard_writer.add_scalar("BLEU/val", bleu, epoch)

        if epoch > cfg.runner.warmup:
            scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs("saved", exist_ok=True)
            torch.save(model.state_dict(), f"saved/model-{val_loss:.3f}.pt")

        tensorboard_writer.add_scalar("PPL/train", math.exp(train_loss), epoch)
        tensorboard_writer.add_scalar("PPL/val", math.exp(val_loss), epoch)

        if cfg.runner.dry_run:
            break
    
    tensorboard_writer.flush()
    tensorboard_writer.close()
    logger.info("Training finished.")
