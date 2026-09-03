import time

from tqdm import tqdm
import torch
from rxrx1.training.checkpoint import save_checkpoint
from rxrx1.training.experiment import TrainingResults

from rxrx1.utils.logger import (
    log_epoch_result,
    log_best_checkpoint,
)

from rxrx1.utils.tracking import log_wandb_epoch


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    batch_normalizer=None,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        if batch_normalizer is not None:
            images = batch_normalizer(images, batch)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

        pbar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
        )

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    batch_normalizer=None,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Val", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        if batch_normalizer is not None:
            images = batch_normalizer(images, batch)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

        pbar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
        )

    return total_loss / total_samples, total_correct / total_samples


def fit_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    checkpoint_enabled,
    checkpoint_path,
    logger,
    run,
    train_batch_normalizer=None,
    val_batch_normalizer=None,
):
    if epochs <= 0:
        raise ValueError(f"epochs must be greater than 0, got {epochs}")

    results = TrainingResults()
    epoch_runtimes = []

    for epoch in tqdm(range(epochs), desc="Epoch"):
        epoch_number = epoch + 1
        epoch_start_time = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            batch_normalizer=train_batch_normalizer,
        )
        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            batch_normalizer=val_batch_normalizer,
        )

        epoch_runtime_seconds = time.perf_counter() - epoch_start_time
        epoch_runtime_minutes = epoch_runtime_seconds / 60
        epoch_runtimes.append(epoch_runtime_seconds)

        is_best = results.update_epoch(
            epoch=epoch_number,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )

        log_epoch_result(
            logger=logger,
            epoch=epoch_number,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            runtime_minutes=epoch_runtime_minutes,
        )
        log_wandb_epoch(
            run=run,
            epoch=epoch_number,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            runtime_minutes=epoch_runtime_minutes,
        )

        if is_best and checkpoint_enabled:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch_number,
                val_acc=val_acc,
                path=checkpoint_path,
            )
            log_best_checkpoint(
                logger=logger,
                epoch=epoch_number,
                val_acc=val_acc,
                val_loss=val_loss,
            )

    results.set_runtime(epoch_runtimes)
    return results

