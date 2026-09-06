import time

from tqdm import tqdm
import torch

from rxrx1.training.checkpoint import save_checkpoint
from rxrx1.training.experiment import TrainingResults
from rxrx1.utils.logger import log_epoch_result, log_best_checkpoint
from rxrx1.utils.tracking import log_wandb_epoch


def _prepare_metadata(batch, device):
    return {
        "cell_type_idx": batch["cell_type_idx"].to(device),
        "well_position": batch["well_position"].to(device),
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scheduler=None,
    batch_normalizer=None,
    batch_transform=None,
):
    metric_enabled = getattr(criterion, "metric_enabled", False)
    if metric_enabled and batch_transform is not None:
        raise ValueError(
            "Metric supervision requires unmixed observations; disable batch_transform/MixUp/CutMix."
        )
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Train", leave=False)

    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        metadata = _prepare_metadata(batch, device)

        if batch_normalizer is not None:
            images = batch_normalizer(images, batch)

        targets = labels
        if batch_transform is not None:
            images, targets = batch_transform(images, labels)

        optimizer.zero_grad()

        if metric_enabled:
            outputs, embeddings = model(images, metadata, return_embeddings=True)
            loss = criterion(outputs, targets, embeddings, batch)
        else:
            outputs = model(images, metadata)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)
        if targets.ndim == 2:
            # Credit each prediction by its weight in the mixed target.
            total_correct += targets.gather(1, predictions[:, None]).sum().item()
        else:
            total_correct += (predictions == targets).sum().item()

        total_samples += batch_size

        pbar.set_postfix(
            loss=(f"{total_loss / total_samples:.4f}"), acc=(f"{total_correct / total_samples:.4f}")
        )

    return (total_loss / total_samples, total_correct / total_samples)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, batch_normalizer=None):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Val", leave=False)

    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        metadata = _prepare_metadata(batch, device)

        if batch_normalizer is not None:
            images = batch_normalizer(images, batch)

        outputs = model(images, metadata)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        total_correct += (outputs.argmax(dim=1) == labels).sum().item()

        total_samples += batch_size

        pbar.set_postfix(
            loss=(f"{total_loss / total_samples:.4f}"), acc=(f"{total_correct / total_samples:.4f}")
        )

    return (total_loss / total_samples, total_correct / total_samples)


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
    scheduler=None,
    train_batch_normalizer=None,
    val_batch_normalizer=None,
    epoch_callback=None,
    train_batch_transform=None,
):
    if epochs <= 0:
        raise ValueError(f"epochs must be greater than 0, got {epochs}")

    results = TrainingResults()
    epoch_runtimes = []

    for epoch in tqdm(range(epochs), desc="Epoch"):
        epoch_number = epoch + 1
        epoch_start_time = time.perf_counter()
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scheduler=scheduler,
            batch_normalizer=(train_batch_normalizer),
            batch_transform=train_batch_transform,
        )

        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device, batch_normalizer=(val_batch_normalizer)
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
            runtime_minutes=(epoch_runtime_minutes),
        )

        log_wandb_epoch(
            run=run,
            epoch=epoch_number,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            runtime_minutes=(epoch_runtime_minutes),
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
                logger=logger, epoch=epoch_number, val_acc=val_acc, val_loss=val_loss
            )

        if epoch_callback is not None:
            should_stop = epoch_callback(epoch_number, train_loss, train_acc, val_loss, val_acc)

            if should_stop:
                break

    results.set_runtime(epoch_runtimes)

    return results
