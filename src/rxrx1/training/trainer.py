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

def _prepare_mixed_metadata(metadata, mix_info):
    if mix_info is None:
        return metadata

    return {
        **metadata,
        "_mix_permutation": mix_info["permutation"],
        "_mix_lam": mix_info["lam"],
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
    arcface_enabled = getattr(criterion, "arcface_enabled", False)

    if metric_enabled and arcface_enabled:
        raise ValueError("Hierarchical metric loss and first-place ArcFace cannot be enabled together.")

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

        mixed_images = images
        targets = labels
        mixed_metadata = metadata

        if batch_transform is not None:
            mixed_images, targets, mix_info = batch_transform(images, labels)
            mixed_metadata = _prepare_mixed_metadata(metadata, mix_info)

        optimizer.zero_grad()

        if metric_enabled and batch_transform is not None:
            _, embeddings = model(images, metadata, return_embeddings=True)
            metric_loss = criterion.metric_loss(embeddings, labels, batch)
            weighted_metric_loss = criterion.lambda_metric * metric_loss
            weighted_metric_loss.backward()

            outputs = model(mixed_images, mixed_metadata)
            classification_loss = criterion.classification_loss(outputs, targets)
            classification_loss.backward()

            loss = classification_loss.detach() + weighted_metric_loss.detach()

        elif metric_enabled:
            outputs, embeddings = model(images, metadata, return_embeddings=True)
            loss = criterion(outputs, labels, embeddings, batch)
            loss.backward()

        elif arcface_enabled:
            outputs, arc_logits = model(mixed_images, mixed_metadata, return_arc_logits=True)
            loss = criterion(outputs, targets, arc_logits)
            loss.backward()

        else:
            outputs = model(mixed_images, mixed_metadata)
            loss = criterion(outputs, targets)
            loss.backward()

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)

        if targets.ndim == 2:
            total_correct += targets.gather(1, predictions[:, None]).sum().item()
        else:
            total_correct += (predictions == targets).sum().item()

        total_samples += batch_size
        pbar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
        )

    return total_loss / total_samples, total_correct / total_samples

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
    model, train_loader, val_loader, optimizer, criterion, device, epochs,
    checkpoint_enabled, checkpoint_path, logger, run, scheduler=None,
    train_batch_normalizer=None, val_batch_normalizer=None, epoch_callback=None,
    train_batch_transform=None, last_checkpoint_path=None, start_epoch=0,
    stop_after_epoch=None, train_acc_checkpoint_config=None,
):
    if epochs <= 0: raise ValueError(f"epochs must be greater than 0, got {epochs}")
    if start_epoch < 0 or start_epoch >= epochs: raise ValueError(f"Invalid start_epoch={start_epoch} for epochs={epochs}")
    if stop_after_epoch is not None and not start_epoch < stop_after_epoch <= epochs:
        raise ValueError(f"Invalid stop_after_epoch={stop_after_epoch} for start_epoch={start_epoch}, epochs={epochs}")

    results = TrainingResults()
    epoch_runtimes = []
    validation_enabled = val_loader is not None
    acc_cfg = train_acc_checkpoint_config or {}
    save_by_acc = bool(acc_cfg.get("enabled", False))
    acc_threshold = float(acc_cfg.get("threshold", 1.0))

    for epoch in tqdm(range(start_epoch, epochs), desc="Epoch"):
        epoch_number = epoch + 1
        start = time.perf_counter()

        if hasattr(train_loader.batch_sampler, "set_epoch"): train_loader.batch_sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler, batch_normalizer=train_batch_normalizer,
            batch_transform=train_batch_transform,
        )

        if validation_enabled:
            val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device, batch_normalizer=val_batch_normalizer)
        else:
            val_loss, val_acc = None, None

        runtime = time.perf_counter() - start
        epoch_runtimes.append(runtime)
        is_best = results.update_epoch(epoch_number, train_loss, train_acc, val_loss, val_acc)

        log_epoch_result(logger, epoch_number, train_loss, train_acc, val_loss, val_acc, runtime / 60)
        log_wandb_epoch(run, epoch_number, train_loss, train_acc, val_loss, val_acc, runtime / 60)

        if checkpoint_enabled and last_checkpoint_path is not None:
            save_checkpoint(
                model, optimizer, epoch_number, last_checkpoint_path,
                scheduler=scheduler, val_acc=val_acc,
                total_epochs=epochs, steps_per_epoch=len(train_loader),
            )

        if checkpoint_enabled and save_by_acc and train_acc >= acc_threshold and last_checkpoint_path is not None:
            snapshot = last_checkpoint_path.parent / "train_acc_snapshots" / f"epoch_{epoch_number:02d}_trainacc_{train_acc:.4f}.pt"
            save_checkpoint(
                model, optimizer, epoch_number, snapshot,
                scheduler=scheduler, val_acc=val_acc,
                total_epochs=epochs, steps_per_epoch=len(train_loader),
            )
            logger.info("Train-acc checkpoint saved | epoch=%d | train_acc=%.4f | threshold=%.4f | path=%s", epoch_number, train_acc, acc_threshold, snapshot)

        if validation_enabled and is_best and checkpoint_enabled:
            save_checkpoint(
                model, optimizer, epoch_number, checkpoint_path,
                scheduler=scheduler, val_acc=val_acc,
                total_epochs=epochs, steps_per_epoch=len(train_loader),
            )
            log_best_checkpoint(logger, epoch_number, val_acc, val_loss)

        if epoch_callback is not None and epoch_callback(epoch_number, train_loss, train_acc, val_loss, val_acc): break

        if stop_after_epoch is not None and epoch_number >= stop_after_epoch:
            logger.info("Training paused | completed_epoch=%d | total_epochs=%d", epoch_number, epochs)
            break

    results.set_runtime(epoch_runtimes)
    return results