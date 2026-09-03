import logging
from pathlib import Path

def setup_logger(name, log_file, log_level=logging.INFO):
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def log_label_revised(logger, train_labels, original_val_labels,val_labels):
    logger.info(
        "Label check passed | train_labels=%d | "
        "original_val_labels=%d | val_labels=%d | removed_val_labels=%d",
        len(train_labels),
        len(original_val_labels),
        len(val_labels),
        len(original_val_labels - val_labels),
    )


def log_training_started(
    logger,
    device,
    train_samples,
    val_samples,
):
    logger.info(
        "Training started | "
        "device=%s | "
        "train_samples=%d | "
        "val_samples=%d",
        device,
        train_samples,
        val_samples,
    )


def log_epoch_result(
    logger,
    epoch,
    train_loss,
    train_acc,
    val_loss,
    val_acc,
    runtime_minutes,
):
    logger.info(
        "Epoch %d | "
        "train_loss=%.4f | "
        "train_acc=%.4f | "
        "val_loss=%.4f | "
        "val_acc=%.4f | "
        "runtime=%.2f min",
        epoch,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
        runtime_minutes,
    )


def log_best_checkpoint(
    logger,
    epoch,
    val_acc,
    val_loss,
):
    logger.info(
        "Best checkpoint saved | "
        "epoch=%d | "
        "val_acc=%.4f | "
        "val_loss=%.4f",
        epoch,
        val_acc,
        val_loss,
    )


def log_training_finished(
    logger,
    results,
):
    logger.info(
        "Training finished | "
        "final_train_acc=%.4f | "
        "final_train_loss=%.4f | "
        "final_val_acc=%.4f | "
        "final_val_loss=%.4f | "
        "best_epoch=%d | "
        "best_train_acc=%.4f | "
        "best_train_loss=%.4f | "
        "best_val_acc=%.4f | "
        "best_val_loss=%.4f | "
        "runtime_per_epoch_minutes=%.2f",
        results.final_train_acc,
        results.final_train_loss,
        results.final_val_acc,
        results.final_val_loss,
        results.best_epoch,
        results.best_train_acc,
        results.best_train_loss,
        results.best_val_acc,
        results.best_val_loss,
        results.runtime_per_epoch_minutes,
    )


def log_training_failed(
    logger,
    error,
):
    logger.exception(
        "Training failed | error=%s",
        error,
    )