from pathlib import Path
import random
import numpy as np
import torch

def save_checkpoint(model, optimizer, epoch, path, scheduler=None, val_acc=None, total_epochs=None, steps_per_epoch=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "val_acc": val_acc,
        "total_epochs": total_epochs,
        "steps_per_epoch": steps_per_epoch,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, total_epochs=None, steps_per_epoch=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if total_epochs is not None and ckpt.get("total_epochs") not in {None, total_epochs}: raise ValueError(f"Checkpoint total_epochs={ckpt.get('total_epochs')} != current {total_epochs}")
    if steps_per_epoch is not None and ckpt.get("steps_per_epoch") not in {None, steps_per_epoch}: raise ValueError(f"Checkpoint steps_per_epoch={ckpt.get('steps_per_epoch')} != current {steps_per_epoch}")

    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None: optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None: scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if ckpt.get("python_rng_state") is not None: random.setstate(ckpt["python_rng_state"])
    if ckpt.get("numpy_rng_state") is not None: np.random.set_state(ckpt["numpy_rng_state"])
    if ckpt.get("torch_rng_state") is not None: torch.set_rng_state(ckpt["torch_rng_state"].cpu())
    if torch.cuda.is_available() and ckpt.get("cuda_rng_state") is not None: torch.cuda.set_rng_state_all([x.cpu() for x in ckpt["cuda_rng_state"]])
    return ckpt