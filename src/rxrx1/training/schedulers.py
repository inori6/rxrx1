import math

import torch


def _build_cosine_scheduler(
    optimizer,
    total_steps,
    warmup_ratio,
    min_lr_ratio,
):
    if not 0 <= warmup_ratio < 1:
        raise ValueError(
            "warmup_ratio must satisfy "
            "0 <= warmup_ratio < 1."
        )

    if not 0 <= min_lr_ratio <= 1:
        raise ValueError(
            "min_lr_ratio must satisfy "
            "0 <= min_lr_ratio <= 1."
        )

    warmup_steps = round(
        total_steps
        * warmup_ratio
    )

    def lr_lambda(step):
        if (
            warmup_steps > 0
            and step < warmup_steps
        ):
            return (
                step + 1
            ) / warmup_steps

        decay_steps = max(
            total_steps
            - warmup_steps,
            1,
        )

        progress = (
            step
            - warmup_steps
        ) / decay_steps

        progress = min(
            max(progress, 0.0),
            1.0,
        )

        cosine = 0.5 * (
            1.0
            + math.cos(
                math.pi
                * progress
            )
        )

        return (
            min_lr_ratio
            + (
                1.0
                - min_lr_ratio
            )
            * cosine
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )


def _build_onecycle_scheduler(
    optimizer,
    total_steps,
    pct_start,
    div_factor,
    final_div_factor,
):
    if not 0 < pct_start < 1:
        raise ValueError(
            "pct_start must satisfy "
            "0 < pct_start < 1."
        )

    max_lrs = [
        float(group["lr"])
        for group in optimizer.param_groups
    ]

    return (
        torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=(
                final_div_factor
            ),
            anneal_strategy="cos",
        )
    )


def build_scheduler(
    optimizer,
    config,
    epochs,
    steps_per_epoch,
):
    scheduler_config = (
        config.get("scheduler")
        or {}
    )

    scheduler_name = str(
        scheduler_config.get(
            "name",
            "none",
        )
    ).lower()

    if scheduler_name in {
        "none",
        "",
        "null",
    }:
        return None

    if epochs <= 0:
        raise ValueError(
            "epochs must be greater "
            "than zero."
        )

    if steps_per_epoch <= 0:
        raise ValueError(
            "steps_per_epoch must be "
            "greater than zero."
        )

    total_steps = (
        epochs
        * steps_per_epoch
    )

    if scheduler_name == "cosine":
        return _build_cosine_scheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_ratio=float(
                scheduler_config.get(
                    "warmup_ratio",
                    0.05,
                )
            ),
            min_lr_ratio=float(
                scheduler_config.get(
                    "min_lr_ratio",
                    0.01,
                )
            ),
        )

    if scheduler_name == "onecycle":
        return _build_onecycle_scheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            pct_start=float(
                scheduler_config.get(
                    "pct_start",
                    0.1,
                )
            ),
            div_factor=float(
                scheduler_config.get(
                    "div_factor",
                    10.0,
                )
            ),
            final_div_factor=float(
                scheduler_config.get(
                    "final_div_factor",
                    1000.0,
                )
            ),
        )

    raise ValueError(
        "Unsupported scheduler: "
        f"{scheduler_name}"
    )