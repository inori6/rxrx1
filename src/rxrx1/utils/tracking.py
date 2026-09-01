from pathlib import Path

import wandb

def setup_wandb(config: dict, project_root: Path):
    wandb_config = config.get("wandb", {})

    if not wandb_config.get("enabled", False):
        return None

    run = wandb.init(
        project=wandb_config["project"],
        name=config["experiment"]["name"],
        config=config,
        mode=wandb_config["mode"],
        dir=str(project_root / wandb_config["dir"]),
    )

    return run