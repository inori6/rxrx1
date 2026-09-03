from dataclasses import dataclass


@dataclass
class TrainingResults:
    final_train_loss: float | None = None
    final_train_acc: float | None = None
    final_val_loss: float | None = None
    final_val_acc: float | None = None

    best_epoch: int = 0
    best_train_loss: float | None = None
    best_train_acc: float | None = None
    best_val_loss: float | None = None
    best_val_acc: float = float("-inf")

    runtime_per_epoch_minutes: float | None = None


    def update_epoch(
        self,
        epoch,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
    ):
        # -----------------------
        # Final result
        # -----------------------

        self.final_train_loss = train_loss
        self.final_train_acc = train_acc
        self.final_val_loss = val_loss
        self.final_val_acc = val_acc

        # -----------------------
        # Best result
        # -----------------------

        if val_acc <= self.best_val_acc:
            return False

        self.best_epoch = epoch

        self.best_train_loss = train_loss
        self.best_train_acc = train_acc

        self.best_val_loss = val_loss
        self.best_val_acc = val_acc

        return True


    def set_runtime(self, epoch_runtimes):
        if not epoch_runtimes:
            raise ValueError(
                "Cannot calculate runtime: no epochs were executed."
            )

        self.runtime_per_epoch_minutes = (
            sum(epoch_runtimes)
            / len(epoch_runtimes)
            / 60
        )