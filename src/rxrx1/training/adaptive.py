import math


_STATE_CODE = {
    "DECAY": 0.0,
    "HOLD": 1.0,
    "CONFIRM": 2.0,
    "STOP": 3.0,
}


class AdaptiveContinuationScheduler:
    def __init__(
        self,
        optimizer,
        total_epochs,
        steps_per_epoch,
        min_lr_ratio=0.01,
    ):
        if total_epochs <= 0:
            raise ValueError("total_epochs must be greater than zero.")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be greater than zero.")
        if not 0 <= min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must satisfy 0 <= value <= 1.")

        self.optimizer = optimizer
        self.total_epochs = int(total_epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [
            float(group.get("initial_lr", group["lr"]))
            for group in optimizer.param_groups
        ]
        self.mode = "decay"
        self.current_epoch = 0
        self.phase_increment = 0.0
        self._step_count = 0
        self.phase = self._infer_phase()
        self._last_lr = [float(group["lr"]) for group in optimizer.param_groups]

    def _infer_phase(self):
        multipliers = []
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            if base_lr > 0:
                multipliers.append(float(group["lr"]) / base_lr)

        multiplier = sum(multipliers) / len(multipliers) if multipliers else 1.0
        if self.min_lr_ratio >= 1.0:
            return 1.0

        x = (multiplier - self.min_lr_ratio) / (1.0 - self.min_lr_ratio)
        x = min(max(x, 0.0), 1.0)
        cos_value = min(max(2.0 * x - 1.0, -1.0), 1.0)
        return math.acos(cos_value) / math.pi

    def begin_epoch(self, epoch_number, mode):
        mode = str(mode).lower()
        if mode not in {"hold", "decay"}:
            raise ValueError(f"Unsupported adaptive scheduler mode: {mode}")

        self.current_epoch = int(epoch_number)
        self.mode = mode

        if mode == "hold":
            self.phase_increment = 0.0
            return

        remaining_epochs = self.total_epochs - self.current_epoch + 1
        remaining_steps = max(remaining_epochs * self.steps_per_epoch, 1)
        self.phase_increment = max(1.0 - self.phase, 0.0) / remaining_steps

    def step(self):
        self._step_count += 1
        if self.mode == "hold":
            self._last_lr = [float(group["lr"]) for group in self.optimizer.param_groups]
            return

        self.phase = min(1.0, self.phase + self.phase_increment)
        cosine = 0.5 * (1.0 + math.cos(math.pi * self.phase))
        multiplier = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        self._last_lr = []
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            lr = base_lr * multiplier
            group["lr"] = lr
            self._last_lr.append(lr)

    def get_last_lr(self):
        return list(self._last_lr)

    def state_dict(self):
        return {
            "type": "adaptive_continuation_cosine",
            "total_epochs": self.total_epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": list(self.base_lrs),
            "phase": self.phase,
            "phase_increment": self.phase_increment,
            "mode": self.mode,
            "current_epoch": self.current_epoch,
            "_step_count": self._step_count,
            "_last_lr": list(self._last_lr),
        }

    def load_state_dict(self, state):
        if state.get("type") != "adaptive_continuation_cosine":
            raise ValueError("Checkpoint scheduler is not an adaptive continuation scheduler.")
        if int(state["total_epochs"]) != self.total_epochs:
            raise ValueError(
                f"Adaptive scheduler total_epochs={state['total_epochs']} != current {self.total_epochs}"
            )
        if int(state["steps_per_epoch"]) != self.steps_per_epoch:
            raise ValueError(
                f"Adaptive scheduler steps_per_epoch={state['steps_per_epoch']} != current {self.steps_per_epoch}"
            )

        self.min_lr_ratio = float(state["min_lr_ratio"])
        self.base_lrs = [float(x) for x in state["base_lrs"]]
        self.phase = float(state["phase"])
        self.phase_increment = float(state.get("phase_increment", 0.0))
        self.mode = str(state.get("mode", "decay"))
        self.current_epoch = int(state.get("current_epoch", 0))
        self._step_count = int(state.get("_step_count", 0))
        self._last_lr = [float(x) for x in state["_last_lr"]]

        if len(self._last_lr) != len(self.optimizer.param_groups):
            raise ValueError("Adaptive scheduler LR group count mismatch.")

        for lr, group in zip(self._last_lr, self.optimizer.param_groups):
            group["lr"] = lr


class AdaptiveTrainingController:
    def __init__(self, config, max_epochs):
        self.window_epochs = int(config.get("window_epochs", 5))
        self.speed_ratio_threshold = float(config.get("speed_ratio_threshold", 0.60))
        self.acc_plateau_delta = float(config.get("acc_plateau_delta", 0.005))
        self.loss_plateau_rel = float(config.get("loss_plateau_rel", 0.005))
        self.confirmation_epochs = int(config.get("confirmation_epochs", 5))
        self.force_decay_last_epochs = int(config.get("force_decay_last_epochs", 10))
        self.max_epochs = int(max_epochs)

        if self.window_epochs < 2:
            raise ValueError("adaptive.window_epochs must be at least 2.")
        if self.confirmation_epochs < 1:
            raise ValueError("adaptive.confirmation_epochs must be at least 1.")

        self.history = []
        self.state = "DECAY"
        self.next_mode = "decay"
        self.confirm_remaining = 0
        self.confirm_start_acc = None
        self.confirm_start_loss = None
        self.should_stop = False

    @staticmethod
    def _loss_drop_rel(start, end):
        return (float(start) - float(end)) / max(abs(float(start)), 1.0e-12)

    @staticmethod
    def _speed_ratio(current, previous):
        eps = 1.0e-12
        if previous > eps:
            return current / previous
        return float("inf") if current > eps else 0.0

    def _window_metrics(self):
        w = self.window_epochs
        previous = self.history[-2 * w:-w]
        current = self.history[-w:]

        prev_acc_gain = previous[-1]["acc"] - previous[0]["acc"]
        curr_acc_gain = current[-1]["acc"] - current[0]["acc"]
        prev_loss_drop = self._loss_drop_rel(previous[0]["loss"], previous[-1]["loss"])
        curr_loss_drop = self._loss_drop_rel(current[0]["loss"], current[-1]["loss"])

        return {
            "acc_gain": curr_acc_gain,
            "loss_drop_rel": curr_loss_drop,
            "acc_speed_ratio": self._speed_ratio(curr_acc_gain, prev_acc_gain),
            "loss_speed_ratio": self._speed_ratio(curr_loss_drop, prev_loss_drop),
        }

    def update(self, epoch, train_acc, train_loss):
        epoch = int(epoch)
        train_acc = float(train_acc)
        train_loss = float(train_loss)
        self.history.append({"epoch": epoch, "acc": train_acc, "loss": train_loss})

        metrics = {
            "adaptive/state_code": _STATE_CODE[self.state],
            "adaptive/confirm_remaining": float(self.confirm_remaining),
            "adaptive/should_stop": float(self.should_stop),
        }

        force_decay_start = self.max_epochs - self.force_decay_last_epochs + 1
        if epoch >= force_decay_start:
            self.state = "DECAY"
            self.next_mode = "decay"
            self.confirm_remaining = 0
            self.confirm_start_acc = None
            self.confirm_start_loss = None
            self.should_stop = False
            metrics.update({
                "adaptive/state_code": _STATE_CODE[self.state],
                "adaptive/confirm_remaining": 0.0,
                "adaptive/should_stop": 0.0,
            })
            return metrics

        if self.state == "CONFIRM":
            self.confirm_remaining -= 1
            acc_gain = train_acc - self.confirm_start_acc
            loss_drop = self._loss_drop_rel(self.confirm_start_loss, train_loss)
            metrics["adaptive/confirm_acc_gain"] = acc_gain
            metrics["adaptive/confirm_loss_drop_rel"] = loss_drop

            recovered = (
                acc_gain >= self.acc_plateau_delta
                or loss_drop >= self.loss_plateau_rel
            )

            if recovered:
                self.state = "DECAY"
                self.next_mode = "decay"
                self.confirm_remaining = 0
                self.confirm_start_acc = None
                self.confirm_start_loss = None
            elif self.confirm_remaining <= 0:
                self.state = "STOP"
                self.next_mode = "hold"
                self.should_stop = True
            else:
                self.next_mode = "hold"

            metrics.update({
                "adaptive/state_code": _STATE_CODE[self.state],
                "adaptive/confirm_remaining": float(self.confirm_remaining),
                "adaptive/should_stop": float(self.should_stop),
            })
            return metrics

        if len(self.history) < 2 * self.window_epochs:
            self.state = "DECAY"
            self.next_mode = "decay"
            metrics["adaptive/state_code"] = _STATE_CODE[self.state]
            return metrics

        window = self._window_metrics()
        metrics.update({
            "adaptive/acc_gain": window["acc_gain"],
            "adaptive/loss_drop_rel": window["loss_drop_rel"],
            "adaptive/acc_speed_ratio": window["acc_speed_ratio"],
            "adaptive/loss_speed_ratio": window["loss_speed_ratio"],
        })

        plateau = (
            window["acc_gain"] < self.acc_plateau_delta
            and window["loss_drop_rel"] < self.loss_plateau_rel
        )

        if plateau:
            remaining = max(self.max_epochs - epoch, 0)
            if remaining == 0:
                self.state = "STOP"
                self.next_mode = "hold"
                self.should_stop = True
            else:
                self.state = "CONFIRM"
                self.next_mode = "hold"
                self.confirm_remaining = min(self.confirmation_epochs, remaining)
                self.confirm_start_acc = train_acc
                self.confirm_start_loss = train_loss
        elif (
            window["acc_speed_ratio"] >= self.speed_ratio_threshold
            and window["loss_speed_ratio"] >= self.speed_ratio_threshold
        ):
            self.state = "HOLD"
            self.next_mode = "hold"
        else:
            self.state = "DECAY"
            self.next_mode = "decay"

        metrics.update({
            "adaptive/state_code": _STATE_CODE[self.state],
            "adaptive/confirm_remaining": float(self.confirm_remaining),
            "adaptive/should_stop": float(self.should_stop),
        })
        return metrics

    def state_dict(self):
        return {
            "window_epochs": self.window_epochs,
            "speed_ratio_threshold": self.speed_ratio_threshold,
            "acc_plateau_delta": self.acc_plateau_delta,
            "loss_plateau_rel": self.loss_plateau_rel,
            "confirmation_epochs": self.confirmation_epochs,
            "force_decay_last_epochs": self.force_decay_last_epochs,
            "max_epochs": self.max_epochs,
            "history": list(self.history),
            "state": self.state,
            "next_mode": self.next_mode,
            "confirm_remaining": self.confirm_remaining,
            "confirm_start_acc": self.confirm_start_acc,
            "confirm_start_loss": self.confirm_start_loss,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state):
        if int(state.get("max_epochs", self.max_epochs)) != self.max_epochs:
            raise ValueError(
                f"Adaptive controller max_epochs={state.get('max_epochs')} != current {self.max_epochs}"
            )
        self.history = list(state.get("history", []))
        self.state = str(state.get("state", "DECAY"))
        self.next_mode = str(state.get("next_mode", "decay"))
        self.confirm_remaining = int(state.get("confirm_remaining", 0))
        self.confirm_start_acc = state.get("confirm_start_acc")
        self.confirm_start_loss = state.get("confirm_start_loss")
        self.should_stop = bool(state.get("should_stop", False))


def snapshot_optimizer_parameters(optimizer):
    snapshots = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("group_name", f"group_{index}"))
        snapshots[name] = [
            parameter.detach().cpu().clone()
            for parameter in group["params"]
            if parameter.requires_grad
        ]
    return snapshots


def calculate_group_update_metrics(optimizer, snapshots, start_lrs):
    metrics = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("group_name", f"group_{index}"))
        before = snapshots[name]
        parameters = [p for p in group["params"] if p.requires_grad]

        delta_sq = 0.0
        parameter_sq = 0.0
        for parameter, previous in zip(parameters, before):
            current = parameter.detach().float().cpu()
            previous = previous.float()
            delta_sq += (current - previous).pow(2).sum().item()
            parameter_sq += previous.pow(2).sum().item()

        update_ratio = math.sqrt(delta_sq) / max(math.sqrt(parameter_sq), 1.0e-12)
        end_lr = float(group["lr"])
        mean_lr = 0.5 * (float(start_lrs[name]) + end_lr)
        normalized = update_ratio / max(mean_lr, 1.0e-16)

        metrics[f"lr/{name}"] = end_lr
        metrics[f"update_ratio/{name}"] = update_ratio
        metrics[f"normalized_update/{name}"] = normalized

    return metrics
