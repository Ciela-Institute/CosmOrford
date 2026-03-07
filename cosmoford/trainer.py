import time
import datetime
import torch
from lightning.pytorch.cli import ArgsType, LightningCLI
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import WeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn

torch.set_float32_matmul_precision('medium')


class EpochProgressPrinter(Callback):
    """Prints a clean one-line summary per epoch to stdout (for SLURM log files).

    Replaces the tqdm progress bar, which uses carriage-return overwrites that
    make log files unreadable.  Example output::

        Epoch   1/30 | 0:00:52 | train_loss: 0.1234 | val_score: 9.853 | val_mse: 0.0012 | lr: 1.234e-03
    """

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.monotonic()

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        elapsed = datetime.timedelta(seconds=int(time.monotonic() - self._epoch_start))
        epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs or "?"

        parts = [f"Epoch {epoch:>3}/{max_epochs} | {elapsed}"]

        metrics = trainer.callback_metrics
        for key in ["train_loss", "val_score", "val_mse"]:
            if key in metrics:
                parts.append(f"{key}: {metrics[key].item():.4f}")

        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0]["lr"]
            parts.append(f"lr: {lr:.3e}")

        print(" | ".join(parts), flush=True)


class EMAWeightAveraging(WeightAveraging):
    def __init__(self):
        super().__init__(avg_fn=get_ema_avg_fn(decay=0.99))

    def should_update(self, step_idx=None, epoch_idx=None):
        # Start after 1000 steps.
        return (step_idx is not None) and (step_idx >= 100)

class CustomSaveConfigCallback(SaveConfigCallback):
    """Saves full training configuration
    Otherwise wandb won't log full configuration but only flattened module and data hyperparameters
    """

    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        for logger in trainer.loggers:
            if issubclass(type(logger), WandbLogger):
                logger.experiment.config.update(self.config.as_dict())
        return super().save_config(trainer, pl_module, stage)

def trainer_cli(args: ArgsType = None, run: bool = True):
    return LightningCLI(
        args=args,
        run=run,
        save_config_kwargs={"overwrite": True},
        save_config_callback=CustomSaveConfigCallback,
        parser_kwargs={"parser_mode": "omegaconf"},
    )

if __name__ == "__main__":
    trainer_cli(run=True)
