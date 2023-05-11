import itertools
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from sklearn.metrics import roc_auc_score
from torch import nn
from efficientad.models import get_pdn_small, get_pdn_medium, get_autoencoder
from typing import Optional, Sequence, Tuple, List, Any, Dict, Union
from torch.optim import Optimizer
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import logging
from efficientad.utils import map_normalization, predict, teacher_normalization

log = logging.getLogger(__name__)


class TeacherNormalizationCallback(Callback):
    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: "EfficientAd") -> None:
        """Called on every stage."""
        if not hasattr(trainer, "datamodule") or not hasattr(trainer, "log_every_n_steps"):
            raise ValueError("Trainer must have a datamodule and log_every_n_steps attribute.")

        pl_module.teacher_mean, pl_module.teacher_std = teacher_normalization(
            pl_module.teacher, train_loader=trainer.datamodule.train_dataloader(), device=pl_module.device
        )


class MapNormalizationCallback(Callback):
    @rank_zero_only
    def on_validation_start(self, trainer: pl.Trainer, pl_module: "EfficientAd") -> None:
        """Called on every stage."""
        if not hasattr(trainer, "datamodule") or not hasattr(trainer, "log_every_n_steps"):
            raise ValueError("Trainer must have a datamodule and log_every_n_steps attribute.")

        pl_module.q_st_start, pl_module.q_st_end, pl_module.q_ae_start, pl_module.q_ae_end = map_normalization(
            validation_loader=trainer.datamodule.map_normalization_dataloader(),
            teacher=pl_module.teacher,
            student=pl_module.student,
            autoencoder=pl_module.autoencoder,
            teacher_mean=pl_module.teacher_mean,
            teacher_std=pl_module.teacher_std,
            desc="Intermediate map normalization",
            out_channels=pl_module.out_channels,
            device=pl_module.device,
        )


class EfficientAd(LightningModule):
    def __init__(
        self,
        teacher_pretrained_weights: str,
        model_size: str = "small",
        out_channels: int = 384,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[object] = None,
        lr_scheduler_interval: Optional[str] = "step",
        max_steps: int = 70000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        if model_size == "small":
            self.teacher = get_pdn_small(out_channels=self.out_channels)
            self.student = get_pdn_small(out_channels=self.out_channels * 2)
        elif model_size == "medium":
            self.teacher = get_pdn_medium(out_channels=self.out_channels)
            self.student = get_pdn_medium(out_channels=self.out_channels * 2)
        else:
            raise ValueError(f"Unknown model size: {model_size}")

        state_dict = torch.load(teacher_pretrained_weights, map_location=self.device)
        self.teacher.load_state_dict(state_dict)
        self.teacher.eval()

        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.autoencoder = get_autoencoder(out_channels=self.out_channels)

        self.optimizer = optimizer
        self.schedulers = lr_scheduler
        self.lr_scheduler_interval = lr_scheduler_interval
        self.max_steps = max_steps
        self.teacher_mean: torch.Tensor
        self.teacher_std: torch.Tensor

    def configure_optimizers(self) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """Get default optimizer if not passed a value.

        Returns:
            optimizer and lr scheduler as Tuple containing a list of optimizers and a list of lr schedulers
        """
        # get default optimizer
        if getattr(self, "optimizer", None) is None or not self.optimizer:
            self.optimizer = optimizer = torch.optim.Adam(
                itertools.chain(self.student.parameters(), self.autoencoder.parameters()), lr=1e-4, weight_decay=1e-5
            )

        # get default scheduler
        if getattr(self, "schedulers", None) is None or not self.schedulers:
            self.schedulers = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=int(0.95 * self.max_steps), gamma=0.1
            )

        lr_scheduler_conf = {
            "scheduler": self.schedulers,
            "interval": self.lr_scheduler_interval,
            "monitor": "train_loss_epoch",
            "strict": False,
        }
        return [self.optimizer], [lr_scheduler_conf]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        """Redefine optimizer zero grad."""
        optimizer.zero_grad(set_to_none=True)

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        return [MapNormalizationCallback(), TeacherNormalizationCallback()]

    def training_step(self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], batch_idx: int):
        (image_st, image_ae), image_penalty = batch
        self.teacher.eval()
        log.info("Training")
        with torch.no_grad():
            teacher_output_st = self.teacher(image_st)
            teacher_output_st = (teacher_output_st - self.teacher_mean) / self.teacher_std

        student_output_st = self.student(image_st)[:, : self.out_channels]
        distance_st = (teacher_output_st - student_output_st) ** 2
        d_hard = torch.quantile(distance_st, q=0.999)
        loss_hard = torch.mean(distance_st[distance_st >= d_hard])

        if image_penalty is not None:
            student_output_penalty = self.student(image_penalty)[:, : self.out_channels]
            loss_penalty = torch.mean(student_output_penalty**2)
            loss_st = loss_hard + loss_penalty
        else:
            loss_st = loss_hard

        ae_output = self.autoencoder(image_ae)

        with torch.no_grad():
            teacher_output_ae = self.teacher(image_ae)
            teacher_output_ae = (teacher_output_ae - self.teacher_mean) / self.teacher_std

        student_output_ae = self.student(image_ae)[:, self.out_channels :]
        distance_ae = torch.pow(teacher_output_ae - ae_output, 2)
        distance_stae = torch.pow(ae_output - student_output_ae, 2)
        loss_ae = torch.mean(distance_ae)
        loss_stae = torch.mean(distance_stae)
        loss_total = loss_st + loss_ae + loss_stae

        self.log("train_loss", loss_total, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_st", loss_st, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_ae", loss_ae, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_stae", loss_stae, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss_total

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx):
        image, target, _ = batch

        map_combined, _, _ = predict(
            image=image,
            teacher=self.teacher,
            student=self.student,
            autoencoder=self.autoencoder,
            teacher_mean=self.teacher_mean,
            teacher_std=self.teacher_std,
            q_st_start=self.q_st_start,
            q_st_end=self.q_st_end,
            q_ae_start=self.q_ae_start,
            q_ae_end=self.q_ae_end,
            out_channels=self.out_channels,
        )

        return {"y_score": torch.max(map_combined), "y_true": target}

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        y_score = torch.stack([x["y_score"] for x in outputs])
        y_true = torch.stack([x["y_true"] for x in outputs])

        auc = roc_auc_score(y_true=y_true.cpu().numpy(), y_score=y_score.cpu().numpy())
        self.log("val_auc", auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
