import torchmetrics.classification
from transformers import ViTModel as TransformersViTModel
import torch
from .mlp import ClassifierHead
from .lightning_base import BaseLightningModel
import os
from pytorch_lightning import LightningModule
from transformers import AutoImageProcessor
from datasets import load_dataset, load_dataset_builder
import torchmetrics
cache_dir = os.environ.get("PSCRATCH", "/tmp")


class ViTModel(torch.nn.Module):

    def __init__(
        self,
        pretrained_model_name: str,
        num_classes: int,
        dropout: float = 0.1,
        hidden_layers: int = 0,
        hidden_dim: int = 512,
        batch_norm: bool = False,
        layer_norm: bool = False,
        hidden_activation: str = "relu",
        output_activation=None,
    ):

        super().__init__()
        self.backbone = TransformersViTModel.from_pretrained(
            pretrained_model_name, cache_dir=cache_dir
        )
        self.classifier = ClassifierHead(
            input_dim=self.backbone.config.hidden_size,
            num_classes=num_classes,
            dropout=dropout,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
        )

    def forward(self, x):
        last_hidden_state = self.backbone(x).last_hidden_state[:, 0, :]
        logits = self.classifier(last_hidden_state)
        return logits


class LightningViTModel(BaseLightningModel):
    def __init__(
        self,
        pretrained_model_name: str,
        dataset_name: str,
        num_classes: int,
        dropout: float = 0.1,
        hidden_layers: int = 0,
        hidden_dim: int = 512,
        batch_norm: bool = False,
        layer_norm: bool = False,
        hidden_activation: str = "relu",
        output_activation=None,
        freeze_backbone: bool = True,
        compile: bool = True,
        learning_rate = 1e-4,
        lr_scheduler_config: dict = {}
    ):

        super().__init__()
        self.backbone = TransformersViTModel.from_pretrained(
            pretrained_model_name, cache_dir=cache_dir
        )
        self.num_classes = num_classes
        self.pretrained_model_name = pretrained_model_name
        self.dataset_name = dataset_name
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.learning_rate = learning_rate
        self.lr_scheduler_config = lr_scheduler_config
        self.classifier = ClassifierHead(
            input_dim=self.backbone.config.hidden_size,
            num_classes=num_classes,
            dropout=dropout,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
        )

        self.vit_processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name, cache_dir=cache_dir, use_fast=True
        )
        self.configure_metrics()
        if compile:
            self.backbone = torch.compile(self.backbone)
            self.classifier = torch.compile(self.classifier)
    
    def configure_metrics(self):
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.num_classes, average="macro"),
                "precision": torchmetrics.classification.Precision(task="multiclass", num_classes=self.num_classes, average="macro"),
                "recall": torchmetrics.classification.Recall(task="multiclass", num_classes=self.num_classes, average="macro"),
                "f1": torchmetrics.classification.F1Score(task="multiclass", num_classes=self.num_classes, average="macro"),
            }, 
            prefix="train_"
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")

    def setup(self, stage='fit'):
        builder = load_dataset_builder(self.dataset_name, cache_dir=cache_dir)
        builder.download_and_prepare()

    def forward(self, x):
        last_hidden_state = self.backbone(x).last_hidden_state[:, 0, :]
        logits = self.classifier(last_hidden_state)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.classifier.parameters() if self.freeze_backbone else self.parameters(), lr=self.learning_rate)
        if self.lr_scheduler_config:
            warm_up_scheduler = None
            warm_up_steps = self.lr_scheduler_config.pop('warmup_steps') if 'warmup_steps' in self.lr_scheduler_config else 0
            if warm_up_steps:
                warm_up_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x : float(min ( x / warm_up_steps, 1.0)))
            
            scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_config.pop("lr_scheduler"))
            scheduler = scheduler(optimizer, **self.lr_scheduler_config)
            if warm_up_scheduler:
                scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warm_up_scheduler, scheduler], milestones=[warm_up_steps])
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return [optimizer]

    def training_step(self, batch, batch_idx):
        pixel_values = batch[0]["pixel_values"]
        labels = batch[0]['label']
        logits = self(pixel_values)
        preds = logits.argmax(dim=-1)
        self.train_metrics.update(preds, labels)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        self.log_single_metric("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch['label']
        logits = self(pixel_values)
        preds = logits.argmax(dim=-1)
        self.val_metrics.update(preds, labels)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        self.log_single_metric("val_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log_single_dict(self.train_metrics.compute())
        self.train_metrics.reset()
    
    def on_validation_epoch_end(self):
        self.log_single_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()
