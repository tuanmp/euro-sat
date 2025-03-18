from datetime import datetime
from pytorch_lightning import Trainer
import os
from transform.vit_processor import get_image_processor
import datasets as ds
import click
import yaml
from models.vit import LightningViTModel
import torch
from functools import partial

cache_dir = os.environ.get("PSCRATCH", "/tmp")

def preprocess(example, image_processor):
    image = example["image"]
    image = image_processor(image, return_tensors="pt")["pixel_values"].squeeze(0)
    example["pixel_values"] = image
    del example["image"]
    return example

@click.command()
@click.option("--config", default="configs/lightning_default.yaml", help="Path to the config file.")
@click.option("--fast_dev_run", default=False, help="Run a fast dev run.")
def train(config, fast_dev_run):
    
    # load config file
    with open(config, "r") as f:
        config = yaml.safe_load(f)
    
    # load model
    model = LightningViTModel(
        pretrained_model_name=config["model"]["pretrained_model_name"],
        dataset_name=config["dataset"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        hidden_layers=config["model"]["hidden_layers"],
        hidden_dim=config["model"]["hidden_dim"],
        batch_norm=config["model"]["batch_norm"],
        layer_norm=config["model"]["layer_norm"],
        hidden_activation=config["model"]["hidden_activation"],
        output_activation=config["model"]["output_activation"],
        learning_rate=config["training"]["learning_rate"],
        freeze_backbone=config["training"]["freeze_backbone"],
        lr_scheduler_config=config["training"].get("lr_scheduler_config", {})
    )
    # load data
    # datasets = ds.load_dataset(config["dataset"], cache_dir=cache_dir)
    train_dataset = ds.load_dataset(config["dataset"], split='train', cache_dir=cache_dir)
    val_dataset = ds.load_dataset(config["dataset"], split='validation', cache_dir=cache_dir)
    # load image processor
    image_processor = get_image_processor(
        config["model"]["pretrained_model_name"],
        cache_dir=cache_dir,
        use_fast=True
    )

    train_dataset = train_dataset.with_transform(
        partial(preprocess, image_processor=image_processor)
    )
    val_dataset = val_dataset.with_transform(partial(preprocess, image_processor=image_processor))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True
    )
    
    model_tag = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    print("Configuring Trainer ...")
    save_dir = os.path.join(config["training"].get("save_dir", "./checkpoints"))

    trainer = Trainer(
        accelerator = config["training"].get("accelerator", "auto"),
        strategy = config["training"].get("strategy", "auto"),
        devices = config["training"].get("devices", "auto"),
        num_nodes = config["training"].get("num_nodes", 1),
        precision = config["training"].get("precision", 32),
        fast_dev_run = fast_dev_run,
        max_epochs = config["training"]["num_epochs"],
        enable_model_summary=True,
        default_root_dir = save_dir
    )


    print("Begin training...")
    trainer.fit(model, [train_dataloader], [val_dataloader])
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, model_tag + ".pt"))


if __name__=="__main__":
    train()