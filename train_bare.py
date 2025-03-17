import torch 
import yaml
from models.vit import ViTModel
import click
import datasets as ds
import os
from transformers import AutoImageProcessor
from functools import partial
from tqdm import tqdm
from datetime import datetime

cache_dir = os.environ.get("PSCRATCH", '/tmp')
device = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess(example, image_processor):
    image = example["image"]
    image = image_processor(image, return_tensors="pt")["pixel_values"].squeeze(0)
    example["pixel_values"] = image
    del example["image"]
    return example

def train_epoch(model, optimizer, loss_function, dataloader):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(pixel_values)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
    
    return total_loss / len(dataloader)

# Validation function
def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            logits = model(pixel_values)
            preds = torch.argmax(logits, dim=1)  # Predicted class
            correct += (preds == labels).sum().item()  # Count correct predictions
            total += labels.size(0)
    
    return correct / total  # Return accuracy

@click.command()
@click.option("--config", default="configs/default.yaml", help="Path to the config file.")
def train(config):
    
    # load config file
    with open(config, "r") as f:
        config = yaml.safe_load(f)
    
    # load model
    model = ViTModel(
        pretrained_model_name=config["model"]["pretrained_model_name"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        hidden_layers=config["model"]["hidden_layers"],
        hidden_dim=config["model"]["hidden_dim"],
        batch_norm=config["model"]["batch_norm"],
        layer_norm=config["model"]["layer_norm"],
        hidden_activation=config["model"]["hidden_activation"],
        output_activation=config["model"]["output_activation"]
    )
    # load data
    # datasets = ds.load_dataset(config["dataset"], cache_dir=cache_dir)
    train_dataset = ds.load_dataset(config["dataset"], split='train', cache_dir=cache_dir)
    val_dataset = ds.load_dataset(config["dataset"], split='validation', cache_dir=cache_dir)
    # load image processor
    image_processor = AutoImageProcessor.from_pretrained(
        config["model"]["pretrained_model_name"],
        cache_dir=cache_dir,
        use_fast=True
    )

    train_dataset = train_dataset = train_dataset.with_transform(
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

    # Define loss function and optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    loss_func = torch.nn.CrossEntropyLoss()  # Multi-class classification loss
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=config["training"]["learning_rate"])

    # freeze backbone
    if config['training']['freeze_backbone']:
        for param in model.backbone.parameters():
            param.requires_grad = False
    
    model_tag = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    num_epochs = config["training"]["num_epochs"]
    print("Begin training...")
    # Training loop
    val_acc = evaluate(model, val_dataloader)
    print(f"Epoch 0: Validation Accuracy: {val_acc:.4f}")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, optimizer, loss_func, train_dataloader)
        val_acc = evaluate(model, val_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    save_dir = os.path.join(config["training"].get("save_dir", "./checkpoints"))
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, model_tag + ".pt"))


if __name__=="__main__":
    train()