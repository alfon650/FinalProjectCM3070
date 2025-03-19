import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
from torchvision.models import resnet50, vgg16, efficientnet_b0, vit_b_16, vit_l_16
import argparse


from config import val_transform, train_transform, NUM_CLASSES, vit_kwargs
from dataset import ButterflyDataset
from classifier import ButterflyClassifier
from models import ButterflyModel, ButterflyModelVIT, VisionTransformer


def train(epochs, batch_size, model_name, lr, dropout, freeze = 0):
    classifier_mapping = {
        "resnet50": ButterflyClassifier(
            model=ButterflyModel(base_model=resnet50(weights='DEFAULT'), num_classes=NUM_CLASSES),
            num_classes=NUM_CLASSES,
            lr=lr
        ),
        "vgg16": ButterflyClassifier(
            model=ButterflyModel(base_model=vgg16(weights='DEFAULT'), num_classes=NUM_CLASSES),
            num_classes=NUM_CLASSES,
            lr=lr
        ) ,
        "efficientnet_b0": ButterflyClassifier(
            model=ButterflyModel(base_model=efficientnet_b0(weights='DEFAULT'), num_classes=NUM_CLASSES),
            num_classes=NUM_CLASSES,
            lr=lr
        ),
        "vit":ButterflyClassifier(
            model=VisionTransformer(**vit_kwargs),
            num_classes=NUM_CLASSES,
            lr=lr
        ),
        "vit_b_16": ButterflyClassifier(
            model=ButterflyModelVIT(base_model=vit_b_16(weights='DEFAULT'), num_classes=NUM_CLASSES, vit_model_freeze=freeze, dropout=dropout),
            num_classes=NUM_CLASSES,
            lr=lr
        ),
        "vit_l_16": ButterflyClassifier(
            model=ButterflyModelVIT(base_model=vit_l_16(weights='DEFAULT'), num_classes=NUM_CLASSES, vit_model_freeze=freeze, dropout=dropout),
            num_classes=NUM_CLASSES,
            lr=lr
        )
    }
    butterfly_classifier = classifier_mapping[model_name]


    logging_dir =  "./logs/"
    os.makedirs(logging_dir, exist_ok=True)
    logging_name = "butterfly_classifier"
    dataset_path = "./datasets/butterfly dataset/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv("./datasets/butterfly dataset/Training_set.csv")
    # Same random state for all experiments
    X_train, X_val, y_train, y_val = train_test_split(df["filename"], df["label"], test_size=0.2, random_state=42)
    print(f"X and y train shape: {len(X_train)} / {len(y_train)} ")
    print(f"X and y test shape: {len(X_val)} / {len(y_val)} ")
    train_dataset = ButterflyDataset(df_X=X_train, df_y=y_train, img_dir=f'{dataset_path}/train/', transform=train_transform, num_classes=NUM_CLASSES)
    val_dataset = ButterflyDataset(df_X=X_val, df_y=y_val, img_dir=f'{dataset_path}/train/',transform=val_transform, num_classes=NUM_CLASSES)
    print(f"Train dataset: {len(train_dataset)}")
    print(f"Val dataset: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

    #code adapted from https://medium.com/@rautshiska/logging-multiple-metrics-at-different-stages-with-tensorboard-and-pytorch-lightning-400509e834b2
    logger = TensorBoardLogger(logging_dir, name=f"{logging_name}_{model_name}")

    # Checkpointing and logging
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="butterfly-{epoch:02d}-{val_loss:.4f}_"+f"{model_name}",
    )


    # Trainer for training without loops
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
        log_every_n_steps=0,
        logger=logger
    )

    trainer.fit(butterfly_classifier.to(device), train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a butterfly classifier model")
    
    parser.add_argument('--epochs', type=int, required=True, 
                       help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, required=True, 
                       help='Batch size for training')
    parser.add_argument('--model_name', type=str, required=True, 
                       help='Model architecture to use (resnet50, vgg16, efficientnet_b0, vit, vit_b_16, vit_l_16)')
    parser.add_argument('--lr', type=float, required=True, 
                       help='Learning rate')
    
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout rate (only applicable for vit_b_16 or vit_l_16)')
    parser.add_argument('--freeze', type=int, default=0,
                       help='Number of layers to freeze (only applicable for vit_b_16 or vit_l_16)')
    
    args = parser.parse_args()
    
    valid_models = ["resnet50", "vgg16", "efficientnet_b0", "vit", "vit_b_16", "vit_l_16"]
    
    if args.model_name not in valid_models:
        raise ValueError(f"model_name must be one of {valid_models}, got {args.model_name}")
    
    vit_models = ["vit_b_16", "vit_l_16"]
    if args.model_name not in vit_models:
        if args.dropout != 0.0:
            raise ValueError("dropout can only be set for vit_b_16 or vit_l_16 models")
        if args.freeze != 0:
            raise ValueError("freeze can only be set for vit_b_16 or vit_l_16 models")
    
    if args.dropout < 0 or args.dropout > 1:
        raise ValueError("dropout must be between 0 and 1")
    
    if args.freeze < 0:
        raise ValueError("freeze must be non-negative")
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=args.model_name,
        lr=args.lr,
        dropout=args.dropout,
        freeze=args.freeze
    )