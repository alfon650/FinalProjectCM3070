import lightning.pytorch as L
from torch.utils.data import DataLoader
from torchvision.models import resnet50, vgg16, efficientnet_b0, vit_b_16, vit_l_16
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse  # Added for argument parsing

from src.config import NUM_CLASSES, vit_kwargs, val_transform
from src.models import ButterflyModelVIT, ButterflyModel, VisionTransformer
from src.classifier import ButterflyClassifier
from src.dataset import ButterflyDataset

# Argument parser setup
parser = argparse.ArgumentParser(description="Evaluate a butterfly classification model")
parser.add_argument("--model_name", type=str, required=True, 
                   choices=["resnet50", "vgg16", "efficientnet_b0", "vit", "vit_b_16", "vit_l_16"],
                   help="Name of the model to evaluate")
parser.add_argument("--checkpoint_path", type=str, required=True,
                   help="Path to the model checkpoint file")
args = parser.parse_args()

df = pd.read_csv("./datasets/butterfly dataset/Training_set.csv")
dataset_path = "./datasets/butterfly dataset/"
X_train, X_val, y_train, y_val = train_test_split(df["filename"], df["label"], test_size=0.2, random_state=42)
val_dataset = ButterflyDataset(df_X=X_val, df_y=y_val, img_dir=f'{dataset_path}/train/', transform=val_transform, num_classes=NUM_CLASSES)
val_loader = DataLoader(val_dataset, batch_size=128, num_workers=8, shuffle=False)

def evaluate(model_name, checkpoint_path):
    model_mapping = {
        "resnet50": resnet50(weights='DEFAULT'),
        "vgg16": vgg16(weights='DEFAULT'),
        "efficientnet_b0": efficientnet_b0(weights='DEFAULT'),
        "vit": VisionTransformer(**vit_kwargs),
        "vit_b_16": vit_b_16(weights='DEFAULT'),
        "vit_l_16": vit_l_16(weights='DEFAULT')
    }
    base_model = model_mapping[model_name]
    if "vit" in model_name and not ("vit_b_16" in model_name or "vit_l_16" in model_name):
        butterfly_classifier = ButterflyClassifier.load_from_checkpoint(
            checkpoint_path,
            model=base_model,
            num_classes=NUM_CLASSES
        )
    else:
        butterfly_classifier = ButterflyClassifier.load_from_checkpoint(
            checkpoint_path,
            model=ButterflyModelVIT(base_model=base_model, num_classes=NUM_CLASSES) if "vit_" in model_name else ButterflyModel(base_model=base_model, num_classes=NUM_CLASSES),
            num_classes=NUM_CLASSES
        )

    # Code adapted from https://medium.com/@rautshiska/logging-multiple-metrics-at-different-stages-with-tensorboard-and-pytorch-lightning-400509e834b2
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        logger=False,
        callbacks=[],
        enable_checkpointing=False,
        enable_model_summary=False,
        inference_mode=True,
        default_root_dir=None
    )

    print(f"Model: {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.validate(butterfly_classifier.to(device), dataloaders=val_loader)
    # End code adapted

# Call the evaluate function with parsed arguments
if __name__ == "__main__":
    evaluate(args.model_name, args.checkpoint_path)