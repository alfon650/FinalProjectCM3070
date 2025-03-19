import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

# code adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class ButterflyDataset(Dataset):
    def __init__(self, df_X: pd.DataFrame, df_y: pd.DataFrame, img_dir: str, transform, num_classes=75):
        self.filenames = df_X
        self.labels = df_y
        self.img_dir = img_dir
        self.num_classes = num_classes
        self.label_encoder = LabelEncoder()
        self.labels_label = self.label_encoder.fit_transform(df_y.values)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames.iloc[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(self.labels_label[idx], dtype=torch.long)
        return image, label
#end code adapted