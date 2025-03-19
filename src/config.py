import torchvision.transforms as transforms

#code adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html
vit_kwargs={
    'embed_dim': 256,
    'hidden_dim': 512,
    'num_heads': 32,
    'num_layers': 12,
    'patch_size': 32,
    'num_channels': 3,
    'num_patches': 64,
    'num_classes': 75,
    'dropout': 0.2
}
#end code adapted

NUM_CLASSES = 75

# Code adapted from https://pytorch.org/vision/0.9/transforms.html 
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=20, shear=0.2, scale=(0.8, 1.2)),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
# end code adapted