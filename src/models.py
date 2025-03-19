import torch
import torch.nn as nn

# code adapted from https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
class ButterflyModel(nn.Module):
    def __init__(self, base_model, num_classes=75):
        super(ButterflyModel, self).__init__()
        # Remove the final layer from the base model
        self.base_model = nn.Sequential(*(list(base_model.children())[:-1]))

        # to determine the output size of the base model
        dummy_input = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            feature_size = self.base_model(dummy_input).view(1, -1).size(1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(feature_size, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.batch_norm(x)
        x = self.fc2(x)
        return x

class ButterflyModelVIT(nn.Module):
    def __init__(self, base_model, num_classes=75, dropout=0.5, vit_model_freeze=0):
        super(ButterflyModelVIT, self).__init__()

        self.base_model = base_model
        self.base_model.heads = nn.Identity()

        num_encoder_layer = len(self.base_model.encoder.layers)
        freeze_layers = min(vit_model_freeze, num_encoder_layer)

        for i, block in enumerate(self.base_model.encoder.layers):
            if i < freeze_layers:
                # Code adapted from https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088
                for param in block.parameters():
                    param.requires_grad = False
                # end code adapted

        if 'vit_b_16' in str(type(base_model)).lower():
            feature_size = 768
        elif 'vit_l_16' in str(type(base_model)).lower():
            feature_size = 1024
        else:
          raise Exception("Model not supported")

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(feature_size, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

        for param in self.fc1.parameters():
            param.requires_grad = True
        for param in self.fc2.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.base_model(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.batch_norm(x)
        x = self.fc2(x)
        return x
#end code adapted


# code adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html

def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor image has shape [B, C, H, W] b=batch size or number of images, C=number of channels (3=RGB), h=pixels height, w=pixels width
        patch_size - pixels per dimension of the patches (integer)
        flatten_channels - True-> the patches retur a flattened format
                           instead of image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # hint: [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # hint: [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # hint: [B, H'*W', C*p_H*p_W]
    return x


class AttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class VisionTransformer(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers
        self.input_layer = nn.Linear(num_channels*(patch_size**2), embed_dim)
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        # Parameters and dimensions
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,1+num_patches,embed_dim))


    def forward(self, x):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add classification token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]

        # Apply Transformer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out

  #end code adapted