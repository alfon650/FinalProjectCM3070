# FinalProjectCM3070

This is my final project for the degree CM3070.


## Prerequisites

- **Dataset**: You need to download and place the butterfly dataset in the `./datasets/butterfly dataset/` directory to execute the scripts. The dataset is required for both training and evaluation.
- **Hardware**: This project was developed and tested using an A100 GPU from Google Colaboratory.
- **Dependencies**: Ensure you have all required Python libraries installed 

    ```bash
    pip install -r requirements.txt
    ```

### Training the models

```bash
python train.py --epochs <number_of_epochs> --batch_size <batch_size> --model_name <model_name> --lr <learning_rate> [--dropout <dropout_rate>] [--freeze <freeze_layers>]
```

- `<number_of_epochs>`: Number of epochs to train (e.g., `10`).
- `<batch_size>`: Batch size for training (e.g., `32`).
- `<model_name>`: Model architecture to use (options: `resnet50`, `vgg16`, `efficientnet_b0`, `vit`, `vit_b_16`, `vit_l_16`).
- `<learning_rate>`: Learning rate (e.g., `0.001`).
- `<dropout_rate>`: (Optional) Dropout rate for `vit_b_16` or `vit_l_16` models (default: `0.0`, range: `0` to `1`).
- `<freeze_layers>`: (Optional) Number of layers to freeze for `vit_b_16` or `vit_l_16` models (default: `0`, non-negative integer).

**Example:**
```bash
python train.py --epochs 10 --batch_size 32 --model_name resnet50 --lr 0.001
```

### Evaluate the models

```bash
python evaluate.py --model_name <model_name> --checkpoint_path <path_to_checkpoint>
```

- `<model_name>`: Model architecture to evaluate (options: `resnet50`, `vgg16`, `efficientnet_b0`, `vit`, `vit_b_16`, `vit_l_16`).
- `<path_to_checkpoint>`: Path to the saved model checkpoint file (e.g., `./logs/butterfly_classifier_resnet50/checkpoints/butterfly-epoch=09-val_loss=0.1234_resnet50.ckpt`).

**Example:**
```bash
python evaluate.py --model_name resnet50 --checkpoint_path ./logs/butterfly_classifier_resnet50/checkpoints/butterfly-epoch=09-val_loss=0.1234_resnet50.ckpt
```

## Additional Notes

- **Notebook**: I have included a notebook called `train_evaluation_pipeline.ipynb` that I used for my experiments. It provides a step-by-step pipeline for training and evaluating models.