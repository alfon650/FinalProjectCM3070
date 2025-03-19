import torch
import torch.optim as optim
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
import lightning as L
import torch.nn as nn

#code adapted from https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
class ButterflyClassifier(L.LightningModule):
    def __init__(self, model, num_classes, lr=0.001):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

        # metrics
        # code adapted from https://pytorch.org/torcheval/stable/generated/torcheval.metrics.MulticlassAccuracy.html
        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes, average="weighted")
        self.train_precision = MulticlassPrecision(num_classes=num_classes, average="weighted")
        self.train_recall = MulticlassRecall(num_classes=num_classes, average="weighted")

        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes, average="weighted")
        self.val_precision = MulticlassPrecision(num_classes=num_classes, average="weighted")
        self.val_recall = MulticlassRecall(num_classes=num_classes, average="weighted")
        # end code adapted

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        self.train_accuracy.update(preds, labels)
        self.train_precision.update(preds, labels)
        self.train_recall.update(preds, labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        self.val_accuracy.update(preds, labels)
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        train_accuracy = self.train_accuracy.compute()
        train_precision = self.train_precision.compute()
        train_recall = self.train_recall.compute()

        self.log("train_accuracy", train_accuracy, prog_bar=True)
        self.log("train_precision", train_precision, prog_bar=True)
        self.log("train_recall", train_recall, prog_bar=True)

        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_recall.reset()

    def on_validation_epoch_end(self):
        val_accuracy = self.val_accuracy.compute()
        val_precision = self.val_precision.compute()
        val_recall = self.val_recall.compute()

        self.log("val_accuracy", val_accuracy, prog_bar=True)
        self.log("val_precision", val_precision, prog_bar=True)
        self.log("val_recall", val_recall, prog_bar=True)

        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
#end code adapted