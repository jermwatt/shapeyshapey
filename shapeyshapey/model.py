import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from metrics import MyAccuracy
from torch import nn


# network model
class NN(pl.LightningModule):
    def __init__(self,
                 input_size,
                 num_classes,
                 lr=1e-3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass',
                                              num_classes=num_classes)
        self.my_accuracy = MyAccuracy()
        self.f1score = torchmetrics.F1Score(task='multiclass',
                                            num_classes=num_classes)
        self.lr = lr

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # shorthand for using the same code for train, val and test
    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.shape[0], -1)
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        return loss, logits, y 

    def training_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        accuracy = self.my_accuracy(logits, y)
        f1score = self.f1score(logits, y)
        self.log_dict({"train_loss": loss,
                       "train_accuracy": accuracy,
                       "train_f1score": f1score},
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        accuracy = self.my_accuracy(logits, y)
        self.log_dict({"val_loss": loss,
                      "val_accuracy": accuracy},
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        x = x.reshape(x.shape[0], -1)
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
