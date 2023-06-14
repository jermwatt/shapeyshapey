import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
from encoder import Encoder
import scipy.optimize

def project_doubly_stochastic(X):
    # Ensure that the tensor is non-negative
    X = torch.clamp(X, min=0)

    # Normalize rows and columns iteratively
    for _ in range(10):
        # Normalize rows
        X /= X.sum(dim=1, keepdim=True)

        # Normalize columns
        X /= X.sum(dim=0, keepdim=True)

    # Convert tensor to numpy array
    X_np = X.detach().numpy()

    # Perform linear sum assignment using the Hungarian algorithm
    row_indices, col_indices = scipy.optimize.linear_sum_assignment(-X_np)

    # Create a tensor with zeros
    X_projected = torch.zeros_like(X)

    # Assign the non-zero values from the original tensor to the projected tensor
    X_projected[row_indices, col_indices] = X[row_indices, col_indices]

    return X_projected




# network model
class NN(pl.LightningModule):
    def __init__(self,
                 encoder_class: object = Encoder,
                 width: int = 32,
                 height: int = 32,
                 base_channel_size: int = 32,
                 num_input_channels: int = 1,
                 lr: float = 1e-3
                 ):

        super().__init__()
        # Saving hyperparameters of autoencoder
        # self.save_hyperparameters()

        # Creating encoder and decoder
        self.encoder = encoder_class(width, height)

        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

        # set other class params
        self.lr = lr
        self.loss_fn = F.l1_loss
        self.width = width
        self.height = height

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # step
        optimizer.step(closure=optimizer_closure)

        # project
        for param in self.parameters():
            param.data = param.data.clamp(min=0.)

            # project to doubly stochastic matrix
            param.data = project_doubly_stochastic(param.data)


    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        x_hat = self.encoder(x)
        return x_hat

    # shorthand for using the same code for train, val and test
    def _common_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.encoder(x)
        loss = self.loss_fn(x_hat, y)
        
        # print(f"batch_idx: {batch_idx}")
        # print(f"loss: {loss}")
        # print('sum of x is: ', torch.sum(x))
        # print('sum of x_hat is: ', torch.sum(x_hat))
        # print('sum of y is: ', torch.sum(y))
        
        return loss, x, x_hat, y

    def training_step(self, batch, batch_idx):        
        loss, x, x_hat, y = self._common_step(batch, batch_idx)
        self.log_dict({"train_loss": loss},
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True,
                      sync_dist=True)

        # adding some images for testing purposes to logger
        x, y = batch
        if batch_idx == 2:
            x = x[:5]
            x_hat = x_hat[:5]
            y = y[:5]

            imgs = torch.stack([x.view(-1,1,self.width,self.height), 
                                x_hat.view(-1,1,self.width,self.height), 
                                y.view(-1,1,self.width,self.height)], dim=1).flatten(0, 1)
            
            grid = torchvision.utils.make_grid(imgs, nrow=3 , normalize=True, range=(-1, 1))            
            self.logger.experiment.add_image('input vs reconstruction vs output',
                                             grid,
                                             self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, x, x_hat, y = self._common_step(batch, batch_idx)
        self.log_dict({"val_loss": loss},
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True,
                      sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, x, x_hat, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        x = x.reshape(x.shape[0], -1)
        x_hat = self.forward(x)
        return x_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
