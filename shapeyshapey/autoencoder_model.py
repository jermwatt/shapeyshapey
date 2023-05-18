import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
from encoder import Encoder
from decoder import Decoder


# network model
class NN(pl.LightningModule):
    """
    Inputs:
        - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
        - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
        - latent_dim : Dimensionality of latent representation z
        - act_fn : Activation function used throughout the encoder network
    """


    def __init__(self,
                 latent_dim: int,
                 encoder_class: object = Encoder,
                 decoder_class: object = Decoder,
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
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)

        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

        # set other class params
        self.lr = lr
        self.loss_fn = F.l1_loss
        self.width = width
        self.height = height

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    # shorthand for using the same code for train, val and test
    def _common_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.loss_fn(x_hat, x)
        return loss, x_hat, x

    def training_step(self, batch, batch_idx):        
        loss, x_hat, y = self._common_step(batch, batch_idx)
        self.log_dict({"train_loss": loss},
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True,
                      sync_dist=True)

        # adding some images for testing purposes to logger
        x, y = batch
        if batch_idx == 50:
            x = x[:5]
            x_hat = x_hat[:5]

            imgs = torch.stack([x.view(-1,1,self.width,self.height), x_hat.view(-1,1,self.width,self.height)], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1, 1))            
            self.logger.experiment.add_image('original vs reconstruction',
                                             grid,
                                             self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, x_hat, y = self._common_step(batch, batch_idx)
        self.log_dict({"val_loss": loss},
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True,
                      sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, x_hat, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        x = x.reshape(x.shape[0], -1)
        x_hat = self.forward(x)
        return x_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
