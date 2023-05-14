import pytorch_lightning as pl
from model import NN
from dataset import MnistDataModule
import config
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if __name__ == '__main__':
    # Load Data
    dm = MnistDataModule(
        data_dir=parent_dir + '/dataset',
        batch_size=config.BATCH_SIZE,
        num_workers=4
    )

    # Initialize network
    model = NN(input_size=config.INPUT_SIZE,
               num_classes=config.NUM_CLASSES,
               lr=config.LEARNING_RATE).to(config.device)

    # setup trainer
    trainer = pl.Trainer(max_epochs=3)
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
