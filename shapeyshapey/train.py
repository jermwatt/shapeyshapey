import pytorch_lightning as pl
from model import NN
from dataset import ShapeDataModule
from callbacks import MyPrintingCallback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import config
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tb_dir = parent_dir + '/tb_logs'

if __name__ == '__main__':
    
    # logger
    logger = TensorBoardLogger(
        save_dir=parent_dir + '/tb_logs',
        name='mnist_model_v0'
    )

    # Load Data
    dm = ShapeDataModule(
        data_dir=parent_dir + '/shape_dataset',
        batch_size=config.BATCH_SIZE,
        num_workers=1
    )

    # Initialize network
    model = NN(lr=config.LEARNING_RATE).to(config.device)

    # setup trainer
    trainer = pl.Trainer(logger=logger,
                         min_epochs=1,
                         max_epochs=5,
                         accelerator=config.ACCELERATOR,
                         callbacks=[MyPrintingCallback(),
                                    EarlyStopping(monitor='val_loss')],
                         precision=config.PRECISION,
                         devices=config.DEVICES,
                         num_nodes=1)

    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
