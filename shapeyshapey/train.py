import pytorch_lightning as pl
from model import NN
from dataset import MnistDataModule
from callbacks import MyPrintingCallback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
import torch
import config
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tb_dir = parent_dir + '/tb_logs'

if __name__ == '__main__':
    # profiler
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_dir),
        trace_memory=True,
        schedule=torch.profiler.schedule(
            skip_first=1,
            wait=1,
            warmup=1,
            active=20)
    )

    # logger
    logger = TensorBoardLogger(
        save_dir=parent_dir + '/tb_logs',
        name='mnist_model_v0'
    )

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
    trainer = pl.Trainer(profiler=profiler,
                         logger=logger,
                         min_epochs=1,
                         max_epochs=2,
                         accelerator=config.ACCELERATOR,
                         callbacks=[MyPrintingCallback(),
                                    EarlyStopping(monitor='val_loss')],
                         precision=config.PRECISION,
                         devices=config.DEVICES,
                         num_nodes=1)

    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
