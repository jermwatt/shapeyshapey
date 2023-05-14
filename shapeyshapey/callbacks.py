from pytorch_lightning.callbacks import Callback, EarlyStopping


class MyPrintingCallback(Callback):
    def __nit__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print('Starting to train ...')

    def on_train_end(self, trainer, pl_module):
        print('Finished training')