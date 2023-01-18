import os

import pytorch_lightning as pl

from data_loader_mnist import MNISTDataModule
from mnist import LitModel


def model_train():
    # without dataloader module
    # train_loader = DataLoader(MNIST(os.getcwd(), download=True, transform=transforms.ToTensor()))

    data_module = MNISTDataModule(data_dir=os.path.join(os.getcwd(), 'mnist_data'))
    trainer = pl.Trainer(max_epochs=10)
    model = LitModel()

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module, ckpt_path="best")
    # save best model
    trainer.save_checkpoint("best.ckpt")


def model_test():
    # load model
    mnist_data_module = MNISTDataModule(data_dir=os.path.join(os.getcwd(), 'mnist_data'))
    model = LitModel.load_from_checkpoint("best.ckpt")
    # test model
    trainer = pl.Trainer()
    trainer.test(model, datamodule=mnist_data_module)


if __name__ == '__main__':
    model_train()
