# PyTorch Lightning Example

## Why PyTorch lightning

PyTorch Lightning is the deep learning framework with “batteries included” for professional AI researchers and machine learning engineers who need maximal flexibility while super-charging performance at scale.

Lightning organizes PyTorch code to remove boilerplate and unlock scalability.

## What is this example

This is a simple example of how to use PyTorch Lightning to train a model on the MNIST dataset.

## Install pytorch lightning

```bash
pip install pytorch-lightning
```

## Run

```bash
python3 mnist.py
```

## Results

```bash
Epoch 0: 100%|██████████| 5500/5500 [00:12<00:00, 435.99it/s, loss=1.86, v_num=0]
`Trainer.fit` stopped: `max_epochs=1` reached.
```