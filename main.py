import torch
import torchvision
import lightning as L
from lightning.fabric import seed_everything
from fire import Fire
from datamodule import CRNNDataModule
from preparacion_datos import CaptchaDataset
from crnn import CaptchaCRNN
import numpy as np

aumentos = torchvision.transforms.Compose(
    [
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
    ]
)

BATCH_SIZE = 4

def main():
    ## pruebas para map-to-sequence
    modelo = CaptchaCRNN()
    dataset = CaptchaDataset(augments=aumentos)
    dm = CRNNDataModule(dataset=dataset, batch_size=BATCH_SIZE)

    feature_seq_tensor = []

    for data in dm.train_dataloader():
        # data es una lista con el tensor de la imagen y una tupla que contiene SOLO el string objetivo
        output_batch = modelo(data[0])
        print(output_batch[0][0])

if __name__ == '__main__':
    # Fire(main)
    main()
