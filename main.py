import torch
import torchvision
import lightning as L
from lightning.fabric import seed_everything
from fire import Fire
from datamodule import CRNNDataModule
from preparacion_datos import CaptchaDataset
from crnn import CaptchaCRNN

aumentos = torchvision.transforms.Compose(
    [
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
    ]
)

BATCH_SIZE = 2

def main():
    ## pruebas para map-to-sequence
    modelo = CaptchaCRNN()
    dataset = CaptchaDataset(augments=aumentos)
    dm = CRNNDataModule(dataset=dataset, batch_size=BATCH_SIZE)

    feature_vector_list = []

    for data in dm.train_dataloader():
        # data es una lista con el tensor de la imagen y una tupla que contiene SOLO el string objetivo
        x = data[0].detach().clone()
        output = modelo(x)
        # un batch contiene n imágenes con 512 capas. cada capa contiene 2 listas, y cada lista 11
        for tensor in output:
            
            # recorrer columnas
            for i in range(len(tensor[0, 0])):
                feature_vector = []
                for j in range(len(tensor)):
                    feature_vector.append(tensor[j, 0, i])
                    feature_vector.append(tensor[j, 1, i])

                
                feature_vector_list.append(feature_vector.copy())
        break

    print(len(feature_vector_list))


    # return

if __name__ == '__main__':
    # Fire(main)
    main()
