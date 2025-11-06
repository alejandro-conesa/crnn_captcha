import torch
import os
from PIL import Image

class CaptchaDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_txt="dataset/guides", augments=None, name="base"):
        self.images = []
        self.labels = []
        self.path_to_txt = path_to_txt
        os.makedirs(self.path_to_txt, exist_ok=True)
        self.stage = "loaded"
        self.augments = augments
        self.name = name

        self.load_all()
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.stage == "loaded":
            img = Image.open(self.images[index])
            if self.augments is not None:
                img = self.augments(img)
            return img, self.labels[index]
        else:
            return self.images[index], self.labels[index]
    
    def __str__(self):
        return (
            f"Dataset {self.name} con {len(self.images)} elementos\n"
            + f"Item 0: {self.images[0]}, {self.labels[0]}"
        )
    
    def process_images(self, dataset_path="dataset/"):
        for path in os.listdir(dataset_path):
            self.images.append(dataset_path + path)
            label = path[:5]
            self.labels.append(label)
    
    def create_subsets(self):
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            self, lengths=[0.65, 0.15, 0.2]
        )
        return train_ds, val_ds, test_ds

    def load_from_subset(self, subset):
        for x, y in subset:
            self.images.append(x)
            self.labels.append(y)
        self.save_to_text()
    
    def save_to_text(self):
        with open(file=f"{self.path_to_txt}/{self.name}.txt", mode="w") as wfile:
            for i in range(len(self.images)):
                wfile.write(self.images[i] + " " + self.labels[i] + "\n")
    
    def load_from_text(self, file):
        with open(file=file, mode="r") as rfile:
            line = rfile.readline()
            while line != "":
                processed_line = line.strip("\n").split(sep=" ")
                self.images.append(processed_line[0])
                self.labels.append(processed_line[1])
                line = rfile.readline()

    def load_all(self):
        if self.name == "base":
            self.train_ds = CaptchaDataset(name="train")
            self.val_ds = CaptchaDataset(name="val")
            self.test_ds = CaptchaDataset(name="test")
            self.stage = "loading"

            if len(os.listdir(self.path_to_txt)) == 4:
                # cargar por texto
                self.load_from_text(file=self.path_to_txt + "/base.txt")
                self.train_ds.load_from_text(file=self.path_to_txt + "/train.txt")
                self.val_ds.load_from_text(file=self.path_to_txt + "/val.txt")
                self.test_ds.load_from_text(file=self.path_to_txt + "/test.txt")
            else:
                # procesar y crear texto
                self.process_images()
                train_subset, val_subset, test_subset = self.create_subsets()
                # los subsets son una clase "envoltorio" de los dataset
                # gracias a eso podemos hacer toda la lógica de procesamiento en la clase base
                self.train_ds.load_from_subset(train_subset)
                self.val_ds.load_from_subset(val_subset)
                self.test_ds.load_from_subset(test_subset)
        
            self.save_to_text()
            self.stage = "loaded"
    
    def get_splits(self):
        return self.train_ds, self.val_ds, self.test_ds

if __name__ == "__main__":
    dataset = CaptchaDataset()
    print(dataset)
    train_ds, val_ds, test_ds = dataset.get_splits()
    print(train_ds)
    print(val_ds)
    print(test_ds)
