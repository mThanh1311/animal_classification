import os 
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize


class Animal (Dataset):
    def __init__(self, root = "./Dataset", train=True, transform=None):
        self.categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

        if train:
            data_path = os.path.join(root, "train")
        else:
            data_path = os.path.join(root, "test")

        # just save the image path to list
        self.images_path = []
        self.labels = []

        for category in self.categories:
            category_path = os.path.join(data_path, category)
            for image_name in os.listdir(category_path):
                image_path = os.path.join(category_path, image_name)
                self.images_path.append(image_path)
                self.labels.append(self.categories.index(category))

        self.transform = transform

    def __len__(self):
        return (len(self.labels))
    
    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index])

        if self.transform:
            image = self.transform(image)

        label = self.labels[index]

        return image, label
    
if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        Resize((224,224))
    ])

    dataset = Animal(root = "./Dataset", train=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=8,
                              shuffle=True,
                              drop_last=True)
    for image, label in (train_loader):
        print(image.shape)
        print(label)