import os.path
import torch
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import random
import os
import numpy as np
T = transforms


class DataMyload(Dataset):
    def __init__(self, train_list_dir, test_list_dir, image_dir, transform, train=True):
        train_labels = []
        train_images_names = []
        test_labels = []
        test_images_names = []
        with open(train_list_dir,"r") as f: 
            lines = f.readlines()      
            for line in lines:
                if 'noise' not in line:
                    image_name, label = line.strip('\n').split(' ')
                    #print(image_name, label)
                    label = int(label) / 100.0
                    train_labels.append(label)
                    train_images_names.append(image_name)
        with open(test_list_dir,"r") as f: 
            lines = f.readlines()      
            for line in lines:
                if 'noise' not in line:
                    image_name, label = line.strip('\n').split(' ')
                    label = int(label) / 100.0
                    test_labels.append(label)
                    test_images_names.append(image_name)

        if train:
            self.list = train_images_names
            self.labels = train_labels
        else:
            self.list = test_images_names
            self.labels = test_labels
        self.image_dir = image_dir
        self.transform = transform
    def __getitem__(self, index):
        filename = self.list[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        label = np.float32(self.labels[index])

        return self.transform(image), label

    def __len__(self):
        """Return the number of images."""
        return len(self.list)


def get_loader(train_list_dir='../train.txt',\
     test_list_dir='../test.txt',\
     image_dir='/root/data/aishijie/Project/Morph_mtcnn_1.3_0.35_0.3/',\
     batch_size=16, image_size=224, num_workers=20, train=True):
    """Build and return a data loader."""
    transform = []
    transform.append(T.RandomHorizontalFlip())
    transform.append(T.RandomCrop(image_size))
    #transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size()))
    transform = T.Compose(transform)

    dataset = DataMyload(train_list_dir, test_list_dir, image_dir, transform, train)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=num_workers)
    return data_loader

if __name__ == "__main__":
    dataset = get_loader()
    dataloader = iter(dataset)
    imgs, labels = next(dataloader)
    print(imgs.shape, labels.shape)
    print(labels)
