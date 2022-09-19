import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, num_classes, images_per_class, val_images_per_class, batch_size):
    transforms_train = T.Compose([
                    T.Resize((256, 256)), # TODO
                    T.RandomHorizontalFlip(),
                    T.RandAugment(),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),])
    transforms_val = T.Compose([
                    T.Resize((256, 256)), # TODO
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),])


    data_train_temp = datasets.ImageFolder(data_dir, transform=transforms_train)
    data_val_temp = datasets.ImageFolder(data_dir, transform=transforms_val)

    indices_train = []
    indices_val = []
    for i in range(num_classes):
        for j in range(images_per_class):
            if j < images_per_class-val_images_per_class:
                indices_train.append(i*images_per_class+j)
            else:
                indices_val.append(i*images_per_class+j)

    data_train = torch.utils.data.Subset(data_train_temp, indices_train)
    data_val = torch.utils.data.Subset(data_val_temp, indices_val)

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader