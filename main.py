import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
import wandb

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import random

from torchvision import datasets
from torch.utils.data import DataLoader
from einops import rearrange

from utils.data_augmentation import RandomMixup
from utils.data_loaders import get_data_loaders
from utils.train_and_evaluate import train_epoch, evaluate_epoch
from ViT_model import ViT
from ResNet import ResNet18

wandb.init(project="ImageNet1k-10")
config = wandb.config
torch.manual_seed(7)

LR = 1e-3
BATCH_SIZE = 16
N_EPOCHS = int(1e5) # inf :)
DROPOUT = 0.1
WEIGHT_DECAY = 0.01
IMAGE_SIZE = 256
PATCH_SIZE = 16
NUM_CLASSES = 10
IMAGES_PER_CLASS = 1300
VAL_IMAGES_PER_CLASS = 200
DATA_DIR = 'ImageNet1k-10'

# Load model
path_to_model_load = r''
load_model = False

config.lr = LR
config.batch_size = BATCH_SIZE
config.n_epochs = N_EPOCHS
config.dropout = DROPOUT
config.weight_decay = WEIGHT_DECAY


if __name__ == "__main__":

    start_time = time.time()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('Running on the GPU')
    else:
        device = torch.device("cpu")
        print('Running on the CPU')

    train_loader, val_loader = get_data_loaders(DATA_DIR, NUM_CLASSES, IMAGES_PER_CLASS, VAL_IMAGES_PER_CLASS, BATCH_SIZE)
    mixup = RandomMixup(num_classes=NUM_CLASSES)

    model = ViT(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=NUM_CLASSES, channels=3,
                dim=128, depth=6, heads=16, mlp_dim=256, dropout=DROPOUT, stochastic_depth_prob=0).to(device)
    #model = ResNet18(num_classes=NUM_CLASSES, dropout=DROPOUT).to(device)

    wandb.watch(model)

    # Load saved model
    if os.path.exists(path_to_model_load) and load_model:
        print('Loading model.')
        model.load_state_dict(torch.load(path_to_model_load))
        model.train()

    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    max_acc_val = 85
    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        start_time_epoch = time.time()
        
        loss_train = train_epoch(model, optimizer, train_loader, device, loss_fun, IMAGES_PER_CLASS, VAL_IMAGES_PER_CLASS, NUM_CLASSES, mixup)
        loss_val, acc_val = evaluate_epoch(model, val_loader, device, loss_fun, VAL_IMAGES_PER_CLASS, NUM_CLASSES)
        
        wandb.log({"loss train": loss_train, "loss val": loss_val, "acc val": acc_val, "Time for epoch": (time.time() - start_time_epoch)})
        print('Execution time for Epoch:', '{:5.2f}'.format(time.time() - start_time_epoch), 'seconds')
        
        if max_acc_val < acc_val:
            max_acc_val = acc_val
            print('Saving model')
            path_to_model_save = r'Saved_models/accuracy_' + str(acc_val.item()) + ".pt"
            torch.save(model.state_dict(), path_to_model_save)
        
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds\n')
