# Deep learning imports
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
import wandb

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from einops import rearrange

# Other imports
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import random



# Functions to display single or a batch of sample images
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def show_batch(dataloader):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()    
    imshow(make_grid(images))
    
def show_image(dataloader):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    random_num = random.randint(0, len(images)-1)
    imshow(images[random_num])
    label = labels[random_num]
    print(f'Label: {label}, Shape: {images[random_num].shape}')


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, stochastic_depth_prob=0.5):
        super().__init__()
        assert image_size % patch_size == 0, 'image size must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, stochastic_depth_prob)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            # TODO dropout here?
            # GELU is an activation function similar to RELU, but have been shown to improve the performance https://arxiv.org/pdf/1606.08415v4.pdf
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )
        self.init_params()


    # Xavier initialization of parameters. This can help, but I didn't see that big of an impact in this case.
    # TODO remove?
    def init_params(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, img, training = True):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)

        # TODO check this, copy the cls-token batchsize times
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        # prepend the cls token to every image
        x = torch.cat((cls_tokens, x), dim=1)
        # TODO check if pos-embedding << x
        x += self.pos_embedding
        x = self.transformer(x, training)

        # Only uses the cls-token to classify the image
        x = self.to_cls_token(x[:, 0])
        x = torch.sigmoid(x)
        return self.mlp_head(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, stochastic_depth_prob_rate_last):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([])
        self.stochastic_depth_prob_rate_last = stochastic_depth_prob_rate_last
        for _ in range(depth):

            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout)))
            ]))

    def forward(self, x, training):
        d  = self.depth - 1
        for layer_num, (attn, ff) in enumerate(self.layers):
            
            # Stochastic depth probability implementation
            if self.depth > 1 and training:
                prob_to_skip = self.stochastic_depth_prob_rate_last*layer_num/(self.depth-1)
                rand_num = np.random.rand()
                if rand_num < prob_to_skip:
                    return x

            x = attn(x)
            x = ff(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.l1 = nn.Linear(dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, dim)
        self.dropout = dropout

    def forward(self, x):
        x = self.l1(x)
        x = F.dropout(x, self.dropout)
        x = F.gelu(x)
        x = self.l2(x)
        x = F.dropout(x, self.dropout)
        return x


class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    d`"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.
    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch, target):
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )
        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target
        
        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target


wandb.init(project="ViT-ImageNet1k_20")
config = wandb.config
torch.manual_seed(42)

scaling_factor = 1
LR = 3e-3/(scaling_factor) # TODO vary the LR (or Batch Size) # TODO parameter sweep during the weekend?
#BATCH_SIZE_TRAIN = 1024//scaling_factor
#BATCH_SIZE_VAL = 1024//scaling_factor
BATCH_SIZE_TRAIN = 64 # 1024 #32
BATCH_SIZE_VAL = 64 #1024 #32 #TODO test 32 batch size but higher LR!!
N_EPOCHS = int(1e10) # inf :)
DROPOUT = 0.1
WEIGHT_DECAY = 0.01
NUM_CLASSES = 20
IMAGE_SIZE = 256
#DATA_DIR = 'tiny-imagenet-200' # Original images come in shapes of [3,64,64]
DATA_DIR = 'ImageNet1k-20'

config.lr = LR
config.batch_size_train = BATCH_SIZE_TRAIN
config.batch_size_val = BATCH_SIZE_VAL
config.n_epochs = N_EPOCHS
config.dropout = DROPOUT
config.weight_decay = WEIGHT_DECAY

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Running on the GPU')
else:
    device = torch.device("cpu")
    print('Running on the CPU')

# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
#VALID_DIR = os.path.join(DATA_DIR, r'val/images')
VALID_DIR = os.path.join(DATA_DIR, 'train') #TODO

transforms_train = T.Compose([
                T.Resize((256, 256)), # TODO
                T.RandomHorizontalFlip(),
                T.RandAugment(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),             
])
transforms_val = T.Compose([
                T.Resize((256, 256)), # TODO
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),             
])


data_train_temp = datasets.ImageFolder(TRAIN_DIR, transform=transforms_train)
data_val_temp = datasets.ImageFolder(VALID_DIR, transform=transforms_val)
'''# train_size_frac = 0.8
# train_size = int(len(data_train_temp)*train_size_frac)
# val_size = len(data_train_temp) - train_size'''

indices_train = []
indices_val = []
NUM_OF_SAMPLES = 1300
VAL_SIZE = 100
for i in range(NUM_CLASSES):
    for j in range(NUM_OF_SAMPLES):
        if j < NUM_OF_SAMPLES-VAL_SIZE:
            indices_train.append(i*NUM_OF_SAMPLES+j)
        else:
            indices_val.append(i*NUM_OF_SAMPLES+j)

data_train = torch.utils.data.Subset(data_train_temp, indices_train)
data_val = torch.utils.data.Subset(data_val_temp, indices_val)

train_loader = DataLoader(data_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
val_loader = DataLoader(data_val, batch_size=BATCH_SIZE_VAL, shuffle=False)

#show_batch(train_loader)
#show_batch(val_loader)
#train_loader = DataLoader(data_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
#val_loader = DataLoader(data_val, batch_size=BATCH_SIZE_VAL, shuffle=False)

'''#import torchvision
orig_set = datasets.Imagefolder(TRAIN_DIR, transform=transforms_train)  # your dataset
n = len(orig_set)  # total number of examples
n_test = int(0.1 * n)  # take ~10% for test
test_set = torch.utils.data.Subset(orig_set, range(n_test))  # take first 10%
train_set = torch.utils.data.Subset(orig_set, range(n_test, n))  # take the rest 

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
val_loader = DataLoader(test_set, batch_size=BATCH_SIZE_VAL, shuffle=False)'''
'''
data_train = datasets.ImageFolder(TRAIN_DIR, transform=transforms_train)
train_loader = DataLoader(data_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
data_val = datasets.ImageFolder(VALID_DIR, transform=transforms_val)
val_loader = DataLoader(data_val, batch_size=BATCH_SIZE_VAL, shuffle=False)'''
    

mixup = RandomMixup(num_classes=NUM_CLASSES)

def train_epoch(model, optimizer, data_loader):
    total_samples = len(data_loader.dataset)
    model.train()
    total_loss = 0
    start_time_epoch = time.time()
    time_sice_print = time.time()
    for i, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        (data, target) = mixup(data, target)     

        optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        loss = loss_fun(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if time.time()-time_sice_print > 60*1:
            time_sice_print = time.time()
            print(f'{(time.time()-start_time_epoch)//60} min elapsed this epoch')
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            
    avg_loss = total_loss /((NUM_OF_SAMPLES-VAL_SIZE)*NUM_CLASSES)
    return avg_loss


def evaluate(model, data_loader):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            #output = F.log_softmax(model(data, training=False), dim=1)
            # TODO change back when ResNet18 is not used
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / (VAL_SIZE*NUM_CLASSES)
    acc = 100*(correct_samples / total_samples)
    print('Average val loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)')
    return avg_loss, acc


start_time = time.time()
#model = ViT(image_size=64, patch_size=8, num_classes=NUM_CLASSES, channels=3,
#            dim=64, depth=16, heads=8, mlp_dim=128, dropout=DROPOUT).to(device)



# TODO try without depth?
#model = ViT(image_size=IMAGE_SIZE, patch_size=IMAGE_SIZE//8, num_classes=NUM_CLASSES, channels=3,
#            dim=64, depth=8, heads=8, mlp_dim=128, dropout=DROPOUT, stochastic_depth_prob=0).to(device)

model = ViT(image_size=IMAGE_SIZE, patch_size=IMAGE_SIZE//8, num_classes=NUM_CLASSES, channels=3,
            dim=64, depth=16, heads=8, mlp_dim=128, dropout=DROPOUT, stochastic_depth_prob=0).to(device)
# TODO try with other batch size


#from ResNet import ResNet18
#from ResNet import ResNet18
#model = ResNet18(num_classes=NUM_CLASSES, dropout=DROPOUT).to(device)

wandb.watch(model)

# Load model
path_to_model_load = r'Models\min_loss_0.7818917236328125_epoch_104.pt'
load_model = True
if os.path.exists(path_to_model_load) and load_model:
    print('Loading model.')
    model.load_state_dict(torch.load(path_to_model_load))
    model.train()

loss_fun = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

min_loss_val = np.inf
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    start_time_epoch = time.time()
    loss_train = train_epoch(model, optimizer, train_loader)
    loss_val, acc_val = evaluate(model, val_loader)
    wandb.log({"loss train": loss_train, "loss val": loss_val, "acc val": acc_val, "Time for epoch": (time.time() - start_time_epoch)})
    print('Execution time for Epoch:', '{:5.2f}'.format(time.time() - start_time_epoch), 'seconds')
    if min_loss_val > loss_val:
        min_loss_val = loss_val
        print('Saving model')
        path_to_model_save = r'Models/min_loss_' + str(loss_val) + '_epoch_' + str(epoch) + ".pt"
        torch.save(model.state_dict(), path_to_model_save)
    
print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds\n')



