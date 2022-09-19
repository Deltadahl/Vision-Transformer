import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from einops import rearrange


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
            # GELU is an activation function similar to RELU, but have been shown to improve the performance https://arxiv.org/pdf/1606.08415v4.pdf
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )


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