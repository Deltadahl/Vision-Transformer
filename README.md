# Vision Transformer - Explanation and Implementation
This repo explains how the Vision Tranformer (ViT) work, how to implement it and train it from scratch in PyTorch.

The dataset that is used is **ImageNet1k** bus only 10 classes is used so that the model can be trained on a single GPU in reasonable time.

## How does the ViT work? 
Convolutional neural networks (CNN) dominated the field of computer vision in the yeards 2012-2020. But in 2020 the paper [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) showed that ViT can attain state-of-the-art (SOTA) result with less computational cost.

The arcitecture of the ViT is shown in Figure 1. 
A 2D image is split into a number of patches e.g. 9 2D patches. Each patch is flattened and maped with a linear projection. 
The output of this mapping is concatinated with an extra learnable class [cls] embedding. The state of the [cls] embedding is randomly initialized, but it will accumulate information from the other tokens in the transformer and is used as the output of the transformer.


Unlike a CNN, a ViT have no inherent way to retrieve position from its input. Therefore a positional embedding is introduced. It could be concatinated with all embedded patches, but that comes with a computational cost, therefore the positional embedding is added to the embedded patches, whitch empirically gives good results [(Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929).
After the positional encoding is added the embedded patches is fed into the **Transformer encoder**.

<img src="Figures/Vit_fig_from_paper.png" width="800"> 

Figure 1: Model overview [[1]](https://arxiv.org/abs/2010.11929).


### Transformer Encoder
The ViT uses the encoder introduced in the famous [Attention Is All You Need](https://arxiv.org/abs/1706.03762?context=cs) paper, see Figure 1.
The encoder consists of two blocks, a multiheaded self-attention and a multilayer perceptron. Before each block a [layernorm](https://arxiv.org/abs/1607.06450) is applied and each block is surounded by a [residual connection](https://arxiv.org/abs/1512.03385). A residual connection not needed in theory, but empirically it is found to make a big differance. The residual connection can help the network to learn a desired mapping $H(x)$ by instead letting the network fit another mapping $F(x) := H(x) - x$ and then add $x$ to $F(x)$ to get the desired $H(x)$. 

The self-attention used here is a simple function of three matrices $Q, K, V$ (queries, keys, and values)
\begin{equation}
\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^{\top}}{\sqrt{d_k}})V,
\end{equation}
where the scaling factor $d_k$ is the dimension of the queries and keys.

Instead of performing a single attention function with $d_{model}$-dimensional queries, keys and values, multiheaded self-attention performes three linear projections to $d_k$, $d_k$ and $d_v$ dimensions respectively.

