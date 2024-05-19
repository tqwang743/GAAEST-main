# Graph Attention Automatic Encoder Based on Contrastive Learning for Recognition of Spatial Domain of Spatial Transcriptomics



## Overview
GAAEST is designed for spatial domains recognition of spatial transcriptomics (ST) data. 

In the GAAEST, there are five main components, namely data pre-processing, neighbor graph construction and data augmentation, autoencoder, self-supervised contrastive learning for embedding refinement and spatial clustering. Specifically, GAAEST takes ST data as input, which consists of a gene expression matrix and spatial location information of spots. Firstly, we filter out the low-quality genetic data in the raw gene expression matrix. Next, a neighbor graph is constructed as the input of the autoencoder, with edges representing the spatial neighborhood relationships, and nodes characterizing spots with gene expression vectors attached. Also, data augmentation is performed to construct a corrupted neighbor graph. Then, a two-layer GAT is adopted to encode the input graph into a low-dimensional embedding, and the decoder with two linear layers is utilized to reconstruct the gene expression matrix. To optimize the low-dimensional embedding further, the self-supervised contrastive learning can be employed to capture the spatial context information. Finally, the reconstructed gene expression data are clustered by Mclust method to achieve the recognition of spatial domain of ST data .

## Requirements
You'll need to install the following packages in order to run the codes.
* python>=3.8
* torch>=1.8.0
* cudnn>=10.2
* numpy==1.22.3
* scanpy==1.9.1
* anndata==0.8.0
* rpy2==3.4.1
* pandas==1.4.2
* scipy==1.8.1
* scikit-learn==1.1.1
* tqdm==4.64.0
* matplotlib==3.4.2
* R>=4.0.3

## Getting started
See run.py and Tutorial for GAAEST.ipynb

## Software dependencies
scanpy

pytorch

pyG