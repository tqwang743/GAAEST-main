import os
import torch
import pandas as pd
import scanpy as sc
import random
import numpy as np
from sklearn import metrics
import multiprocessing as mp
from GAAEST import GAAEST
import matplotlib.pyplot as plt

##[1] Basic settings
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
save_model_file = 'D:\code\GAAEST-main\weights.pth'
# the location of R, which is necessary for mclust algorithm. Please replace the path below with local R installation path
os.environ['R_HOME'] = 'D:\R\R-4.2.2'
n_clusters = 7
dataset = '151673'
seed = 41
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


##[2] read data
file_fold = 'D:\code\GAAEST-main\Data/' + str(dataset) #please replace 'file_fold' with the download path
adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
adata.var_names_make_unique()

##[3] train model
# define model
model = GAAEST.GAAEST(adata, device=device,save_model_file=save_model_file)
# train model
adata = model.train()

##[4] clustering
from GAAEST.utils import clustering
radius = 35
tool = 'mclust' # mclust, leiden, and louvain
if tool == 'mclust':
   clustering(adata, n_clusters, radius=radius, method=tool, refinement=True) # For DLPFC dataset, we use optional refinement step.
elif tool in ['leiden', 'louvain']:
   clustering(adata, n_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01, refinement=False)

##[5] add ground_truth
df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
df_meta_layer = df_meta['layer_guess']
adata.obs['ground_truth'] = df_meta_layer.values
# filter out NA nodes
adata = adata[~pd.isnull(adata.obs['ground_truth'])]

##[6] calculate metric
ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth'])
NMI = metrics.normalized_mutual_info_score(adata.obs['domain'], adata.obs['ground_truth'])
AMI = metrics.adjusted_mutual_info_score(adata.obs['domain'], adata.obs['ground_truth'])
FM = metrics.fowlkes_mallows_score(adata.obs['domain'], adata.obs['ground_truth'])
adata.uns['ARI'] = ARI
adata.uns['NMI'] = NMI
adata.uns['AMI'] = AMI
adata.uns['FM'] = FM

print('Dataset:', dataset)
print('ARI:', ARI)
print('NMI:', NMI)
print('AMI:', AMI)
print('FM:', FM)

##[7] plotting spatial clustering result
#Spatial domian recognition
sc.pl.spatial(adata,
              img_key="hires",
              color=["ground_truth", "domain"],
              title=["Ground truth", "ARI=%.4f"%ARI+" NMI=%.4f"%NMI+" AMI=%.4f"%AMI+" FM=%.4f"%FM],
              show=True)

#UMAP
sc.pp.neighbors(adata, use_rep='emb')
sc.tl.umap(adata)
used_adata = adata[adata.obs['ground_truth']!='nan',]
#PAGA
sc.tl.paga(used_adata, groups='domain')
plt.rcParams["figure.figsize"] = (4,3)
sc.pl.paga_compare(used_adata, legend_fontsize=10, frameon=False, size=20,
                   title=dataset+'_GAAEST', legend_fontoutline=2,threshold=0.3, show=True)