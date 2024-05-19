import torch
from .preprocess import preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, \
    construct_interaction_KNN, add_contrastive_label, get_feature, permutation, fix_seed
import time
import random
import numpy as np
from .model import Encoder, Encoder_sparse
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
from .layers import Clusterator, Discriminator_cluster
from .utils import Transfer_pytorch_Data
import scipy.sparse as sp
from torch.autograd import Variable

class GAAEST():
    def __init__(self,
                 adata,
                 device=torch.device('cpu'),
                 learning_rate=0.001,
                 weight_decay=0.00,
                 epochs=600,
                 dim_input=3000,
                 dim_output=64,
                 random_seed=41,
                 alpha=10,
                 beta=1,
                 gama=1,
                 lane=1,
                 theta=0.1,
                 lamda1=10,
                 lamda2=1,
                 datatype='10X',
                 save_model_file='D:\code\GAAEST\weights.pth'
                 ):

        self.adata = adata.copy()
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_output = dim_output
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.gama = gama
        self.lane = lane
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.datatype = datatype
        self.tau = 0.5
        self.save_model_file = save_model_file

        fix_seed(self.random_seed)
        self.fc1 = torch.nn.Linear(self.dim_output, 128).to(self.device)
        self.fc2 = torch.nn.Linear(128, self.dim_output).to(self.device)

        if 'highly_variable' not in adata.var.keys():
            preprocess(self.adata)

        if 'adj' not in adata.obsm.keys():
          if self.datatype in ['Stereo', 'Slide']:
               construct_interaction_KNN(self.adata)
          else:
               construct_interaction(self.adata)

        if 'label_CSL' not in adata.obsm.keys():
            add_contrastive_label(self.adata)

        if 'feat' not in adata.obsm.keys():
            get_feature(self.adata)

        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        self.adj = self.adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(
            self.device)

        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output

        if self.datatype in ['Stereo', 'Slide']:
            # using sparse
            print('Building sparse matrix ...')
            self.adj = preprocess_adj_sparse(self.adj).to(self.device)
        else:
            # standard version
            fix_seed(self.random_seed)
            self.adj = preprocess_adj(self.adj)
            self.adj = torch.FloatTensor(self.adj).to(self.device)

        if 'Spatial_Net' not in self.adata.uns.keys():
            raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
        self.data = Transfer_pytorch_Data(self.adata).to(self.device)

    def train(self):
        fix_seed(self.random_seed)
        if self.datatype in ['Stereo', 'Slide']:
            self.model = Encoder_sparse(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        else:
            self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        self.loss_gf = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate,
                                          weight_decay=self.weight_decay)
        fix_seed(self.random_seed)
  
        print('Begin to train ST data...')
        self.model.train()

        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            fix_seed(self.random_seed)
            self.features_a = permutation(self.features)
            self.hiden_feat, self.emb, ret, ret_a, Zs, Zs_a = self.model(self.features, self.features_a,
                                                                         self.data.edge_index)

            ####  ①global featureloss####
            self.loss_gf_1 = self.loss_gf(ret, self.label_CSL)
            self.loss_gf_2 = self.loss_gf(ret_a, self.label_CSL)

            ####  ②local loaction loss####
            Zs = self.projection(Zs)  # P(Hu)
            Zs_a = self.projection(Zs_a)  # P(Hv)
            self.loss_ll = self.cont_ll(Zs, Zs_a)

            ####   ③gene reconstruction loss####
            self.loss_feat = F.mse_loss(self.features, self.emb)

            ####  ④contextual feature loss####
            ret_cf = self.cf_forward(Zs, Zs_a, 100)
            lbl_1 = torch.ones(1, ret_cf.shape[1] // 2)
            lbl_2 = torch.zeros(1, ret_cf.shape[1] // 2)
            lbl = torch.cat((lbl_1, lbl_2), 1).to(self.device)
            self.cont_cf = self.cont_bxent(lbl, ret_cf)


            ####  【total loss】 ####
            loss = self.alpha * self.loss_feat + self.beta * (
                        self.loss_gf_1 + self.loss_gf_2) + self.gama * self.loss_ll + self.lane * self.cont_cf
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Optimization finished for ST data!")
        self.save_model(save_model_file=self.save_model_file)

        with torch.no_grad():
            self.model.eval()
            if self.datatype in ['Stereo', 'Slide']:
                self.emb_rec = self.model(self.features, self.features_a, self.data.edge_index)[1]
                self.emb_rec = F.normalize(self.emb_rec, p=2, dim=1).detach().cpu().numpy()
            else:
                fix_seed(self.random_seed)
                self.emb_rec = self.model(self.features, self.features_a, self.data.edge_index)[
                    1].detach().cpu().numpy()
            self.adata.obsm['emb'] = self.emb_rec

            return self.adata

    def save_model(
        self,
        save_model_file
        ):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model(
        self,
        save_model_file
        ):
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % save_model_file)


