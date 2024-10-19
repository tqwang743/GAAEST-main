import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
import sklearn.cluster
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

def cluster(data, k, temp, num_iter, init, cluster_temp):
    cuda0 = torch.cuda.is_available()

    if cuda0:
        mu = init.cuda()
        data = data.cuda()
        cluster_temp = cluster_temp.cuda()
    else:
        mu = init

    data = data / (data.norm(dim=1)[:, None] + 1e-6)  # prevent zero-division loss with 1e-6
    for t in range(num_iter):

        mu = mu / (mu.norm(dim=1)[:, None] + 1e-6) #prevent zero-division with 1e-6

        dist = torch.mm(data, mu.transpose(0,1))

        # cluster responsibilities via softmax
        r = F.softmax(cluster_temp*dist, dim=1)
        # total responsibility of each cluster
        cluster_r = r.sum(dim=0)
        # mean of points in each cluster weighted by responsibility
        cluster_mean = r.t() @ data
        # update cluster means
        new_mu = torch.diag(1/cluster_r) @ cluster_mean
        mu = new_mu
    
    r = F.softmax(cluster_temp*dist, dim=1)
    return mu, r


class Clusterator(nn.Module):
    '''
    The ClusterNet architecture. The first step is a 2-layer GCN to generate embeddings.
    The output is the cluster means mu and soft assignments r, along with the 
    embeddings and the the node similarities (just output for debugging purposes).
    
    The forward pass inputs are x, a feature matrix for the nodes, and adj, a sparse
    adjacency matrix. The optional parameter num_iter determines how many steps to 
    run the k-means updates for.
    '''
    def __init__(self, nout, K):
        super(Clusterator, self).__init__()
        
        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.nout = nout
        self.init =  torch.rand(self.K, nout)
        
    def forward(self, embeds, cluster_temp, num_iter=10):
        mu_init, _ = cluster(embeds, self.K, 1, num_iter, cluster_temp = torch.tensor(cluster_temp), init = self.init)
        #self.init = mu_init.clone().detach()
        mu, r = cluster(embeds, self.K, 1, 1, cluster_temp = torch.tensor(cluster_temp), init = mu_init.clone().detach())
        
        return mu, r

class Discriminator_cluster(nn.Module):##local-context discriminator Dc##
    def __init__(self, n_in, n_h , n_nb , num_clusters ):
        super(Discriminator_cluster, self).__init__()
        
        self.n_nb = n_nb
        self.n_h = n_h
        self.num_clusters=num_clusters

    def forward(self, c, c2, h_0, h_pl, h_mi, S, s_bias1=None, s_bias2=None):
        
        c_x = c.expand_as(h_0)
        
        sc_1 =torch.bmm(h_pl.view(self.n_nb, 1, self.n_h), c_x.view(self.n_nb, self.n_h, 1))
        sc_2 = torch.bmm(h_mi.view(self.n_nb, 1, self.n_h), c_x.view(self.n_nb, self.n_h, 1))

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1,sc_2),0).view(1,-1)

        return logits