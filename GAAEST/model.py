import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from .GAT import GATConv
from.preprocess import fix_seed

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
    
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum 
          
        return F.normalize(global_emb, p=2, dim=1) 

def buildNetwork(layers, input_dim,activation="relu", dropout=0.):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        net.append(nn.BatchNorm1d(layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        elif activation=="elu":
            net.append(nn.ELU())
        elif activation=="lrelu":
            net.append(nn.LeakyReLU(negative_slope=0.2))
        if dropout > 0:
            net.append(nn.Dropout(p=dropout))
        net.append(nn.Linear(layers[-1],input_dim))
    return nn.Sequential(*net)   

class Encoder(Module):
     def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu, random_seed=41):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.random_seed = random_seed
        fix_seed(self.random_seed)
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        self.disc = Discriminator(self.out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        self.random_seed = random_seed
        in_dim = self.in_features 
        num_hidden = 256
        out_dim = self.out_features
        fix_seed(self.random_seed)
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,dropout=0, add_self_loops=False, bias=False)#初始化encoder的第一个GAT层
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,dropout=0, add_self_loops=False, bias=False)#初始化encoder的第二个GAT层
        self.decoder = buildNetwork([self.out_features]+[num_hidden], input_dim=self.in_features,activation='elu', dropout=0.)###【修改】

     def forward(self, feat, feat_a, edge_index):
        h1 = F.elu(self.conv1(feat, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        h3 = self.decoder(h2)
        

        h1_a = F.elu(self.conv1(feat_a, edge_index))
        h2_a = self.conv2(h1_a, edge_index, attention=False)
        h3_a =self.decoder(h2_a)

        fix_seed(self.random_seed)

        g = self.read(h2, self.graph_neigh)
        g = self.sigm(g)

        g_a = self.read(h2_a, self.graph_neigh)
        g_a = self.sigm(g_a)

        ret = self.disc(g, h2, h2_a)
        ret_a = self.disc(g_a, h2_a, h2)

        return h2, h3 , ret, ret_a, h2, h2_a

    
class Encoder_sparse(Module):
    """
    Sparse version of Encoder
    """

    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu, random_seed=41):
        super(Encoder_sparse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.random_seed = random_seed
        fix_seed(self.random_seed)
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        self.disc = Discriminator(self.out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        self.random_seed = random_seed
        in_dim = self.in_features
        num_hidden = 256
        out_dim = self.out_features
        fix_seed(self.random_seed)
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False, dropout=0, add_self_loops=False,
                             bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False, dropout=0, add_self_loops=False,
                             bias=False)
        self.decoder = buildNetwork([self.out_features] + [num_hidden], input_dim=self.in_features, activation='elu',
                                    dropout=0.)

    def forward(self, feat, feat_a, edge_index):
        fix_seed(self.random_seed)
        h1 = F.elu(self.conv1(feat, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        h3 = self.decoder(h2)

        h1_a = F.elu(self.conv1(feat_a, edge_index))
        h2_a = self.conv2(h1_a, edge_index, attention=False)
        h3_a = self.decoder(h2_a)

        g = self.read(h2, self.graph_neigh)
        g = self.sigm(g)

        g_a = self.read(h2_a, self.graph_neigh)
        g_a = self.sigm(g_a)

        ret = self.disc(g, h2, h2_a)
        ret_a = self.disc(g_a, h2_a, h2)

        return h2, h3, ret, ret_a, h2, h2_a

