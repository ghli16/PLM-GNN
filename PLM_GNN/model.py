import re
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils import data
import torch.utils.data as data
import torch.nn.functional as Fun
import torch_sparse
from torch_scatter import scatter_mean, scatter_add
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import (
    TransformerConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    radius_graph
)
from einops import rearrange
import esm
from esm import Alphabet, pretrained, FastaBatchedDataset
from PLM_GNN.module import TransformerLayer, MLPLayer
from PLM_GNN.trainer import set_seed
from data import *

def split_batch(x,batchid):
    x =  x.unsqueeze(0)
    unique_batch_ids = torch.unique(batchid)
    batchx = []
    for batch_id in unique_batch_ids:
        batch_indices = torch.nonzero(batchid == batch_id).squeeze()
        batchx.append(x[:,batch_indices])
    return batchx
        
class GNNLayer(nn.Module):
    """
    define GNN layer for subsequent computations
    """
    def __init__(self, num_hidden, dropout=0.2, num_heads=4):
        super(GNNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])

        self.attention = TransformerConv(in_channels=num_hidden, out_channels=int(num_hidden / num_heads), heads=num_heads, dropout = dropout, edge_dim = num_hidden, root_weight=False)
        self.PositionWiseFeedForward = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )
        self.edge_update = EdgeMLP(num_hidden, dropout)
        self.context = Context(num_hidden)

    def forward(self, h_V, edge_index, h_E, batch_id):
        dh = self.attention(h_V, edge_index, h_E)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.PositionWiseFeedForward(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        # update edge
        h_E = self.edge_update(h_V, edge_index, h_E)

        # context node update
        h_V = self.context(h_V, batch_id)

        return h_V, h_E


class EdgeMLP(nn.Module):
    """
    define MLP operation for edge updates
    """
    def __init__(self, num_hidden, dropout=0.2):
        super(EdgeMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(3*num_hidden, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, edge_index, h_E):
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W12(self.act(self.W11(h_EV)))
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E


class Context(nn.Module):
    def __init__(self, num_hidden):
        super(Context, self).__init__()

        self.V_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.Sigmoid()
                                )

    def forward(self, h_V, batch_id):
        c_V = scatter_mean(h_V, batch_id, dim=0)
        h_V = h_V * self.V_MLP_g(c_V[batch_id])
        return h_V


class Graph_encoder(nn.Module):
    """
    construct the graph encoder module
    """
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim,
                 seq_in=False, num_layers=4, drop_rate=0.2):
        super(Graph_encoder, self).__init__()

        self.seq_in = seq_in
        if self.seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim += 20
        
        self.node_embedding = nn.Linear(node_in_dim, hidden_dim, bias=True)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim, bias=True)
        self.norm_nodes = nn.BatchNorm1d(hidden_dim)
        self.norm_edges = nn.BatchNorm1d(hidden_dim)
        
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.layers = nn.ModuleList(
                GNNLayer(num_hidden=hidden_dim, dropout=drop_rate, num_heads=4)
            for _ in range(num_layers))


    def forward(self, h_V, edge_index, h_E, seq, batch_id):
        if self.seq_in and seq is not None:
            seq = self.W_s(seq)
            h_V = torch.cat([h_V, seq], dim=-1)

        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_E = self.W_e(self.norm_edges(self.edge_embedding(h_E)))

        for layer in self.layers:
            h_V, h_E = layer(h_V, edge_index, h_E, batch_id)
        
        return h_V

    
class Attention(nn.Module):
    """
    define the attention module
    """
    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = torch.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, input):  				# input.shape = (1, seq_len, input_dim)
        x = torch.tanh(self.fc1(input))  	# x.shape = (1, seq_len, dense_dim)
        x = self.fc2(x)  					# x.shape = (1, seq_len, attention_hops)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)  		# attention.shape = (1, attention_hops, seq_len)
        return attention

class GNN(nn.Module): 
    """
    construct the GraphEC-pH model
    """
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, num_layers, dropout, augment_eps, task,device,return_embedding=False):
        super(GNN, self).__init__()
        self.augment_eps = augment_eps
        self.device = device
        self.hidden_dim = hidden_dim
        self.return_embedding = return_embedding
        # define the encoder layer
        self.Graph_encoder = Graph_encoder(node_in_dim=node_input_dim, edge_in_dim=edge_input_dim, hidden_dim=hidden_dim, seq_in=False, num_layers=num_layers, drop_rate=dropout)

        # define the attention layer
        self.attention = Attention(hidden_dim,dense_dim=16,n_heads=4)
        
        self.input_block = nn.Sequential(
                                         nn.LayerNorm(node_input_dim, eps=1e-6)
                                        ,nn.Linear(node_input_dim, hidden_dim)
                                        ,nn.LeakyReLU()
                                        )
        self.hidden_block = []
        num_emb_layers = 2
        for i in range(num_emb_layers - 1):
            self.hidden_block.extend([
                                      nn.LayerNorm(hidden_dim, eps=1e-6)
                                     ,nn.Dropout(dropout)
                                     ,nn.Linear(hidden_dim, hidden_dim)
                                     ,nn.LeakyReLU()
                                     ])
            if i == num_emb_layers - 2:
                self.hidden_block.extend([nn.LayerNorm(hidden_dim, eps=1e-6)])
        self.hidden_block = nn.Sequential(*self.hidden_block)
        self.clf1 = nn.Linear(hidden_dim, hidden_dim)
        self.clf2 = nn.Linear(hidden_dim, 7)
        self.dropout = nn.Dropout(p=dropout)
        for p in self.parameters():
            if p.dim() > 1:
                set_seed(42)
                nn.init.xavier_uniform_(p)
                

    def forward(self, X, h_V, edge_index, seq, batch_id):

        if self.training and self.augment_eps > 0:
            set_seed(42)
            X = X + self.augment_eps * torch.randn_like(X)
            h_V = h_V + self.augment_eps * torch.randn_like(h_V)
        
        print(h_V.size())

        h_V_geo, h_E = get_geo_feat(X, edge_index)
        
        h_V = torch.cat([h_V, h_V_geo], dim=-1)

        h_V = self.Graph_encoder(h_V, edge_index, h_E, seq, batch_id) # [num_residue, hidden_dim]


        batchx = split_batch(h_V,batch_id) 
        feature_embedding = torch.tensor([]).to(self.device)
        for h_vi in batchx:
            att = self.attention(h_vi) 
            h_vi = att @ h_vi 
            h_vi = torch.sum(h_vi,1) 
            feature_embedding = torch.cat((feature_embedding,h_vi),dim=0) 
        h_V = feature_embedding 

        emb = h_V
        if self.return_embedding : 
            return emb
        else :
            emb = self.clf1(emb)
            emb = torch.relu(emb)
            emb = self.dropout(emb)
            emb = self.clf2(emb)
            return emb




    
class CNN_Transformer(nn.Module):

    def __init__(self, emb_dim, repr_layer, num_layers, heads,
                 hid_dim=256, dropout_rate=0.4, num_classes=7, attn_dropout=0.05, return_embedding=False, return_attn=False):

        super().__init__()
        self.pretrained_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.padding_idx = alphabet.padding_idx
        self.dim = hid_dim
        self.repr_layer = repr_layer
        self.num_layers = num_layers
        self.conv1 = nn.Conv1d(emb_dim, hid_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv1d(hid_dim, hid_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv = nn.Conv1d(emb_dim, hid_dim, 1, 1, bias=False)
        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(hid_dim, heads, dropout_rate, attn_dropout)
                for _ in range(self.num_layers)
            ]
        )
        self.clf = nn.Linear(hid_dim, num_classes)

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.return_embedding = return_embedding
        self.return_attn = return_attn

    def forward(self, strs, toks):

        toks = toks
        padding_mask = (toks != self.padding_idx)[:, 1:-1]

        out = self.pretrained_model(
            toks, repr_layers=[self.repr_layer], return_contacts=False)
        x = out["representations"][self.repr_layer][:, 1:-1, :]  # (bs, seq_len, emb_dim)
        x = x * padding_mask.unsqueeze(-1).type_as(x)

        x = rearrange(x, 'b n d -> b d n')
        x = self.conv1(x)
        x = self.conv2(x)        
        x = rearrange(x, 'b d n -> b n d')
        x = self.fc1(x)
        batch = toks.shape[0]
        for layer in self.layers:
            x, attn = layer(
                x, mask=padding_mask.unsqueeze(1).unsqueeze(2)
            )

        out = torch.cat([x[i, :len(strs[i]) + 1].mean(0).unsqueeze(0)
                        for i in range(batch)], dim=0) # average pooling along the sequence

        if self.return_embedding:
            return out
        else:
            logits = self.clf(out)
            if self.return_attn:
                return logits, attn
            else:
                return logits



        

class Conbine_two(nn.Module):
    def __init__(self, emb_dim=1280, repr_layer=33, num_layers=1, heads=4,
                 hid_dim=256, dropout_rate=0.4, num_classes=7, attn_dropout=0.05, return_embedding1=False, return_embedding2=False,return_attn=False,
                 node_input_dim=1024 + 184, edge_input_dim=450, augment_eps=0.15, task='Sol', device=0):
        super(Conbine_two, self).__init__()
        
        self.esm_ = CNN_Transformer(emb_dim, repr_layer, num_layers, heads, hid_dim, dropout_rate, num_classes, attn_dropout, return_embedding1, return_attn)
        
        self.geo = GNN(node_input_dim, edge_input_dim, hidden_dim=hid_dim, num_layers=num_layers, dropout=dropout_rate, augment_eps=augment_eps, task=task, device=device, return_embedding=return_embedding1)
        
        self.return_embedding2 = return_embedding2


        self.clf1 = nn.Linear(hid_dim, num_classes)
        
    def forward(self, strs, toks, X, h_V, edge_index, seq, batch_id):
        esm_f = self.esm_(strs, toks)
        
        geo_f = self.geo(X, h_V, edge_index, seq, batch_id)
        
        fused = esm_f + geo_f
        out = self.clf1(fused)
        if self.return_embedding2:
            return fused,esm_f,geo_f
        else:
            return out



        
        
def get_geo_feat(X, edge_index):
    """
    get geometric node features and edge features
    """
    print(X.size())
    pos_embeddings = _positional_embeddings(edge_index)
    node_angles = _get_angle(X)
    node_dist, edge_dist = _get_distance(X, edge_index)
    node_direction, edge_direction, edge_orientation = _get_direction_orientation(X, edge_index)
    print('node_angles:',node_angles.size())
    print('node_dist:',node_dist.size())
    print('node_direction:',node_direction.size())
    geo_node_feat = torch.cat([node_angles, node_dist, node_direction], dim=-1)
    geo_edge_feat = torch.cat([pos_embeddings, edge_orientation, edge_dist, edge_direction], dim=-1)

    return geo_node_feat, geo_edge_feat


def _positional_embeddings(edge_index, num_embeddings=16):
    """
    get the positional embeddings
    """
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=edge_index.device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    PE = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return PE

def _get_angle(X, eps=1e-7):
    """
    get the angle features
    """
    # psi, omega, phi
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
    D = F.pad(D, [1, 2]) # This scheme will remove phi[0], psi[-1], omega[-1]
    D = torch.reshape(D, [-1, 3])
    dihedral = torch.cat([torch.cos(D), torch.sin(D)], 1)

    # alpha, beta, gamma
    cosD = (u_2 * u_1).sum(-1) # alpha_{i}, gamma_{i}, beta_{i+1}
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.acos(cosD)
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    bond_angles = torch.cat((torch.cos(D), torch.sin(D)), 1)

    node_angles = torch.cat((dihedral, bond_angles), 1)
    
    # print('node_angles',node_angles.size())
    
    return node_angles # dim = 12

def _rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        
    return RBF

    # D_sigma = D_mu[1] - D_mu[0]

import torch

def _get_distance(X, edge_index):
    """
    get the distance features
    """
    if torch.isnan(X).any() or torch.isinf(X).any():
        raise ValueError("X contains NaN or Inf values")
    
    if torch.isnan(edge_index).any() or torch.isinf(edge_index).any():
        raise ValueError("edge_index contains NaN or Inf values")
    
    N = X.shape[0]
    if (edge_index >= N).any() or (edge_index < 0).any():
        
        raise IndexError(f"edge_index contains out-of-range indices. Valid range is [0, {N-1}], but got {edge_index}")
    
    # 提取原子特征
    atom_N = X[:, 0]  # [N, 3]
    atom_Ca = X[:, 1]
    atom_C = X[:, 2]
    atom_O = X[:, 3]
    atom_R = X[:, 4]

    for atom_name, atom in zip(['N', 'Ca', 'C', 'O', 'R'], [atom_N, atom_Ca, atom_C, atom_O, atom_R]):
        if torch.isnan(atom).any() or torch.isinf(atom).any():
            raise ValueError(f"atom_{atom_name} contains NaN or Inf values")

    node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C', 'R-N', 'R-Ca', "R-C", 'R-O']
    node_dist = []
    for pair in node_list:
        atom1, atom2 = pair.split('-')
        E_vectors = vars()['atom_' + atom1] - vars()['atom_' + atom2]
        
        if torch.isnan(E_vectors).any() or torch.isinf(E_vectors).any():
            raise ValueError(f"E_vectors for {pair} contains NaN or Inf values")
        
        rbf = _rbf(E_vectors.norm(dim=-1))
        node_dist.append(rbf)
    node_dist = torch.cat(node_dist, dim=-1)  # dim = [N, 10 * 16]

    atom_list = ["N", "Ca", "C", "O", "R"]
    edge_dist = []
    for atom1 in atom_list:
        for atom2 in atom_list:
            E_vectors = vars()['atom_' + atom1][edge_index[0]] - vars()['atom_' + atom2][edge_index[1]]
            
            if torch.isnan(E_vectors).any() or torch.isinf(E_vectors).any():
                raise ValueError(f"E_vectors for {atom1}-{atom2} contains NaN or Inf values")
            
            rbf = _rbf(E_vectors.norm(dim=-1))
            edge_dist.append(rbf)
    edge_dist = torch.cat(edge_dist, dim=-1)  # dim = [E, 25 * 16]

    return node_dist, edge_dist


def _get_direction_orientation(X, edge_index): # N, CA, C, O, R
    """
    get the direction features
    """
    X_N = X[:,0]  # [L, 3]
    X_Ca = X[:,1]
    X_C = X[:,2]
    u = F.normalize(X_Ca - X_N, dim=-1)
    v = F.normalize(X_C - X_Ca, dim=-1)
    b = F.normalize(u - v, dim=-1)
    n = F.normalize(torch.cross(u, v), dim=-1)
    local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1) # [L, 3, 3] (3 column vectors)

    node_j, node_i = edge_index

    t = F.normalize(X[:, [0,2,3,4]] - X_Ca.unsqueeze(1), dim=-1) # [L, 4, 3]
    node_direction = torch.matmul(t, local_frame).reshape(t.shape[0], -1) # [L, 4 * 3]

    t = F.normalize(X[node_j] - X_Ca[node_i].unsqueeze(1), dim=-1) # [E, 5, 3]
    edge_direction_ji = torch.matmul(t, local_frame[node_i]).reshape(t.shape[0], -1) # [E, 5 * 3]
    t = F.normalize(X[node_i] - X_Ca[node_j].unsqueeze(1), dim=-1) # [E, 5, 3]
    edge_direction_ij = torch.matmul(t, local_frame[node_j]).reshape(t.shape[0], -1) # [E, 5 * 3]
    edge_direction = torch.cat([edge_direction_ji, edge_direction_ij], dim = -1) # [E, 2 * 5 * 3]

    r = torch.matmul(local_frame[node_i].transpose(-1,-2), local_frame[node_j]) # [E, 3, 3]
    edge_orientation = _quaternions(r) # [E, 4]

    return node_direction, edge_direction, edge_orientation

def _quaternions(R):
    """ Convert a batch of 3D rotations [R] to quaternions [Q]
        R [N,3,3]
        Q [N,4]
    """
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
          Rxx - Ryy - Rzz,
        - Rxx + Ryy - Rzz,
        - Rxx - Ryy + Rzz
    ], -1)))
    _R = lambda i,j: R[:,i,j]
    signs = torch.sign(torch.stack([
        _R(2,1) - _R(1,2),
        _R(0,2) - _R(2,0),
        _R(1,0) - _R(0,1)
    ], -1))
    xyz = signs * magnitudes
    # The relu enforces a non-negative trace
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)
    # print('_quaternions',Q.size())
    return Q

