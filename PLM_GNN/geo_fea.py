import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
import esm
from esm import FastaBatchedDataset



def get_geo_feat(X, edge_index):
    """
    get geometric node features and edge features
    """
    pos_embeddings = _positional_embeddings(edge_index)
    node_angles = _get_angle(X)
    node_dist, edge_dist = _get_distance(X, edge_index)
    node_direction, edge_direction, edge_orientation = _get_direction_orientation(X, edge_index)

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

# def _get_distance(X, edge_index):
#     """
#     get the distance features
#     """
#     atom_N = X[:,0]  # [L, 3]
#     atom_Ca = X[:,1]
#     atom_C = X[:,2]
#     atom_O = X[:,3]
#     atom_R = X[:,4]

#     node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C', 'R-N', 'R-Ca', "R-C", 'R-O']
#     node_dist = []
#     for pair in node_list:
#         atom1, atom2 = pair.split('-')
#         E_vectors = vars()['atom_' + atom1] - vars()['atom_' + atom2]
#         rbf = _rbf(E_vectors.norm(dim=-1))
#         node_dist.append(rbf)
#     node_dist = torch.cat(node_dist, dim=-1) # dim = [N, 10 * 16]

#     atom_list = ["N", "Ca", "C", "O", "R"]
#     edge_dist = []
#     for atom1 in atom_list:
#         for atom2 in atom_list:
#             E_vectors = vars()['atom_' + atom1][edge_index[0]] - vars()['atom_' + atom2][edge_index[1]]
#             rbf = _rbf(E_vectors.norm(dim=-1))
#             edge_dist.append(rbf)
#     edge_dist = torch.cat(edge_dist, dim=-1) # dim = [E, 25 * 16]

#     return node_dist, edge_dist


import torch

def _get_distance(X, edge_index):
    """
    get the distance features
    """
    # 检查 X 和 edge_index 的合法性
    if torch.isnan(X).any() or torch.isinf(X).any():
        raise ValueError("X contains NaN or Inf values")
    
    if torch.isnan(edge_index).any() or torch.isinf(edge_index).any():
        raise ValueError("edge_index contains NaN or Inf values")
    
    # 确保 edge_index 中的索引在 X 的范围内
    N = X.shape[0]
    if (edge_index >= N).any() or (edge_index < 0).any():
        
        raise IndexError(f"edge_index contains out-of-range indices. Valid range is [0, {N-1}], but got {edge_index}")
    
    # 提取原子特征
    atom_N = X[:, 0]  # [N, 3]
    atom_Ca = X[:, 1]
    atom_C = X[:, 2]
    atom_O = X[:, 3]
    atom_R = X[:, 4]

    # 检查提取的原子特征的合法性
    for atom_name, atom in zip(['N', 'Ca', 'C', 'O', 'R'], [atom_N, atom_Ca, atom_C, atom_O, atom_R]):
        if torch.isnan(atom).any() or torch.isinf(atom).any():
            raise ValueError(f"atom_{atom_name} contains NaN or Inf values")

    node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C', 'R-N', 'R-Ca', "R-C", 'R-O']
    node_dist = []
    for pair in node_list:
        atom1, atom2 = pair.split('-')
        E_vectors = vars()['atom_' + atom1] - vars()['atom_' + atom2]
        
        # 检查 E_vectors 是否有 NaN 或 Inf 值
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
            
            # 检查 E_vectors 是否有 NaN 或 Inf 值
            if torch.isnan(E_vectors).any() or torch.isinf(E_vectors).any():
                raise ValueError(f"E_vectors for {atom1}-{atom2} contains NaN or Inf values")
            
            rbf = _rbf(E_vectors.norm(dim=-1))
            edge_dist.append(rbf)
    edge_dist = torch.cat(edge_dist, dim=-1)  # dim = [E, 25 * 16]

    return node_dist, edge_dist

def _rbf(D):
    # 定义 RBF 参数
    D_mu = torch.linspace(0.0, 20.0, 16).to(D.device)
    D_sigma = D_mu[1] - D_mu[0]
    
    # 检查 D 是否有 NaN 或 Inf 值
    if torch.isnan(D).any() or torch.isinf(D).any():
        raise ValueError("D contains NaN or Inf values")
    
    D_expand = D.unsqueeze(-1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def _rbf(D):
    # 定义 RBF 参数
    D_mu = torch.linspace(0.0, 20.0, 16).to(D.device)
    D_sigma = D_mu[1] - D_mu[0]
    
    # 检查 D 是否有 NaN 或 Inf 值
    if torch.isnan(D).any() or torch.isinf(D).any():
        raise ValueError("D contains NaN or Inf values")
    
    D_expand = D.unsqueeze(-1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


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

    return Q

