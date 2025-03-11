import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from esm import FastaBatchedDataset
import torch_geometric
import torch
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
import re

class SequenceData(Dataset):

    def __init__(self, fasta_path, transform=None):
        self.fasta_path = fasta_path
        self.transform = transform
        self.check_dataset()

    def check_dataset(self):
        tempdataset = FastaBatchedDataset.from_file(self.fasta_path)
        sequence_labels = [label[:7] for label in tempdataset.sequence_labels]
        sequence_strs = tempdataset.sequence_strs
        self.sequence_labels = np.array(sequence_labels)
        self.sequence_strs = np.array(sequence_strs)
            
    def __getitem__(self, idx):
        label = self.sequence_labels[idx]
        seq_str = self.sequence_strs[idx]
        if self.transform is not None:
            label = self.transform(label)
        return label, seq_str

    def __len__(self):
        return len(self.sequence_labels)


class GraphData(torch.utils.data.Dataset):
    """
    构建数据集
    """
    def __init__(self, fasta_file, args):
        self.dataset = {}
        self.IDs = []
        self.sequence_labels = []
        self.radius = args.r
        self.letter_to_num = {
            'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
            'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
            'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
            'N': 2, 'Y': 18, 'M': 12
        }
        self.label_dict = {
            'VFC0001': 1, 'VFC0086': 2, 'VFC0204': 3, 'VFC0235': 4,
            'VFC0258': 5, 'VFC0271': 6, 'VFC0272': 0
        }

        with open(fasta_file, "r") as f:
            lines = f.readlines()

        name = None
        for line in lines:
            if line[0] == ">":
                name = line[1:].strip()
                self.IDs.append(name)
                
            else:
                sequence = line.strip()
                self.dataset[name] = sequence
                label = self.label_dict.get(name[:7], -1)
                self.sequence_labels.append(label)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        return self._featurize_graph(idx)

    def _featurize_graph(self, idx):
        name = self.IDs[idx]
        with torch.no_grad():
 
            X = torch.load('../../../Features_xyz/' + name + ".tensor")

            seq = torch.tensor([self.letter_to_num[aa] for aa in self.dataset[name]], dtype=torch.long)

            prottrans_feat = torch.load(open('../../../protf_test/' + name + ".tensor", 'rb'))

            pre_computed_node_feat = prottrans_feat
            X_ca = X[:, 1]

            edge_index = radius_graph(X_ca, r=self.radius, loop=True, max_num_neighbors=1000, num_workers=0)

        graph_data = Data(
            name=name,
            seq=seq,
            X=X,
            node_feat=pre_computed_node_feat,
            edge_index=edge_index,
            label=self.sequence_labels[idx],
            strs=self.dataset[name]
        )
        
        # return graph_data,self.dataset[name],name
        return graph_data
    
    
#TSS

# class GraphData(torch.utils.data.Dataset):
#     """
#     构建数据集
#     """
#     def __init__(self, fasta_file, args):
#         self.dataset = {}
#         self.IDs = []
#         self.sequence_labels = []
#         self.radius = args.r
#         self.letter_to_num = {
#             'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
#             'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
#             'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
#             'N': 2, 'Y': 18, 'M': 12
#         }
#         self.label_dict = {'T2SS': 0, 'T3SS': 1, 'T4SS': 2, 'T6SS': 3, 'T7SS': 4}

#         with open(fasta_file, "r") as f:
#             lines = f.readlines()

#         name = None
#         for line in lines:
#             if line[0] == ">":
#                 name = line[1:].strip()
#                 self.IDs.append(name)
                
#             else:
#                 sequence = line.strip()
#                 self.dataset[name] = sequence
#                 label = self.label_dict.get(name[:4], -1)
#                 self.sequence_labels.append(label)

#     def __len__(self):
#         return len(self.IDs)

#     def __getitem__(self, idx):
#         return self._featurize_graph(idx)

#     def _featurize_graph(self, idx):
#         name = self.IDs[idx]
#         with torch.no_grad():
#             X = torch.load('../../../TSS_xyz/' +  name + ".tensor")
#             seq = torch.tensor([self.letter_to_num[aa] for aa in self.dataset[name]], dtype=torch.long)

#             prottrans_feat = torch.load(open('../../../TSS_fea/' + name + ".tensor", 'rb'))
#             pre_computed_node_feat = prottrans_feat
#             X_ca = X[:, 1]
#             edge_index = radius_graph(X_ca, r=self.radius, loop=True, max_num_neighbors=1000, num_workers=0)
            
#         graph_data = Data(
#             name=name,
#             seq=seq,
#             X=X,
#             node_feat=pre_computed_node_feat,
#             edge_index=edge_index,
#             label=self.sequence_labels[idx],
#             strs=self.dataset[name]
#         )
        
#         return graph_data



