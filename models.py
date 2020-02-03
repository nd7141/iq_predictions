import networkx as nx
import numpy as np
import os
import pandas as pd
from scipy.stats import describe
from collections import defaultdict as ddict
import matplotlib.pyplot as plt
import random
import pickle

from torch_geometric.utils.convert import from_networkx
from torch_geometric.datasets import TUDataset
from torch_geometric.data import batch, Data, InMemoryDataset, DataLoader
import torch
import torch_geometric
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import MSELoss, L1Loss

from sklearn .model_selection import KFold


class GIN(torch.nn.Module):
    def __init__(self, input_dim, dim=32):
        super(GIN, self).__init__()

        nn1 = Sequential(Linear(input_dim, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x


class MotifEmbedding():
    def __init__(self, human2data):
        self.human2data = human2data

    def get_triples(self, G, k=3, n_samples=100):
        nodes = set(nx.nodes(G))
        triples = ddict(list)
        for node in G:
            for _ in range(n_samples):
                sample = random.sample(nodes - set([node]), k - 1)
                triples[node].append(sample)
        return triples

    def make_motifs3(self):
        graphs = [
            nx.Graph(),
            nx.Graph([(1, 2)]),
            nx.Graph([(1, 2), (2, 3)]),
            nx.Graph([(1, 2), (2, 3), (3, 1)])
        ]
        [g.add_nodes_from([1, 2, 3]) for g in graphs]
        return graphs

    def get_weight_triple(self, subgraph):
        return np.prod([e[2]['weight'] for e in subgraph.edges(data=True)]) ** (1. / 3)

    def compute_motif3_embedding(self, G, n_samples):
        triples = self.get_triples(G, 3, n_samples)
        motif3 = self.make_motifs3()
        embeddings = dict()
        for node in triples:
            node_embedding = np.zeros((len(motif3),))
            for triple in triples[node]:
                subgraph = nx.subgraph(G, triple + [node])
                for i, motif in enumerate(motif3):
                    if nx.is_isomorphic(motif, subgraph):
                        node_embedding[i] += self.get_weight_triple(subgraph)
                        break
            embeddings[node] = node_embedding
        return embeddings

    def compute_human_embedding(self, graphs, n_samples):
        human_embedding = np.zeros((4,))
        for G in graphs:
            embeddings = self.compute_motif3_embedding(G, n_samples)
            human_embedding += np.sum(list(embeddings.values()), 0)
        return human_embedding / len(graphs)

    def compute_human2embedding(self, n_samples=10):
        self.human2embedding = dict()
        for human in self.human2data:
            print(human)
            graphs = [g for g, _ in self.human2data[human]]
            self.human2embedding[human] = self.compute_human_embedding(graphs, n_samples)
            # break

    def save_embeddings(self):
        with open("human2embedding.pkl", 'wb') as f:
            pickle.dump(self.human2embedding, f)

    def load_embeddings(self):
        with open("human2embedding.pkl", 'rb') as f:
            self.human2embedding = pickle.load(f)

    def kfold_split(self, nsplits, batch_size=128):
        human2data = self.human2data
        humans = np.array(list(human2data.keys()))
        kfold = KFold(nsplits, shuffle=True)
        for train_ix, test_ix in kfold.split(humans):
            train_humans = humans[train_ix]
            test_humans = humans[test_ix]
            return train_humans, test_humans