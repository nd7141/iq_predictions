import pandas as pd
import os
from collections import defaultdict as ddict
import numpy as np
import networkx as nx

class Dataset:
    def __init__(self, folder, target_file, human2data=None):
        self.folder = folder
        self.target_file = target_file
        self.human2data = human2data

    def get_graphs_path(self, ext='csv'):
        return [self.folder + '/' + fn for fn in os.listdir(self.folder) if fn.endswith(ext)]

    def get_labels(self):
        target = pd.read_csv(self.target_file)
        target = target[~target.pcrvtot.isna()]
        return dict(zip(target.twinid, target.pcrvtot))

    def get_matrices(self, sep=','):
        human2iq = self.get_labels()
        human2matrices = ddict(list)
        for p in os.listdir(self.folder):
            human = int(p.split('_')[0])
            if human in human2iq:
                human2matrices[human].append(np.loadtxt(self.folder + '/' + p, delimiter=sep))
        return human2matrices

    def matrix_to_graph_threshold(self, A, percentiles=range(10, 91, 10)):
        flattened = A.flatten()
        flattened_pos = flattened[flattened > 0]
        graphs = []
        for p in percentiles:
            B = A.copy()
            t = np.percentile(flattened_pos, p)
            B[B < t] = 0
            graphs.append(nx.from_numpy_matrix(B))
        return graphs

    def matrix_to_graph_spanner(self, A, stretch, rep=10, weight='weight', seed=None):
        G = nx.from_numpy_matrix(A)
        return [nx.sparsifiers.spanner(G, stretch, weight=weight, seed=seed) for _ in range(rep)]

    def get_graph_data(self, method='threshold', percentiles=range(10, 91, 10), stretch=5, rep=10, weight='weight',
                       seed=None):
        human2iq = self.get_labels()
        human2matrices = self.get_matrices()

        human2data = ddict(list)
        for human in human2matrices:
            label = human2iq[human]
            for matrix in human2matrices[human]:
                if method == 'threshold':
                    graphs = self.matrix_to_graph_threshold(matrix, percentiles)
                elif method == 'spanner':
                    graphs = self.matrix_to_graph_spanner(matrix, stretch=stretch, rep=10, weight='weight', seed=None)
                human2data[human].extend(list(zip(graphs, [label] * len(graphs))))
        self.human2data = human2data
