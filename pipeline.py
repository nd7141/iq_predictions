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

import networkx as nx
import numpy as np
import os
import pandas as pd
from scipy.stats import describe
from collections import defaultdict as ddict
import matplotlib.pyplot as plt
import random
import pickle

class Pipeline():
    def __init__(self, model, human2data):
        self.model = model
        self.human2data = human2data

        self.mse = MSELoss()
        self.mae = L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def from_networkx(self, G, human, y=None, edge_name=None):
        G = nx.convert_node_labels_to_integers(G)
        G = G.to_directed() if not nx.is_directed(G) else G
        edge_index = torch.tensor(list(G.edges)).t().contiguous()

        data = {}

        for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
            for key, value in feat_dict.items():
                data[key] = [value] if i == 0 else data[key] + [value]

        for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
            for key, value in feat_dict.items():
                if edge_name is not None:
                    key = edge_name
                data[key] = [value] if i == 0 else data[key] + [value]

        degrees = nx.degree(G, weight='weight')
        data['x'] = torch.tensor([degrees[node] for node in range(len(G))])

        for key, item in data.items():
            try:
                data[key] = torch.tensor(item)
            except ValueError:
                pass

        data['edge_index'] = edge_index.view(2, -1)

        if y is not None:
            data['y'] = y
        data['human'] = torch.tensor([human])

        data = torch_geometric.data.Data.from_dict(data)
        data.num_nodes = G.number_of_nodes()

        return data

    def create_loader(self, humans, batch_size=128, shuffle=True):
        data_list = []
        for human in humans:
            graphs = self.human2data[human]
            data_list.extend([self.from_networkx(G, human, l, edge_name='edge_attr') for (G, l) in graphs])
        if batch_size == -1:
            batch_size = sum(list(map(len, data_list)))
        return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)

    def kfold_split(self, nsplits, batch_size=128):
        human2data = self.human2data
        humans = np.array(list(human2data.keys()))
        kfold = KFold(nsplits, shuffle=True)
        for train_ix, test_ix in kfold.split(humans):
            train_humans = humans[train_ix]
            test_humans = humans[test_ix]
            train_loader = self.create_loader(train_humans, batch_size=128, shuffle=True)
            test_loader = self.create_loader(test_humans, batch_size=-1, shuffle=False)

            yield train_loader, test_loader

    def compute_human_predictions_and_scores(self, humans, predictions, human_labels):
        "This computes predictions per human by taking mean of individual human's predictions."
        "https://stackoverflow.com/a/56155805/2069858"

        human2score = dict(zip(humans.numpy(), human_labels.numpy()))

        human2ix = dict()
        ix2human = dict()
        avg_human_scores = []
        for ix, human in enumerate(humans.unique()):
            human2ix[human.item()] = ix
            ix2human[ix] = human.item()
            avg_human_scores.append(human2score[ix2human[ix]])

        avg_human_scores = torch.tensor(avg_human_scores).unsqueeze(-1)

        # this part computes avg predictions by human
        indexes = torch.tensor([human2ix[human.item()] for human in humans])
        n_predictions = predictions.size()[0]
        M = torch.zeros(indexes.max() + 1, n_predictions)
        M[indexes, torch.arange(n_predictions)] = 1
        M = torch.nn.functional.normalize(M, p=1, dim=1)
        if len(predictions.size()) == 1:
            predictions = predictions.unsqueeze(-1)
        avg_human_predictions = torch.mm(M, predictions)
        return avg_human_predictions, avg_human_scores

    def trainx(self, data_loader):
        self.model.train()

        loss_epoch = 0
        for data in data_loader:
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.batch)
            predictions, scores = self.compute_human_predictions_and_scores(data.human, output, data.y)

            loss = self.mse(predictions, scores)
            loss_epoch += loss.item()
            loss.backward()
            self.optimizer.step()
        return loss_epoch

    def testx(self, data_loader):
        self.model.eval()

        for data in data_loader:
            output = self.model(data.x, data.edge_index, data.batch)
            predictions, scores = self.compute_human_predictions_and_scores(data.human, output, data.y)
            mse_loss = self.mse(predictions, scores).item()
            mae_loss = self.mae(predictions, scores).item()
        return mse_loss, mae_loss

    def train_and_evaluate(self, n_epochs, train_loader, test_loader):
        print("epoch train_mse train_mae, mse mae")
        test_losses = []
        for epoch in range(n_epochs):
            train_loss = self.trainx(train_loader)
            train_mse, train_mae = self.testx(train_loader)
            mse, mae = self.testx(test_loader)
            test_losses.append((epoch, mse, mae))
            print(f"{epoch} {train_loss:.2f} {train_mse:.2f} {train_mae:.2f}, {mse:.2f} {mae:.2f}")
        return test_losses