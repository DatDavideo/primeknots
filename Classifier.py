import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torch_geometric.nn import GCNConv, RGCNConv, SignedConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import Sequential as GNNSequential
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Categorical, Real, Integer
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt

#  ___       _        
# |   \ __ _| |_ __ _ 
# | |) / _` |  _/ _` |
# |___/\__,_|\__\__,_|
def load_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            line_data = json.loads(line)
            data.append({
                'prime': line_data.get('prime'),
                'source': [i-1 for i in line_data.get('source', [])],
                'target': [i-1 for i in line_data.get('target', [])],
                'sign': line_data.get('sign', []),
                'id': line_data.get('id')})
    return data

def process_data(data, max_nodes, type='standard'):
    processed = []
    for graph in tqdm(data):
        source = graph['source'] + graph['target']
        target = graph['target'] + graph['source']
        sign = graph['sign'] + graph['sign']

        if type in ['standard', 'relational']:
            source = torch.tensor(source, dtype=torch.long)
            target = torch.tensor(target, dtype=torch.long)
            x = torch.eye(max_nodes)
            edge_index = torch.stack([source, target], dim=0)
            if type == 'standard':
                sign = torch.tensor(sign, dtype=torch.float32)
                sign[sign == -1] = 2
            elif type == 'relational':
                sign = torch.tensor(sign, dtype=torch.int64)
                sign[sign == -1] = 0
            entry = Data(x=x, edge_index=edge_index, edge_attr=sign, y=torch.tensor([graph['prime']], dtype=torch.float32), identity = graph['id'])
            processed.append(entry)
        else:
            adj_matrix = np.zeros((max_nodes, max_nodes))
            for s, t, attr in zip(source, target, sign):
                adj_matrix[s-1, t-1] = attr
            vec = adj_matrix.flatten()
            entry = (torch.tensor(vec, dtype=torch.float32), torch.tensor([graph['prime']], dtype=torch.float32))
            processed.append(entry)
    return processed

class AdjBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.x = torch.stack(transposed_data[0], 0)
        self.y = torch.stack(transposed_data[1], 0)

def collate_wrapper(batch):
    return AdjBatch(batch)

def load_data(filename, filename2=None, type='standard'):
    train = load_file(filename)
    if filename2:
        test = load_file(filename2)
        train_size = len(train)
    else:
        train_size = int(0.8 * len(train))
        train, test = random_split(train, [train_size, len(train) - train_size])
        test = list(test)
    val_size = int(0.1 * train_size)
    val, train = random_split(train, [val_size, train_size - val_size])
    max_nodes = 0
    for graph in test + list(train) + list(val):
        num_nodes = len(set(graph['source'] + graph['target']))
        if num_nodes > max_nodes:
            max_nodes = num_nodes
    train_data = process_data(train, max_nodes, type)
    test_data = process_data(test, max_nodes, type)
    val_data = process_data(val, max_nodes, type)
    return max_nodes, train_data, test_data, val_data

#  __  __         _     _    
# |  \/  |___  __| |___| |___
# | |\/| / _ \/ _` / -_) (_-<
# |_|  |_\___/\__,_\___|_/__/
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=16, num_layers=4, activation_type='leakyrelu', p=0.0, neg_slope=0.1):
        super(MLP, self).__init__()
        if activation_type == 'relu':
            activation = nn.ReLU()
        elif activation_type == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation_type == 'leakyrelu':
            activation = nn.LeakyReLU(neg_slope)
        layers = []
        for i in range(num_layers):
            indim = input_size if i == 0 else hidden_size
            outdim = 1 if i == num_layers - 1 else hidden_size
            layers.append(nn.Linear(indim, outdim))
            layers.append(nn.Dropout(p))
            if i != num_layers - 1:
                layers.append(activation)
        self.fc = nn.Sequential(*layers)

    def forward(self, batch):
        return self.fc(batch.x)
    
class GCNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, activation_type='relu', pooling='mean', neg_slope=0.1):
        super(GCNModel, self).__init__()
        if activation_type == 'relu':
            activation = nn.ReLU()
        elif activation_type == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation_type == 'leakyrelu':
            activation = nn.LeakyReLU(neg_slope)
        layers = []
        for i in range(num_layers):
            indim = input_size if i == 0 else hidden_size
            layers.append((GCNConv(indim, hidden_size), 'x, edge_index, edge_weight -> x'))
            layers.append((activation, 'x -> x'))
        self.conv = GNNSequential('x, edge_index, edge_weight', layers)
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'add':
            self.pool = global_add_pool
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, batch):
        x, edge_index, edge_attr, b = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        x = self.conv(x, edge_index, edge_attr)
        x = self.pool(x, b)
        x = self.fc(x)
        return x.view(-1)
    
class RGCNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, activation_type='relu', pooling='mean', neg_slope=0.1):
        super(RGCNModel, self).__init__()
        if activation_type == 'relu':
            activation = nn.ReLU()
        elif activation_type == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation_type == 'leakyrelu':
            activation = nn.LeakyReLU(neg_slope)
        layers = []
        for i in range(num_layers):
            indim = input_size if i == 0 else hidden_size
            layers.append((RGCNConv((indim, indim), hidden_size, num_relations=2, aggr=pooling), 'x, edge_index, edge_type -> x'))
            layers.append((activation, 'x -> x'))
        self.conv = GNNSequential('x, edge_index, edge_type', layers)
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'add':
            self.pool = global_add_pool
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, batch):
        x, edge_index, edge_attr, b = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        x = self.conv(x, edge_index, edge_attr)
        x = self.pool(x, b)
        x = self.fc(x)
        return x.view(-1)
    
def train_model(model, dataloader, optimizer, epochs, manual_batches=1):
    #begin = time.time()
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for i, batch in enumerate(dataloader):
            output = model(batch)
            loss = criterion(output, batch.y)
            loss.backward()
            if (i+1) % manual_batches == 0 or i+1 == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss += loss.item()
        #print(f'Epoch {epoch + 1}, Loss: {epoch_loss/len(dataloader)}')
    end = time.time()
    # print(f'Training time: {end - begin:.2f} seconds')

def test_model(model, dataloader):
    truth = []
    pred = []
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in dataloader:
            output = model(batch)
            predicted = (torch.sigmoid(output) > 0.5).float()
            total += batch.y.size(0)
            correct += (predicted == batch.y).sum().item()
            truth.extend(batch.y.cpu().numpy())
            pred.extend(predicted.cpu().numpy())
    accuracy = correct / total
    # print(f'Accuracy: {accuracy:.2f}')
    f1 = f1_score(truth, pred)
    # print(f'F1 Score: {f1:.4f}')
    return accuracy, f1

def run(mode, size, lr, wd, mom, eps, opt, act, neg, pool, hl, hs, dr, train, val, test=None, eval=False):
        if mode == 0:
            model = MLP(size, hs, hl, act, dr, neg)
        elif mode == 1:
            model = GCNModel(size, hs, hl, act, pool, neg)
        elif mode == 2:
            model = RGCNModel(size, hs, hl, act, pool, neg)
        if opt == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd, eps=eps)
        elif opt == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=mom)
        elif opt == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd, eps=eps)
        base = 0
        #f1 = 0
        train_model(model, train, optimizer, 10, manual_batches=100 if mode == 2 else 1)
        epochs = 10
        acc, _ = test_model(model, val)
        while (acc > base or acc < 0.5) and epochs < 300:
            base = acc
            train_model(model, train, optimizer, 10, manual_batches=100 if mode == 2 else 1)
            epochs += 10
            acc, _ = test_model(model, val)
        if eval:
            acc_test, f1_test = test_model(model, test)
            print(f'RESULT: {acc_test}, {f1_test}')
        return acc

def callback(res):
    print(f"Trial {len(res.func_vals)}")


#  __  __      _      
# |  \/  |__ _(_)_ _  
# | |\/| / _` | | ' \ 
# |_|  |_\__,_|_|_||_|
if __name__ == "__main__":
    batch_size = 100
    mode = 0

    if mode == 0:
        _, train, test, val = load_data('MedialGraphs/medial_graphs.txt', type='adjacency')
        train_dl = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_wrapper)
        test_dl = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate_wrapper)
        val_dl = DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=collate_wrapper)
    elif mode == 1:
        _, train, test, val = load_data('MedialGraphs/medial_graphs.txt', type='standard')
        train_dl = GeoLoader(train, batch_size=batch_size, shuffle=True)
        test_dl = GeoLoader(test, batch_size=batch_size, shuffle=False)
        val_dl = GeoLoader(val, batch_size=batch_size, shuffle=False)
    elif mode == 2:
        _, train, test, val = load_data('MedialGraphs/medial_graphs.txt', type='relational')
        train_dl = GeoLoader(train, batch_size=1, shuffle=True)
        test_dl = GeoLoader(test, batch_size=1, shuffle=False)
        val_dl = GeoLoader(val, batch_size=1, shuffle=False)

    if mode == 0:
        shape = train[0][0].numel()
    else:
        shape = train[0].x.shape[1]

    space = [
            Real(0.0001, 0.01, name='learning_rate'),
            Real(0, 0.1, name='weight_decay'),
            Real(0, 0.1, name='momentum'),
            Real(1e-11, 1e-7, name='eps'),
            Categorical(['adam', 'sgd', 'adagrad'], name='optimizer_type'),
            Categorical(['relu', 'leakyrelu'], name='activation_type'),
            Real(0.0001, 0.01, name='neg_slope'),
            Categorical(['mean', 'max', 'add'], name='pooling_type'),
            Integer(1, 8, name='hidden_layers'),
            Integer(16, 512, name='hidden_size'),
            Real(0.0, 0.5, name='dropout_rate'),
    ]

    @use_named_args(space)
    def tune(learning_rate, weight_decay, momentum, eps, optimizer_type, activation_type, neg_slope, pooling_type, hidden_layers, hidden_size, dropout_rate):
        return -run(mode, shape, learning_rate, weight_decay, momentum, eps, optimizer_type, activation_type, neg_slope, pooling_type, hidden_layers, hidden_size, dropout_rate, train_dl, val_dl)
    
    hyperparams = gp_minimize(tune, space, n_calls=100, random_state=0, acq_func="EI", callback=callback)
    print("Best parameters:")
    for dim, val in zip(space, hyperparams.x):
        print(f"{dim.name}: {val}")

    run(mode, shape, hyperparams.x[0], hyperparams.x[1], hyperparams.x[2], hyperparams.x[3], hyperparams.x[4], hyperparams.x[5], hyperparams.x[6], hyperparams.x[7], hyperparams.x[8], hyperparams.x[9], hyperparams.x[10], train_dl, val_dl, test_dl, eval=True)

    plot_convergence(hyperparams)
    plt.show()

"""
def process_data_signed(data, max_nodes):
    processed = []
    for graph in tqdm(data):
        source = torch.tensor(graph['source'] + graph['target'], dtype=torch.long)
        target = torch.tensor(graph['target'] + graph['source'], dtype=torch.long)
        sign = torch.tensor(graph['sign'] + graph['sign'], dtype=torch.float32)
        pos = torch.stack([source[sign == 1], target[sign == 1]], dim=0)
        neg = torch.stack([source[sign == -1], target[sign == -1]], dim=0)
        x = torch.eye(max_nodes)
        data = Data(x=x, pos_edges=pos, neg_edges=neg, y=torch.tensor([graph['prime']], dtype=torch.float32))
        processed.append(data)
    return processed
"""

"""class SignedModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SignedModel, self).__init__()
        self.conv1 = SignedConv(input_size, hidden_size, first_aggr=True)
        self.conv2 = SignedConv(hidden_size, hidden_size, first_aggr=False)
        self.conv3 = SignedConv(hidden_size, hidden_size, first_aggr=False)
        self.fc = nn.Linear(2 * hidden_size, 1)

    def forward(self, batch):
        x, pos, neg, b = batch.x, batch.pos_edges, batch.neg_edges, batch.batch
        x = torch.relu(self.conv1(x, pos, neg))
        x = torch.relu(self.conv2(x, pos, neg))
        x = torch.relu(self.conv3(x, pos, neg))
        x = global_mean_pool(x, b)
        x = self.fc(x)
        return x.view(-1)
"""

"""
if 3 in mode:
    max_n, train, test = load_data('train_shuffled.txt', type='signed')
    train_dl = GeoLoader(train, batch_size=1, shuffle=True)
    test_dl = GeoLoader(test, batch_size=1, shuffle=False)
    classifier = SignedModel(train[0].x.shape[1], 16)
    optimizer = optim.Adam(classifier.parameters(), lr=0.01)
    train_model(classifier, train_dl, optimizer, epochs, manual_batches=batch_size)
    test_model(classifier, test_dl)
"""
