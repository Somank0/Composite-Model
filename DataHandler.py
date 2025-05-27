import os
import pickle
import torch
from torch_geometric.data import DataLoader
import numpy as np
from time import time
from tqdm import tqdm

class DataHandler:
    def __init__(self, data_folder, idx_name, coords, target, graph_features=[], weights_name=None, ES="no", valid_batch_size=64):
        self.data_folder = data_folder
        self.idx_name = idx_name
        self.coords = coords
        self.target = target
        self.graph_features = graph_features
        self.weights_name = weights_name
        self.ES = ES
        self.valid_batch_size = valid_batch_size

    def loadValidIdx(self):
        prefix = f"{self.data_folder}/{self.idx_name}"
        with open(f"{prefix}_valididx.pickle", "rb") as f:
            self.valid_idx = pickle.load(f)

        train_file = f"{prefix}_trainidx.pickle"
        if os.path.exists(train_file):
            with open(train_file, "rb") as f:
                self.train_idx = pickle.load(f)
        else:
            self.train_idx = np.asarray([])

        print(len(self.valid_idx), "valid points")
        print(len(self.train_idx), "train points")
    
    def loadWeights(self):
        if self.weights_name is None:
            return
        fname = f"{self.data_folder}/{self.weights_name}_weights.pickle"
        with open(fname, "rb") as f:
            self.weights = pickle.load(f)
    
    def loadFeatures(self, predict=False):
        print("loading in features...")
        t0 = time()
        fname = f"{self.data_folder}/{self.coords}feat"
        if self.ES == "yes":
            fname += "_ES"
        elif self.ES == "scaled":
            fname += "_ES_scaled"

        data = torch.load(f"{fname}.pickle")
        print(f"\tTook {time() - t0:.3f} seconds")

        if self.graph_features:
            graph_x = []
            for var in self.graph_features:
                with open(f"{self.data_folder}/{var}.pickle", "rb") as f:
                    tmp = pickle.load(f)
                    # Apply scaling if needed
                    if var == "rho":
                        tmp /= 30.0
                    elif var == "Pho_HadOverEm":
                        tmp /= 0.5
                    graph_x.append(tmp)
            graph_x = np.stack(graph_x, 1) if len(graph_x) > 1 else graph_x[0]

            print("Adding graph features to data objects...")
            for it, gx in tqdm(zip(data, graph_x), total=len(data)):
                it.graph_x = torch.tensor(np.asarray(gx), dtype=torch.float32)

        if not predict:
            print("loading in target...")
            t0 = time()
            with open(f"{self.data_folder}/{self.target}_target.pickle", "rb") as f:
                target = pickle.load(f)
            print(f"\tTook {time() - t0:.3f} seconds")

            print("Matching targets with features...")
            for it, ta in tqdm(zip(data, target), total=len(target)):
                it.y = torch.tensor(np.asarray(ta), dtype=torch.float32)

        self.features = data
        self.num_features = data[0].x.shape[1]
        self.loader = DataLoader(data, batch_size=self.valid_batch_size, shuffle=False, pin_memory=True)
        self.datalen = len(data)
        print("datalen is", self.datalen)
        print("batch size is", self.loader.batch_size)
        print("ES is", self.ES, "and the number of features is", self.num_features)

