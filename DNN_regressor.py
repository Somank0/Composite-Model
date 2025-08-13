import os
import torch
import torch.nn as nn

import os.path as osp
import math

import numpy as np
import gc
from torch.nn.functional import softplus
import torch_geometric.transforms as T

from torch.utils.checkpoint import checkpoint

class DNN_regressor(nn.Module) :
    def __init__(self,input_dim=64,hidden_dim=64, output_dim=1,hidden_layers=1,activ='elu'):
        super().__init__()
        if activ=='elu':
            activation=nn.ELU()
        if activ=='relu':
            activation=nn.ReLU()
        if activ=='sigmoid':
            activation=nn.Sigmoid()
        if activ=='tanh':
            activation=nn.Tanh()
        layers=[]
        layers +=[nn.Linear(input_dim,hidden_dim),activation]
        for i in range(hidden_layers):
            layers += [nn.Linear(hidden_dim,hidden_dim),activation]

        layers +=[nn.Linear(hidden_dim,output_dim),activation]

        self.model = nn.Sequential(*layers)

    def forward(self,data):
        #print("DEBUG: data.x shape =", data.x.shape)
        return self.model(data.x).squeeze(-1)

