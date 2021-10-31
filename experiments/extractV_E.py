from __future__ import print_function
import json, time, os, sys, glob
import shutil

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset

# Library code
sys.path.insert(0, '..')
from struct2seq import *
from utils import *

args, device, model = setup_cli_model()
filepath='../experiments/chain_set.jsonl'
dataset = data.StructureDataset(filepath, truncate=None, max_length=500)
batch = dataset

def featurize(batch,device):
    """ Pack and pad batch into torch tensors """
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    B = 1
    lengths = np.array([len(batch['seq'])], dtype=np.int32)
    L_max = len(batch['seq'])
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)

    i=0
    b=batch
    x = np.stack([b['coords'][c] for c in ['N', 'CA', 'C', 'O']], 1)
    
    l = len(b['seq'])
    x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
    X[i,:,:,:] = x_pad

    # Convert to labels
    indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
    S[i, :l] = indices

    # Mask
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long,device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    return X, S, mask, lengths

X, S, mask, lengths = featurize(batch, device)
hyperparams = vars(args)
features=protein_features.ProteinFeatures(node_features=hyperparams['hidden'], edge_features=hyperparams['hidden'], top_k=hyperparams['k_neighbors'],
            features_type=hyperparams['features'], augment_eps=0,
            dropout=0.1
        )
V, E, E_idx = features(X, lengths, mask)
