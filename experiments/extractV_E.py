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
filepath='/Users/janie/Desktop/pre-training/alphafold2_results/results/jsonl_files/CB6_VH.jsonl'
dataset = data.StructureDataset(filepath, truncate=None, max_length=500)
batch = dataset

X, S, mask, lengths = featurize(batch, device)
hyperparams = vars(args)
features=protein_features.ProteinFeatures(node_features=hyperparams['hidden'], edge_features=hyperparams['hidden'], top_k=hyperparams['k_neighbors'],
            features_type=hyperparams['features'], augment_eps=0,
            dropout=0.1
        )
V, E, E_idx = features(X, lengths, mask)
