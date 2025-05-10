# %%
import os

import pandas as pd
import numpy as np
from scipy.io import mmread
from scipy.sparse import vstack
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, r, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import ListVector, IntVector
import scipy.sparse
import random
import tensorflow as tf
import scrublet as scr
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from torch.utils.data import Dataset


# %%
def generate_sim_dbl_DF(seed, ori_count_array):
    '''
    seed: random seed
    ori_count_array: original count array dataset   (gene*cells)
    a, b: shape parameters for sampling alpha from beta distribution
    simpler: True if using simpler model
    '''
    # randomly take two columns to generate artificial doublets
    random.seed(seed)
    np.random.seed(seed)  # To make sure numpy operations are also reproducible

    # number of doublets to generate
    n_dbl = int(ori_count_array.shape[1]*0.25)

    # initialize 
    sim_doublet = np.zeros((ori_count_array.shape[0], n_dbl))
    
    for i in range(n_dbl):
        m, n = random.choices(range(ori_count_array.shape[1]), k=2)
        sim_doublet[:, i] = (ori_count_array[:, m] + ori_count_array[:, n]) /2
  
    return sim_doublet

# %%
def filter_libsize(count_array):
    '''
    count_array = gene*cells
    '''
    library_size = count_array.sum(axis = 0)

    low_thres = np.percentile(library_size, 5)
    high_thres = np.percentile(library_size, 95)

    mask = (library_size >= low_thres) & (library_size <= high_thres)

    filtered_count_array = count_array[:, mask]

    return filtered_count_array



# %% [markdown]
# # Artificial Doublet Generation Function

# %%
#######################  ######################
def generate_sim_dbl(seed, ori_count_array, a, b, simpler = True):
    '''
    seed: random seed
    ori_count_array: original count array dataset   (gene*cells)
    a, b: shape parameters for sampling alpha from beta distribution
    simpler: True if using simpler model
    '''
    a = a
    b = b
    # randomly take two columns to generate artificial doublets
    random.seed(seed)
    np.random.seed(seed)  # To make sure numpy operations are also reproducible

    # number of doublets to generate
    n_dbl = int(ori_count_array.shape[1]*0.25)

    filtered_count = filter_libsize(ori_count_array)

    # initialize 
    alpha_d = np.zeros(n_dbl)
    sim_doublet = np.zeros((ori_count_array.shape[0], n_dbl))
    
    for i in range(n_dbl):
        m, n = random.choices(range(filtered_count.shape[1]), k=2)

        # sample alpha
        alpha_d[i] = np.random.beta(a, b, size=None)
        if simpler:
            sim_doublet[:, i] = (filtered_count[:, m] + filtered_count[:, n]) * alpha_d[i]
        else:
            lambda_value = (filtered_count[:, m] + filtered_count[:, n]) * alpha_d[i]
            sim_doublet[:, i] = np.random.poisson(lambda_value, lambda_value.shape[0])
    
    return sim_doublet, alpha_d


#%%
def joint_preprocess(observed_cells, simulated_doublets, libsize = True, n_comp = 50, seed = 123):
    '''
    observed_cells = observed cells (gene*cell): ndarray
    simulated_doublets = artificial doublets (gene*cell): ndarray
    n_comp = number of PC components (default 50)
    '''
    combined_data = np.hstack([observed_cells, simulated_doublets])     # gene*cell
    combined_adata = sc.AnnData(combined_data.T)    # cell*gene

    # filter data
    #sc.pp.filter_cells(combined_adata, min_counts = 10)
    sc.pp.filter_genes(combined_adata, min_counts = 1)

    # normalize, log, scale, PCA
    sc.pp.normalize_total(combined_adata, target_sum = 1e4)
    sc.pp.log1p(combined_adata)
    sc.pp.scale(combined_adata)
    sc.tl.pca(combined_adata, n_comps = n_comp)


    # feature selection
    sc.pp.highly_variable_genes(combined_adata, n_top_genes = 2000)

    # Extract PCA embeddings
    X_combined_pca = combined_adata.obsm['X_pca']
    pca_ob = X_combined_pca[:observed_cells.shape[1]]
    pca_ad = X_combined_pca[observed_cells.shape[1]:]

    # calculate library size
    print(f'Calculating library size.....')
    combined_libsize = np.sum(combined_data, axis = 0)
    libs_ob = combined_libsize[:observed_cells.shape[1]].reshape(-1, 1)
    libs_ad = combined_libsize[observed_cells.shape[1]:].reshape(-1, 1)

    # calculate doublet score (kNN)
    print(f'Calculating kNN scores.....')
    knn_pca = combined_adata.obsm['X_pca'] # Always take first 50 PCs

    y_combined = np.hstack([np.zeros(observed_cells.shape[1]), np.ones(simulated_doublets.shape[1])])
    
    np.random.seed(seed)
    k = int(np.sqrt(combined_data.shape[1]) * 2)
    #k = int(np.sqrt(2*combined_data.shape[1]))
    nn_model = NearestNeighbors(n_neighbors = k)
    nn_model.fit(X_combined_pca)

    distances, indices = nn_model.kneighbors(X_combined_pca)

    prop_dbl = np.sum(y_combined[indices] == 1, axis = 1) / k

    knn_ob = prop_dbl[:observed_cells.shape[1]].reshape(-1, 1)
    knn_ad = prop_dbl[observed_cells.shape[1]:].reshape(-1, 1)

    if libsize:
        X_unlabeled = np.hstack([pca_ob, libs_ob, knn_ob])
        X_positive = np.hstack([pca_ad, libs_ad, knn_ad])
    else:
        X_unlabeled = np.hstack([pca_ob, knn_ob])
        X_positive = np.hstack([pca_ad, knn_ad])       


    return X_positive, X_unlabeled




# %% [markdown]
# # Loss Function

# %%
def joint_loss(outputs, pseudo_labels, pi_p, alpha, beta, lambda_weight, positive_indices, unlabeled_indices):
    # Detach and clamp pseudo labels to avoid log(0)
    pseudo_labels = pseudo_labels.detach().clamp(1e-6, 1 - 1e-6)
    # Clamp model outputs (predicted probabilities)
    pred_probs = outputs.clamp(1e-6, 1 - 1e-6)

    # Compute positive loss only if there are positive samples
    if positive_indices.numel() > 0:
        pos_kl = -torch.log(pred_probs[positive_indices] + 1e-6)
        pos_loss = pos_kl.mean()
    else:
        pos_loss = 0.0

    # Compute unlabeled loss only if there are unlabeled samples
    if unlabeled_indices.numel() > 0:
        nois_kl_div = (pseudo_labels[unlabeled_indices] *
                       (torch.log(pseudo_labels[unlabeled_indices] + 1e-6) - torch.log(pred_probs[unlabeled_indices] + 1e-6)) +
                       (1 - pseudo_labels[unlabeled_indices]) *
                       (torch.log(1 - pseudo_labels[unlabeled_indices] + 1e-6) - torch.log(1 - pred_probs[unlabeled_indices] + 1e-6)))
        unl_loss = nois_kl_div.mean()
    else:
        unl_loss = 0.0

    # Combine the two parts
    L_c = lambda_weight * pos_loss + unl_loss

    # Regularization term L_reg1
    prior_probs = torch.tensor([pi_p, 1 - pi_p], device=outputs.device)
    if unlabeled_indices.numel() > 0:
        avg_positive = pred_probs[unlabeled_indices].mean()
        avg_negative = (1 - pred_probs[unlabeled_indices]).mean()
    else:
        # Provide some fallback if there are no unlabeled samples
        avg_positive, avg_negative = pi_p, 1 - pi_p

    avg_probs = torch.tensor([avg_positive, avg_negative], device=outputs.device)
    L_reg1 = prior_probs[0] * (torch.log(prior_probs[0] + 1e-6) - torch.log(avg_probs[0] + 1e-6)) + \
             prior_probs[1] * (torch.log(prior_probs[1] + 1e-6) - torch.log(avg_probs[1] + 1e-6))
    L_reg1 = L_reg1.clamp(1e-6, 1 - 1e-6)

    # Regularization term L_reg2
    if unlabeled_indices.numel() > 0:
        L_reg2 = -(pred_probs[unlabeled_indices] * torch.log(pred_probs[unlabeled_indices] + 1e-6) +
                   (1 - pred_probs[unlabeled_indices]) * torch.log(1 - pred_probs[unlabeled_indices] + 1e-6)).mean()
    else:
        L_reg2 = 0.0

    loss = L_c + alpha * L_reg1 + beta * L_reg2
    return pred_probs, loss, pos_loss, unl_loss, L_c, L_reg1, L_reg2

# %% [markdown]
# # Classifier Model 

# %%
# Define the PyTorch model
class PUModel(nn.Module):
    def __init__(self, input_dim):
        super(PUModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, 3),
            nn.LeakyReLU(),
            nn.Linear(3, 1),  # Output layer
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Replace `create_model` with this:
def create_model(input_dim):
    return PUModel(input_dim)

# %% [markdown]
# # PUDataset Class

# %%
# define class for training dataset
class PUDataset(Dataset):
    def __init__(self, unlabeled_data, positive_data, Z, pi_p):
        self.positive_data = positive_data  # X_p
        self.unlabeled_data = unlabeled_data    # X_u
        self.Z = Z    # for both P and U
        self.Y_pseudo = torch.cat([pi_p * torch.ones(len(unlabeled_data)), torch.ones(len(positive_data))], dim=0)
        self.Y_orig = torch.cat([pi_p * torch.ones(len(unlabeled_data)), torch.ones(len(positive_data))], dim=0)
        self.data = torch.cat([unlabeled_data, positive_data], dim = 0)
        self.unlabeled_indices = torch.arange(len(unlabeled_data))   
        self.positive_indices = torch.arange(len(unlabeled_data), len(self.data))

        self.loss_history = {'epoch':[], 'batch':[], 'pos_loss':[], 'unl_loss':[], 'L_c':[], 'L_reg1':[], 'L_reg2':[], 'lambda_weight':[]}

        self.samples_weights = torch.cat([torch.ones(len(unlabeled_data)) * (1/len(unlabeled_data)), torch.ones(len(positive_data)) * (1/len(positive_data))])     

    def __getitem__(self, idx):
        return idx, self.data[idx], self.Y_orig[idx], self.Y_pseudo[idx]
    
    def __len__(self):
        return len(self.data)
    
    def update_pseudo_labels(self, r, e_start, i):
        '''
        r: the last r epochs to average
        unlabeled_indices: indices of unlabeled samples in the current batch
        '''
        if i >= e_start:
            n_unlabeled = len(self.unlabeled_data)
            #self.Y_pseudo[:] = self.Z[:, (i - r):i].mean(dim = 1)
            self.Y_pseudo[:n_unlabeled] = self.Z[:n_unlabeled, (i - r):i].mean(dim=1)
            # Optionally, you can reassign positives explicitly
            self.Y_pseudo[n_unlabeled:] = 1.0


# %%
# define class for test dataset
class testDataset(Dataset):
    def __init__(self, unlabeled_data, pi_p):
        self.unlabeled_data = unlabeled_data    # X_u
        #self.Z = Z    # for both P and U
        self.Y_pseudo = pi_p * torch.ones(len(unlabeled_data))
        #self.Y_orig = pi_p * torch.ones(len(unlabeled_data))
        self.unlabeled_indices = torch.arange(len(unlabeled_data))   

    def __getitem__(self, idx):
        return idx, self.unlabeled_data[idx], self.Y_pseudo[idx]
    
    def __len__(self):
        return len(self.unlabeled_data)
    

# %% [markdown]
# # Training Loop Function

# %%
def train_PU(model, train_dataset, optimizer, scheduler, e_start, e_end, pi_p, alpha, beta, lambda_init, r, batch_size, device):
    print("Starting train_PU()...")
    breakpoint()

    e_end = int(e_end)

    n_u = len(train_dataset.unlabeled_data)   # number of unlabeled samples in the original dataset
    n_p = len(train_dataset.positive_data)    # number of positive samples in the original dataset
    loss_hist = torch.zeros(e_end)

    for i in range(1, e_end + 1):
        print(f"Starting epoch {i} / {e_end}", flush = True)
        
        model.train()   # set model to training mode
        epoch_loss = 0  # set current epoch loss

        #lambda_weight = ((e_end - i) / (e_end - 1)) * (lambda_init - pi_p) + pi_p
        lambda_weight = lambda_init
        
        # randomsampler
        # train_sampler = WeightedRandomSampler(weights = train_dataset.samples_weights, num_samples = len(train_dataset), replacement = True)

        # Shuffle dataset
        # train_loader = DataLoader(train_dataset, batch_size = batch_size, sampler = train_sampler)
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        for batch_index, (idx, inputs, orig_labels, pseudo_labels) in enumerate(train_loader):
            inputs, orig_labels, pseudo_labels = inputs.to(device), orig_labels.to(device), pseudo_labels.to(device)

            # extract P set indices (inside batch j)
            positive_indices_batch = (orig_labels == 1).nonzero(as_tuple = True)[0]
            # extract U set indices (inside batch j)
            unlabeled_indices_batch = (orig_labels != 1).nonzero(as_tuple = True)[0]

            # make predictions on the current batch and update parameters
            outputs = model(inputs)     # compute output
            pred_probs, loss, pos_loss, unl_loss, L_c, L_reg1, L_reg2 = joint_loss(outputs = outputs, pseudo_labels = pseudo_labels,
                              pi_p = pi_p, 
                              alpha = alpha, beta = beta, lambda_weight = lambda_weight, 
                              positive_indices = positive_indices_batch, 
                              unlabeled_indices = unlabeled_indices_batch)
            optimizer.zero_grad()   # reset gradient
            
            loss.backward()     # compute gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            optimizer.step()    # update model parameters

            # preserve prediction of the model Z
            train_dataset.Z[idx, i-1] = pred_probs.squeeze()    #line9
            epoch_loss += loss.item()


        # update pseudo labels
        train_dataset.update_pseudo_labels(r, e_start, i)
            

        # train_dataset.track_loss(i, batch_index, pos_loss, unl_loss, L_c, L_reg1, L_reg2, lambda_weight)

        before_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        after_lr = optimizer.param_groups[0]['lr']

        loss_hist[i-1] = epoch_loss

        print(f'Epoch {i} / {e_end}, Loss: {epoch_loss:.4f}, Lambda: {lambda_weight:.4f}, before_lr: {before_lr}, after_lr: {after_lr}')
    return train_dataset, loss_hist
    

#%%
def eval_PU(model, test_dataset, batch_size, device):
    print("Starting val_PU()...")
        
    model.eval()   # set model to training mode
    ops = torch.zeros(len(test_dataset))
    with torch.no_grad():        

        # Shuffle dataset
        eval_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
        
        for batch_index, (idx, inputs, pseudo_labels) in enumerate(eval_loader):
            inputs, pseudo_labels = inputs.to(device), pseudo_labels.to(device)

            # predict
            outputs = model(inputs)     # compute output

            #pred_probs = torch.sigmoid(outputs)
            pred_probs = outputs

            # # predict probabilities
            # test_dataset.Y_pseudo[idx] = pred_probs.squeeze()
            ops[idx] = pred_probs.squeeze()

    return ops
