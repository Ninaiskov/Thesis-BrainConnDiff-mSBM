# preamble
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
import seaborn as sns
from sklearn import metrics
from scipy.special import gamma, beta, gammaln, betaln
import nilearn as ni
from nilearn import plotting as nplt
from nilearn.maskers import NiftiLabelsMasker
import nilearn.connectome
import nibabel as nib
import networkx as nx
from matplotlib.colors import ListedColormap
from data_processors import compute_A
from ast import literal_eval
from collections import Counter

os.environ["OMP_NUM_THREADS"] = "4"  # set number of threads to 4

# main directory
main_dir = '/work3/s174162/speciale'
# general plotting parameters
label_fontsize = 12
subtitle_fontsize = 14
title_fontsize = 18
markersize = 1 # for sorted spy plots

############################################################ Data processing functions ############################################################
###################### Synthetic data ######################
def generate_syndata(N, K, S1, S2, Nc_type, eta_similarity, seed=0, disp_data = False, 
                     label_fontsize=label_fontsize, subtitle_fontsize=subtitle_fontsize, title_fontsize=title_fontsize):
    ## Inputs
    # N = total number of nodes (corresponds to n_rois)
    # K = Number of clusters;
    # S1 = Number of first type of graph, e.g. healthy
    # S2 = Number of second type of graph, e.g. sick
    # Nc_type = 'balanced' or 'unbalanced' no. of nodes in each cluster
    # eta_similarity = 'same', 'comp_diff' or 'part_diff' (how similar eta1 and eta2 should be)
    # disp_data = bool for displaying generated data
    
    # Output
    # A = adjacency matrices for all subjects
    
    np.random.seed(seed)
    # STEPS:
    # 1) compute partition (z)
        # balanced or unbalanced
    if Nc_type == 'balanced':
        Nc = int(N/K)
        z = np.kron(np.eye(K),np.ones((Nc,1)))
    elif Nc_type == 'unbalanced': 
        if K == 2:
            Nc_list = [70, 30]
        elif K == 5:
            Nc_list = [60, 20, 10, 5, 5]
        elif K == 10:
            Nc_list = [20, 20, 10, 10, 10, 10, 5, 5, 5, 5]
        else:
            print('Nc_list not specfied for chosen K')

        z = np.zeros((N, K))
        for k in range(K): # len(Nc_list) = K
            Nc = Nc_list[k]
            cumsumNc = int(np.sum(Nc_list[:k]))
            z[cumsumNc:cumsumNc+Nc, k] = 1
    else:
        print('Unknown Nc_type')
    
    # 2) compute block/cluster interactions (eta)
        # same, completely or partially different
    eta1 = np.random.rand(K, K)
    if eta_similarity == 'same':
        eta2 = eta1.copy()
    elif eta_similarity == 'comp_diff':
        eta2 = np.random.rand(K, K)
    elif eta_similarity == 'part_diff':
        #print('Using partially different etas')
        eta2 = eta1.copy()
        if K == 2:
            eta2[0,0] = np.random.rand()
        elif K == 5:
            eta2[:3,:3] = np.random.rand(3,3)
        elif K == 10:
            eta2[:5,:5] = np.random.rand(5,5)
        else:
            print('eta2 not specfied for chosen K')
    else:
        print('eta_similarity not specified') 
    
    # 3) compute adjacency matrices (A)
    A = np.empty((N, N, S1+S2))
    A.fill(np.nan)
    C1 = z @ eta1 @ z.T
    C2 = z @ eta2 @ z.T
    for s in range(S1+S2):
        if s <= S1-1:
            At = C1 > np.random.rand(N, N)
            A[:,:,s] = np.triu(At, 1) + np. triu(At, 1).T
        else:
            At = C2 > np.random.rand(N, N)
            A[:,:,s] = np.triu(At, 1) + np. triu(At, 1).T

    if disp_data:
        fig, ax = plt.subplots()
        cmap_binary = ListedColormap(['k', 'w']) 
        im = ax.imshow(z, interpolation='nearest', aspect='auto', cmap=cmap_binary, extent=(0, z.shape[1], 0, z.shape[0]))
        ax.set_ylabel('Node', fontsize=label_fontsize)
        ax.set_xlabel('Cluster', fontsize=label_fontsize)
        ax.set_title('Partition $z$', fontsize=title_fontsize, weight='bold')
        cbar = fig.colorbar(im, ax=ax, shrink=0.7, ticks=[0,1])
        plt.show()

        fig, ax = plt.subplots(1,2)
        plt.subplots_adjust(wspace=0.3, top=1.22)
        im = ax[0].imshow(eta1)
        ax[0].set_ylabel('Cluster', fontsize=label_fontsize)
        ax[0].set_xlabel('Cluster', fontsize=label_fontsize)
        ax[0].set_title('$\eta_1$', fontsize=subtitle_fontsize, weight='bold')
        im = ax[1].imshow(eta2)
        ax[1].set_ylabel('Cluster', fontsize=label_fontsize)
        ax[1].set_xlabel('Cluster', fontsize=label_fontsize)
        ax[1].set_title('$\eta_2$', fontsize=subtitle_fontsize, weight='bold')
        cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.3)
        fig.suptitle('Cluster-link probability matrices',fontsize=title_fontsize, weight='bold')
        plt.show()

        fig, ax = plt.subplots(1,2)
        plt.subplots_adjust(wspace=0.3, top=1.1)
        im = ax[0].imshow(C1)
        ax[0].set_ylabel('Node', fontsize=label_fontsize)
        ax[0].set_xlabel('Node', fontsize=label_fontsize)
        ax[0].set_title('$C_1$', fontsize=subtitle_fontsize, weight='bold')
        im = ax[1].imshow(C2)
        ax[1].set_ylabel('Node', fontsize=label_fontsize)
        ax[1].set_xlabel('Node', fontsize=label_fontsize)
        ax[1].set_title('$C_2$', fontsize=subtitle_fontsize, weight='bold')
        cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.3)
        #fig.suptitle('$C$',fontsize=15, weight='bold')
        plt.show()

        default_blue = '#1f77b4'
        cmap_binary = ListedColormap(['white', default_blue]) 
        fig, ax = plt.subplots(1,2)
        plt.subplots_adjust(wspace=0.3, top=1.1)
        im = ax[0].spy(A[:,:,0],cmap=cmap_binary)
        ax[0].set_ylabel('Node', fontsize=label_fontsize)
        ax[0].set_xlabel('Node', fontsize=label_fontsize)
        ax[0].set_title('$X_{type1}$', fontsize=subtitle_fontsize, weight='bold')
        im = ax[1].spy(A[:,:,-1],cmap=cmap_binary)
        ax[1].set_ylabel('Node', fontsize=label_fontsize)
        ax[1].set_xlabel('Node', fontsize=label_fontsize)
        ax[1].set_title('$X_{type2}$', fontsize=subtitle_fontsize , weight='bold')
        cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.3)
        fig.suptitle('Adjacency matrices',fontsize=title_fontsize, weight='bold')
        plt.show()
        
    return A, z, eta1, eta2, C1, C2
    

def old_generate_syndata(Nc, K, S1, S2, disp_data = False):
    ## Inputs
    # Nc = size of clusters (no. of nodes in each cluster)
    # K = Number of clusters;
    # S1 = Number of first type of graph, e.g. healthy
    # S2 = Number of second type of graph, e.g. sick
    # disp_data = bool for displaying generated data
    #numS = S1+S2; % Total number of graphs/subjects
    
    # Output
    # A = adjacency matrices for all subjects
    
    # generate two clusters
    z1 = np.kron(np.eye(K),np.ones((Nc,1)))
    z2 = np.kron(np.eye(K),np.ones((Nc,1)))
    # note we compute Kronecker tensor product between eye(K)
    # which is 5x5 and ones(Nc,1) which us 50x1 resulting in the block matrix
    # with dimension 50x5 x 1x5 = 250 x 5
    
    #%% merge first two generated clusters to create a larger cluster 
    z1[:,0] = z1[:,0]+z1[:,1] # adding the links of the second column to the first colum
    z1 = np.delete(z1, 1, 1) # removing 2nd column to maintain the total number of ones (i.e. links) after merging (otherwise it would be 300 and not 250)
    z2[:,0] = z2[:,0]+z2[:,1]
    z2 = np.delete(z2, 1, 1)
    
    #z1=z1(randperm(size(z1,1)),:);
    
    ##% Generate block interactions (eta1 is block interactions for type 1, eta2 for type 2)
    # try 3 cases: 1) same etas, 2)completely different etas, and 3) partially different etas
    eta1 = np.random.rand(K-1, K-1) # test with np.ones((K-1,K-1))
    eta2 = eta1.copy()
    #eta2 = np.random.rand(K-1, K-1)

    #eta2[:5,:5] = np.random.rand(5, 5)
    eta2[:2,:] = np.random.rand(2, K-1) # or actually dim = (K-1)/2 , K-1
    eta2[:,:2] = np.random.rand(K-1, 2)
    
    #%% Create adjacency matrices
    A = np.empty((Nc*K, Nc*K, S1+S2))
    A.fill(np.nan)
    for s in range(S1+S2):
        if s <= S1-1:
            At = z1 @ eta1 @ z1.T > np.random.rand(Nc*K, Nc*K)
            A[:,:,s] = np.triu(At, 1) + np. triu(At, 1).T
        else:
            At = z2 @ eta2 @ z2.T > np.random.rand(Nc*K, Nc*K)
            A[:,:,s] = np.triu(At, 1) + np. triu(At, 1).T
    
    if disp_data:
        plt.figure()
        plt.imshow(z1, interpolation='nearest', aspect='auto', cmap=plt.cm.gray,extent=(0, z1.shape[1], 0, z1.shape[0]))
        plt.title('Partition')

        plt.figure()
        plt.imshow(z1, interpolation='nearest', aspect='auto', cmap=plt.cm.gray, extent=(0, z1.shape[1], 0, z1.shape[0]))
        plt.title('Partition (modified)')

        fig, axs = plt.subplots(1,2, constrained_layout=True)
        axs = axs.ravel()
        axs[0].set_aspect('equal', adjustable='box')
        axs[1].set_aspect('equal', adjustable='box')
        im0 = axs[0].imshow(eta1, extent=(0, eta1.shape[1], 0, eta1.shape[0]))
        axs[0].set_title('eta1')
        im1 = axs[1].imshow(eta2, extent=(0, eta2.shape[1], 0, eta2.shape[0]))
        axs[1].set_title('eta2')
        fig.colorbar(im0, ax=axs[0], shrink=0.5)
        fig.colorbar(im1, ax=axs[1], shrink=0.5)

        fig, axs = plt.subplots(2,5, figsize=(15, 6), constrained_layout=True)
        axs = axs.ravel()

        for s in range(S1+S2):
            axs[s].spy(A[:,:,s], marker='.', markersize=3, color='k')
            axs[s].set_title('Subject: '+str(s+1))

        plt.suptitle('Adjacency matrices')
        plt.show()
        
    return A

###################### fMRI data ######################

def compute_A(A_vals_list, n_rois):
    # input: list of vectors with upper triangular values of adjacency matrix for each subject (stored like this to save memory: A_vals_list is around 10 times smaller, e.g. A.npy is 112.8 MB and A_vals_list.npy is 7 MB)
    
    # output:
    # A       fmri-based adjacency matrices for all subjects: dim = (N x N x S)
    
    As_list = []
    for s in range(len(A_vals_list)):
        As_vals = A_vals_list[s]
        As_triu = np.zeros((n_rois,n_rois))
        As_triu[np.triu_indices(n_rois, 1)] = As_vals
        As = As_triu + As_triu.T # n_rois x n_rois
        As_list.append(As)

    A = np.stack(As_list, axis=2)
    return A
    
def compute_A_collaps(dataset, A_vals_list, n_rois):
    # input: 
    # dataset = 'decnef' or 'hcp'
    # A_vals_list = list of vectors with upper triangular values of adjacency matrix for each subject
    # n_rois = number of areas used in Schaefer atlas
    
    # output:
    # A_type1_collaps = collapsed adjacency matrix for type 1 graph, e.g. fmri (functional) if dataset='hcp'
    # A_type2_collaps = collapsed adjacency matrix for type 2 graph, e.g. dmri (structural) if dataset='hcp'
    
    A = compute_A(A_vals_list, n_rois)
    if dataset == 'decnef':
        my_dir = '/work3/s174162/speciale/'
        healthy_bool = np.load(os.path.join(my_dir, 'data/decnef/healthy_mask.npy'), allow_pickle=True)
        schizo_bool = np.load(os.path.join(my_dir, 'data/decnef/schizo_mask.npy'), allow_pickle=True)
        A_type1 = A[:,:,healthy_bool]
        A_type2 = A[:,:,schizo_bool]
    elif dataset == 'hcp':
    # extracting types 
        A_type1 = A[:,:,:250] # adjacency matrix for fmri
        A_type2 = A[:,:,250:] # adjacency matrix for dmri
    
    # collapsing graphs
    A_type1_collaps = A_type1.sum(axis=2)
    A_type2_collaps = A_type2.sum(axis=2)
    return A_type1_collaps, A_type2_collaps


############################################################ Plotting functions for visualizing results  ############################################################
def get_exp_folders(dataset, model_type, var, main_dir=main_dir):
    # old function used to get experiment folder names

    # list of supfolders for model-specific experiments
    supexp_folders = [item for item in os.listdir(os.path.join(main_dir, 'results/'+dataset)) if model_type in item] 
    
    # list of subfolders for specified model experiments
    exp_folders = [os.path.join(main_dir, 'results/'+dataset+'/'+str(item)) for item in supexp_folders if str(var) in item]
    return exp_folders

def get_stats(exp_paths, par):
    
    '''
    par_list = []
    min_maxiter = 0
    for i in range(len(exp_folders)):
        folder = exp_folders[i]
        sample = np.load(os.path.join(folder,'model_sample.npy'), allow_pickle=True).item()
        sample_maxiter = len(sample[par])
        if min_maxiter < sample_maxiter:
            min_maxiter = sample_maxiter
        par_list.append(sample[par][:min_maxiter])
        # compute mean logP at each iteration across all experiments
        mean_par = np.mean(par_list, axis=0)
        # compute min and max logP at each iteration
        min_par = np.min(par_list, axis=0)
        max_par = np.max(par_list, axis=0)
    '''
    MAPpar_list = []
    par_list = []
    min_maxiter = np.inf
    for folder in exp_paths:
        sample = np.load(os.path.join(folder, 'model_sample.npy'), allow_pickle=True).item()
        MAPpar = sample['MAP'][par]
        par_array = sample[par]
        sample_maxiter = len(par_array)

        min_maxiter = min(min_maxiter, sample_maxiter)
        MAPpar_list.append(MAPpar)
        par_list.append(par_array[:min_maxiter])

    # Pad or truncate arrays to have consistent shape
    par_list = [np.pad(arr, (0, min_maxiter - len(arr)), mode='constant') if len(arr) < min_maxiter else arr[:min_maxiter] for arr in par_list]

    # Compute mean logP at each iteration across all experiments
    mean_par = np.mean(par_list, axis=0)
    # Compute min and max logP at each iteration
    min_par = np.min(par_list, axis=0)
    max_par = np.max(par_list, axis=0)
    
    return MAPpar_list, par_list, mean_par, min_par, max_par

def plot_par(dataset, df, par, miniter_gibbs=None, maxiter_gibbs=None, main_dir=main_dir, label_fontsize=label_fontsize, subtitle_fontsize=subtitle_fontsize, title_fontsize=title_fontsize):
    
    # Input:  
    # dataset: 'hcp' or 'decnef'
    # df: dataframe with experiment overview
    # par: parameter to plot as a function of Gibbs iterations, e.g. 'logP' or 'noc'
    # miniter_gibbs: minimum Gibbs iteration to plot
    # maxiter_gibbs: maximum Gibbs iteration to plot
    
    # Output: plot of parameter as a function of Gibbs iterations

    # Sort by dataset and relevant columns
    if dataset == 'hcp' or dataset == 'decnef':
        df_dataset = df.loc[df.dataset==dataset].sort_values(by=['n_rois','threshold_annealing','model_type','splitmerge','noc'], ascending=True).reset_index()
        unique_models = df_dataset[['model_type', 'n_rois', 'noc']].drop_duplicates()
    elif dataset == 'synthetic':
        df_dataset = df.loc[df.dataset==dataset].sort_values(by=['N','K','S1','S2','Nc_type', 'eta_similarity', 'model_type','splitmerge','noc'], ascending=True).reset_index()
        unique_models = df_dataset[['model_type', 'K', 'S1', 'Nc_type', 'eta_similarity', 'noc']].drop_duplicates()
    else:
        print('unknown dataset')
    
    plt.figure(figsize=(8,6))

    marker_dict = {'nonparametric': '*', 'parametric': 'o'}
    #marker_dict = {'balanced': '*', 'unbalanced': 'o'}
    linestyle_dict = {'same': '-', 'comp_diff': '--', 'part_diff': '-.'}
    cmap = plt.get_cmap('viridis')
    
    noc_values = df_dataset['noc'].unique()
    num_noc = len(noc_values)
    
    for _, row in unique_models.iterrows():
        
        model_type = row['model_type']
        noc = row['noc']
        if dataset == 'hcp' or dataset == 'decnef':
            n_rois = row['n_rois']
            model_df = df_dataset.loc[(df_dataset['model_type'] == model_type) & (df_dataset['n_rois'] == n_rois) & (df_dataset['noc'] == noc)]
            
        elif dataset == 'synthetic':
            K = row['K']
            S1 = row['S1']
            Nc_type = row['Nc_type']
            eta_similarity = row['eta_similarity']
            model_df = df_dataset.loc[(df_dataset['model_type'] == model_type) & (df_dataset['K'] == K) & (df_dataset['S1'] == S1) & 
                                      (df_dataset['Nc_type'] == Nc_type) & (df_dataset['eta_similarity'] == eta_similarity) & (df_dataset['noc'] == noc)]
            linestyle = linestyle_dict.get(eta_similarity)
        else:
            print('unknown dataset')

        marker = marker_dict.get(model_type)
        
        noc_color_index = list(noc_values).index(noc) % num_noc
        
        if dataset == 'hcp' or dataset == 'decnef':
            color_intensity = n_rois / df_dataset['n_rois'].max()  # Higher n_rois, darker intensity
            label = f"{model_type}_nrois{n_rois}_noc{noc}"
        elif dataset == 'synthetic':
            color_intensity = K / df_dataset['K'].max()  # Higher K, darker intensity
            label = f"{model_type}_K{K}_S1{S1}_Nc_type{Nc_type}_eta_similarity{eta_similarity}_noc{noc}"
        else:
            print('unknown dataset')
        color = cmap((color_intensity + noc_color_index) / num_noc)
        
        for _, model_row in model_df.iterrows():
            folders = model_row['exp_name_list']
            paths = [os.path.join(main_dir, 'results', dataset, folder) for folder in folders]
            _, _, mean_par, min_par, max_par = get_stats(paths, par)
            iters = range(len(mean_par))
            
            if miniter_gibbs is None:
                miniter_gibbs = iters[0]
            if maxiter_gibbs is None:
                maxiter_gibbs = iters[-1]
            
            if dataset == 'hcp' or dataset == 'decnef':
                plt.plot(iters[miniter_gibbs:maxiter_gibbs], mean_par[miniter_gibbs:maxiter_gibbs], marker=marker, label=label, color=color)
            elif dataset == 'synthetic':
                plt.plot(iters[miniter_gibbs:maxiter_gibbs], mean_par[miniter_gibbs:maxiter_gibbs], marker=marker, linestyle=linestyle, label=label, color=color)
            else:
                print('unknown dataset')   
            plt.fill_between(iters[miniter_gibbs:maxiter_gibbs], min_par[miniter_gibbs:maxiter_gibbs], max_par[miniter_gibbs:maxiter_gibbs], alpha=0.5, color=color)
        
    plt.title(par + ' - ' + dataset)
    plt.ylabel(par)
    plt.xlabel('Gibbs iterations')
    plt.legend(loc='upper right',fontsize='small', fancybox=True, shadow=True, bbox_to_anchor=(1.4, 0.85))
    plt.show()


def boxplot_par(dataset, df, par, miniter_gibbs=None, maxiter_gibbs=None, main_dir=main_dir, label_fontsize=label_fontsize, subtitle_fontsize=subtitle_fontsize, title_fontsize=title_fontsize):
    
    # Input:  
    # dataset: 'hcp' or 'decnef'
    # df: dataframe with experiment overview
    # par: parameter to plot as a function of Gibbs iterations, e.g. 'logP' or 'noc'
    # miniter_gibbs: minimum Gibbs iteration to plot
    # maxiter_gibbs: maximum Gibbs iteration to plot
    
    # Output: plot of parameter as a function of Gibbs iterations

    # Sort by dataset and relevant columns
    if dataset == 'hcp' or dataset == 'decnef':
        df_dataset = df.loc[df.dataset==dataset].sort_values(by=['n_rois','threshold_annealing','model_type','splitmerge','noc'], ascending=True).reset_index()
        unique_models = df_dataset[['model_type', 'n_rois', 'noc']].drop_duplicates()
    elif dataset == 'synthetic':
        df_dataset = df.loc[df.dataset==dataset].sort_values(by=['N','K','S1','S2','Nc_type', 'eta_similarity', 'model_type','splitmerge','noc'], ascending=True).reset_index()
        unique_models = df_dataset[['model_type', 'K', 'S1', 'Nc_type', 'eta_similarity', 'noc']].drop_duplicates()
    else:
        print('unknown dataset')

    cmap = plt.get_cmap('viridis')
    
    noc_values = df_dataset['noc'].unique()
    num_noc = len(noc_values)
    
    label_list = []
    color_list = []
    MAPpar_data = []
    for _, row in unique_models.iterrows():
        
        model_type = row['model_type']
        noc = row['noc']
        if dataset == 'hcp' or dataset == 'decnef':
            n_rois = row['n_rois']
            model_df = df_dataset.loc[(df_dataset['model_type'] == model_type) & (df_dataset['n_rois'] == n_rois) & (df_dataset['noc'] == noc)]
            
        elif dataset == 'synthetic':
            K = row['K']
            S1 = row['S1']
            Nc_type = row['Nc_type']
            eta_similarity = row['eta_similarity']
            model_df = df_dataset.loc[(df_dataset['model_type'] == model_type) & (df_dataset['K'] == K) & (df_dataset['S1'] == S1) & 
                                      (df_dataset['Nc_type'] == Nc_type) & (df_dataset['eta_similarity'] == eta_similarity) & (df_dataset['noc'] == noc)]
        else:
            print('unknown dataset')

        noc_color_index = list(noc_values).index(noc) % num_noc
        
        if dataset == 'hcp' or dataset == 'decnef':
            color_intensity = n_rois / 300#df_dataset['n_rois'].max()  # Higher n_rois, darker intensity
            label = f"{model_type}_nrois{n_rois}_noc{noc}"
        elif dataset == 'synthetic':
            color_intensity = K / 10#df_dataset['K'].max()  # Higher K, darker intensity
            label = f"{model_type}_K{K}_S1{S1}_Nc_type{Nc_type}_eta_similarity{eta_similarity}_noc{noc}"
        else:
            print('unknown dataset')
        label_list.append(label)
        
        color = cmap((color_intensity + noc_color_index) / num_noc)
        color_list.append(color)
        
        for _, model_row in model_df.iterrows():
            folders = model_row['exp_name_list']
            paths = [os.path.join(main_dir, 'results', dataset, folder) for folder in folders]
            MAPpar_list, _, _, _, _ = get_stats(paths, par)
            MAPpar_data.append(MAPpar_list)
            
    fig, ax = plt.subplots()
    bp = ax.boxplot(MAPpar_data, patch_artist=True)#, boxprops=dict(facecolor=color_list))
    ax.set_xticklabels(label_list, rotation=90)
    for box, color in zip(bp['boxes'], color_list):
        box.set_facecolor(color)
    ax.set_ylabel(par + ' distribution')
    ax.set_title('Boxplots of ' + par + ' - ' + dataset)
    ax.yaxis.grid(True)
    plt.show()

def old_plot_par(dataset, df, par, miniter_gibbs=None, maxiter_gibbs=None, main_dir=main_dir):
    
    # Input: 
    # exp_folder_lists: list of experiment folder names
    # par: parameter to plot as a function of Gibbs iterations, e.g. 'logP' or 'noc'

    # sort by dataset
    if dataset == 'hcp' or dataset == 'decnef':
        df_dataset = df.loc[df.dataset==dataset].sort_values(by=['n_rois','threshold_annealing','model_type','splitmerge','noc'], ascending=True).reset_index()
    elif dataset == 'synthetic':
        df_dataset = df.loc[df.dataset==dataset].sort_values(by=['N','K','S1','S2','Nc_type', 'eta_similarity', 'model_type','splitmerge','noc'], ascending=True).reset_index()
    else:
        print('unknown dataset')
    
    # Output: plot of parameter as a function of Gibbs iterations
    plt.figure(figsize=(8,6))
    for i in range(len(df_dataset.exp_name_list)):
        folders = df_dataset.exp_name_list[i]
        paths = [os.path.join(main_dir,'results/'+dataset+'/'+folder) for folder in folders]
        _, mean_par, min_par, max_par = get_stats(paths, par)
        iters = range(len(mean_par))
        if miniter_gibbs is None:
            miniter_gibbs = iters[0]
        if maxiter_gibbs is None:
            maxiter_gibbs = iters[-1]
        label = df_dataset.model_type[i]+'_nrois'+df_dataset.n_rois[i]+'_noc'+str(df_dataset.noc[i])
        plt.plot(iters[miniter_gibbs:maxiter_gibbs], mean_par[miniter_gibbs:maxiter_gibbs],'-o', label=label)
        plt.fill_between(iters[miniter_gibbs:maxiter_gibbs], min_par[miniter_gibbs:maxiter_gibbs], max_par[miniter_gibbs:maxiter_gibbs], alpha=0.5)

    plt.title(par + ' - ' + dataset)
    plt.ylabel(par)
    plt.xlabel('Gibbs iterations')
    plt.legend(loc='upper right',fontsize='small', fancybox=True, shadow=True, bbox_to_anchor=(1.4, 0.85))
    plt.show()

def oldold_plot_par(dataset, exp1_folders, exp2_folders, par, plot1=True, plot2=True, miniter_gibbs=None, maxiter_gibbs=None, main_dir=main_dir):
    
    # Input: 
    # exp_folder_lists: list of experiment folder names (1 is without annenaling, 2 is with annealing)
    # par: parameter to plot as a function of Gibbs iterations, e.g. 'logP' or 'noc'
    
    # Output: plot of parameter as a function of Gibbs iterations
    
    exp1_paths = [os.path.join(main_dir,'results/'+dataset+'/'+folder) for folder in exp1_folders]
    exp2_paths = [os.path.join(main_dir,'results/'+dataset+'/'+folder) for folder in exp2_folders]
    
    plt.figure()
    #for i in range(len(exp_folder_list)):
    #exp_folder = exp_folder_list[i]
    if plot1:
        _, mean_par, min_par, max_par = get_stats(exp_folder_list1, par)
        iters = range(len(mean_par))
        if miniter_gibbs is None:
            miniter_gibbs = iters[0]
        if maxiter_gibbs is None:
            maxiter_gibbs = iters[-1]
        #plt.errorbar(iters, mean_logP, yerr=max_logP-min_logP, fmt='-o', capsize=5, label=var)
        plt.plot(iters[miniter_gibbs:maxiter_gibbs], mean_par[miniter_gibbs:maxiter_gibbs],'-bo', label='exp1')
        plt.fill_between(iters[miniter_gibbs:maxiter_gibbs], min_par[miniter_gibbs:maxiter_gibbs], max_par[miniter_gibbs:maxiter_gibbs], alpha=0.5, color='blue')

    if plot2:
        _, mean_par, min_par, max_par = get_stats(exp_folder_list2, par)
        iters = range(len(mean_par))
        #miniter_gibbs = iters[0]
        #maxiter_gibbs = iters[-1]
        #plt.errorbar(iters, mean_logP, yerr=max_logP-min_logP, fmt='-o', capsize=5, label=var)
        plt.plot(iters[miniter_gibbs:maxiter_gibbs], mean_par[miniter_gibbs:maxiter_gibbs],'-ro', label='exp2')
        plt.fill_between(iters[miniter_gibbs:maxiter_gibbs], min_par[miniter_gibbs:maxiter_gibbs], max_par[miniter_gibbs:maxiter_gibbs], alpha=0.5, color='red')

    plt.title(par + ' - ' + dataset)
    plt.ylabel(par)
    plt.xlabel('Gibbs iterations')
    plt.legend()
    plt.show()
    
    
def plot_eta(dataset, eta, exp_name, main_dir=main_dir, label_fontsize=label_fontsize, subtitle_fontsize=subtitle_fontsize, title_fontsize=title_fontsize):
    if dataset == 'hcp':
        eta_type1 = eta[:,:,:250] # fmri
        eta_type2 = eta[:,:,250:] # dmri
    elif dataset == 'decnef':
        healthy_mask = np.load(main_dir+'/data/'+dataset+'/healthy_mask.npy')
        schizo_mask = np.load(main_dir+'/data/'+dataset+'/schizo_mask.npy')
        eta_type1 = eta[:,:,healthy_mask] # doublecheck that the masks are correct
        eta_type2 = eta[:,:,schizo_mask]
    elif dataset == 'synthetic':
        eta_type1 = eta[:,:,:S1]
        eta_type2 = eta[:,:,S1:S2]
    else: 
        print('unknown dataset')

    fig, axs = plt.subplots(2, 5, figsize=(15, 7), constrained_layout=True)
    axs = axs.ravel()

    for s in range(10):
        if s < 5:
            im = axs[s].imshow(eta_type1[:,:,s])
            #if dataset=='hcp':
            #    axs[s].set_title('Functional: '+str(s+1), fontsize=subtitle_fontsize, weight='bold')
            #else:
            #    axs[s].set_title('Healthy: '+str(s+1), fontsize=subtitle_fontsize, weight='bold')
            axs[s].set_ylabel('Cluster', fontsize=label_fontsize)
            axs[s].set_xlabel('Cluster', fontsize=label_fontsize)
        else:
            im = axs[s].imshow(eta_type2[:,:,-(s-4)])
            #if dataset=='hcp':
            #    axs[s].set_title('Structural: '+str(s-4), fontsize=subtitle_fontsize, weight='bold')
            #else:
            #    axs[s].set_title('Schizo: '+str(s-4), fontsize=subtitle_fontsize, weight='bold')
            axs[s].set_ylabel('Cluster', fontsize=label_fontsize)
            axs[s].set_xlabel('Cluster', fontsize=label_fontsize)

    if dataset=='hcp':
        axs[0].set_title('Functional', fontsize=subtitle_fontsize, weight='bold')
        axs[5].set_title('Structural', fontsize=subtitle_fontsize, weight='bold')
    else: 
        axs[0].set_title('Healthy', fontsize=subtitle_fontsize, weight='bold')
        axs[5].set_title('Schizophrenia', fontsize=subtitle_fontsize, weight='bold')
        
    fig.suptitle('Cluster-link probability matrices for ' + dataset + ' data,\n Experiment: ' + exp_name, fontsize=title_fontsize, weight='bold')
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.95)
    plt.savefig(main_dir+'/figures/'+dataset+'_eta_types.png', bbox_inches='tight')    
   
   
def plot_circ_stdeta(dataset, eta, noc, threshold=0, main_dir=main_dir, label_fontsize=label_fontsize, subtitle_fontsize=subtitle_fontsize, title_fontsize=title_fontsize):
    # threshold = link probability threshold (e.g based on np.percentile(eta_example.flatten(),99)))
    
    eta_std = np.std(eta, axis=2)
    eta_example = np.triu(eta_std) # upper triangular matrix
    # eta_example = np.triu(eta[:,:,example_sub])

    # create a graph from the matrix
    G = nx.from_numpy_array(eta_example)

    # remove self-edges
    #G.remove_edges_from(nx.selfloop_edges(G))

    # set edge colors based on the edge weights
    edge_colors = [eta_example[i, j] for i, j in zip(*np.where(np.triu(eta_example) > threshold))]
    edge_cmap = plt.cm.Blues

    # draw the graph in a circular layout
    pos = nx.circular_layout(G)
    node_color = sns.color_palette('hls', noc)

    # add only edges above the threshold
    G_thresh = nx.Graph()
    for i in range(noc):
        G_thresh.add_node(i)
    for i, j in zip(*np.where(np.triu(eta_example) > threshold)):
        G_thresh.add_edge(i, j, weight=eta_example[i, j])

    nx.draw_circular(G_thresh, with_labels=True, node_color=node_color, node_size=300,
                    edge_color=edge_colors, edge_cmap=edge_cmap, width=2,
                    font_size=label_fontsize)

    # add a colorbar for the edge colors
    sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=eta_example.min(), vmax=eta_example.max()))
    sm.set_array([])
    plt.colorbar(sm)

    # show the plot
    plt.title('Std eta circular graph')
    plt.show()
    plt.savefig(main_dir+'/figures/'+dataset+'_stdeta_circ.png', bbox_inches='tight')
     
        
def plot_sortedA(Z, dataset, n_rois, atlas_name='schaefer', main_dir=main_dir, label_fontsize=label_fontsize, subtitle_fontsize=subtitle_fontsize, title_fontsize=title_fontsize):
    data_folder_name = atlas_name+str(n_rois)
    folder_path = os.path.join(main_dir, 'data/'+dataset+'/'+data_folder_name)

    # compute A
    A_vals_list = np.load(os.path.join(folder_path,'A4_vals_list.npy'))
    A = compute_A(A_vals_list, n_rois)

    # convert from one-hot representation to integer representation 0, 1, .., K-1 for each node
    Z_int = np.argmax(Z.T, 1).astype('int')

    # sort by based assigned cluster
    sort_idx = np.argsort(Z_int)
    A_sorted = A[sort_idx, :, :][:, sort_idx, :]

    # count number nodes in each cluster
    count = Counter(Z_int[sort_idx])

    # plot
    #colors = np.array(plt.cm.Accent.colors)
    colors = []
    for i in range(len(count)):
        colors.append('#%06X' % np.random.randint(0, 0xFFFFFF))
        
    if dataset == 'hcp':
        A_type1 = A_sorted[:,:,:250] # fmri
        A_type2 = A_sorted[:,:,250:] # dmri
    else: #dataset == 'decnef'
        healthy_mask = np.load(main_dir+'/data/'+dataset+'/healthy_mask.npy')
        schizo_mask = np.load(main_dir+'/data/'+dataset+'/schizo_mask.npy')
        A_type1 = A_sorted[:,:,healthy_mask] # healthy
        A_type2 = A_sorted[:,:,schizo_mask] # schizo

    fig, axs = plt.subplots(2, 5, figsize=(15, 7), constrained_layout=True)
    #fig.subplots_adjust(top=0.9)  # Add space between suptitle and subplots
    #fig.subplots_adjust(hspace=0.6)  # Add space between the first 5 subplots and the last 5 subplots

    axs = axs.ravel()
    default_blue = '#1f77b4'
    cmap_binary = ListedColormap(['white', default_blue]) 

    for s in range(10):
        if s < 5:
            im = axs[s].spy(A_type1[:,:,s], marker='.', markersize=markersize)
            #if dataset=='hcp':
            #    axs[s].set_title('Functional: '+str(s+1), fontsize=subtitle_fontsize, weight='bold')
            #else:
            #    axs[s].set_title('Healthy: '+str(s+1), fontsize=subtitle_fontsize, weight='bold')
            axs[s].set_ylabel('Nodes permuted', fontsize=label_fontsize)
            axs[s].set_xlabel('Nodes permuted', fontsize=label_fontsize)
            axs[s].grid(False)
            # draw clusters partitions on adjacency matrix
            last_val = -0.5
            for i, x in enumerate(np.cumsum(list(count.values()))):
                size = x - last_val
                axs[s].add_patch(plt.Rectangle((last_val,last_val), size, size, fc=colors[i], ec=colors[i]))
                last_val = x
        else:
            im = axs[s].spy(A_type2[:,:,-(s-4)], marker='.', markersize=markersize)
            #if dataset=='hcp':
            #    axs[s].set_title('Structural: '+str(s-4), fontsize=subtitle_fontsize, weight='bold')
            #else:
            #    axs[s].set_title('Schizo: '+str(s-4), fontsize=subtitle_fontsize, weight='bold')
            axs[s].set_ylabel('Nodes permuted', fontsize=label_fontsize)
            axs[s].set_xlabel('Nodes permuted', fontsize=label_fontsize)
            axs[s].grid(False)
            # draw clusters partitions on adjacency matrix
            last_val = -0.5
            for i, x in enumerate(np.cumsum(list(count.values()))):
                size = x - last_val
                axs[s].add_patch(plt.Rectangle((last_val,last_val), size, size, fc=colors[i], ec=colors[i]))
                last_val = x

    if dataset=='hcp':
        axs[0].set_title('Functional', fontsize=subtitle_fontsize, weight='bold')
        axs[5].set_title('Structural', fontsize=subtitle_fontsize, weight='bold')
    else: 
        axs[0].set_title('Healthy', fontsize=subtitle_fontsize, weight='bold')
        axs[5].set_title('Schizophrenia', fontsize=subtitle_fontsize, weight='bold')
        
    fig.suptitle('Sorted adjacency matrices for ' + dataset + ' data for different graph types,\n n_rois='+str(n_rois)+', 6.25th percentile threshold', fontsize=title_fontsize, weight='bold')
    #cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.95)
    plt.savefig(main_dir+'/figures/'+dataset+'_data_sorted.png', bbox_inches='tight')
    
    
############################################################ Classification functions ############################################################
def get_x_data(eta):
    eta_triu_list = []
    for s in range(eta.shape[-1]):
        eta_triu_mat = np.triu(eta[:,:,s])
        eta_triu_vec = eta_triu_mat[np.triu_indices(eta_triu_mat.shape[-1])]
        eta_triu_list.append(eta_triu_vec)
    return np.array(eta_triu_list) # shape: n_samles x n_features

