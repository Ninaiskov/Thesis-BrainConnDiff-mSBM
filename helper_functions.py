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
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from ast import literal_eval
from collections import Counter
from scipy.signal import butter, filtfilt
from scipy.stats import entropy
from sklearn.metrics.cluster import normalized_mutual_info_score

from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, cross_validate, learning_curve, StratifiedKFold, StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import confusion_matrix, make_scorer, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

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
        Z = np.kron(np.eye(K),np.ones((Nc,1)))
    elif Nc_type == 'unbalanced': 
        if K == 2:
            Nc_list = [70, 30]
        elif K == 5:
            Nc_list = [60, 20, 10, 5, 5]
        elif K == 10:
            Nc_list = [20, 20, 10, 10, 10, 10, 5, 5, 5, 5]
        else:
            print('Nc_list not specfied for chosen K')

        Z = np.zeros((N, K))
        for k in range(K): # len(Nc_list) = K
            Nc = Nc_list[k]
            cumsumNc = int(np.sum(Nc_list[:k]))
            Z[cumsumNc:cumsumNc+Nc, k] = 1
    else:
        print('Unknown Nc_type')
    
    # 2) compute block/cluster interactions (eta)
        # same, completely or partially different
    eta1 = np.load(os.path.join(main_dir,'data','synthetic','eta1_K'+str(K)+'.npy')) #np.random.rand(K, K)
    if eta_similarity == 'same':
        eta2 = eta1.copy()
    elif eta_similarity == 'comp_diff':
        eta2 = np.load(os.path.join(main_dir,'data','synthetic','eta2_K'+str(K)+'.npy')) #np.random.rand(K, K)
    elif eta_similarity == 'part_diff':
        #print('Using partially different etas')
        eta2 = eta1.copy()
        if K == 2:
            eta2[0,0] = np.load(os.path.join(main_dir,'data','synthetic','eta2_K2_randvals.npy'))#np.random.rand(1,1)
        elif K == 5:
            eta2[:3,:3] = np.load(os.path.join(main_dir,'data','synthetic','eta2_K5_randvals.npy'))#np.random.rand(3,3)
        elif K == 10:
            eta2[:5,:5] = np.load(os.path.join(main_dir,'data','synthetic','eta2_K10_randvals.npy'))#np.random.rand(5,5)
        else:
            print('eta2 not specfied for chosen K')
    else:
        print('eta_similarity not specified') 
    
    # 3) compute adjacency matrices (A)
    A = np.empty((N, N, S1+S2))
    A.fill(np.nan)
    M1 = Z @ eta1 @ Z.T
    M2 = Z @ eta2 @ Z.T
    for s in range(S1+S2):
        if s <= S1-1:
            #At = M1 > np.load(os.path.join(main_dir,'data','synthetic','S1_N100_randthres.npy')) #np.random.rand(N, N)
            if S1 == 5:
                At = M1 > np.load(os.path.join(main_dir,'data','synthetic','S1_5_N100_randthres.npy'))[:,:,s]
            elif S1 == 10:
                At = M1 > np.load(os.path.join(main_dir,'data','synthetic','S1_10_N100_randthres.npy'))[:,:,s]
            else:
                print('Unknown S1 value')
            A[:,:,s] = np.triu(At, 1) + np. triu(At, 1).T
        else:
            #At = M2 > np.load(os.path.join(main_dir,'data','synthetic','S2_N100_randthres.npy')) #np.random.rand(N, N)
            At = M2 > np.load(os.path.join(main_dir,'data','synthetic','S2_N100_randthres.npy'))[:,:,s-S1]
            A[:,:,s] = np.triu(At, 1) + np. triu(At, 1).T

    if disp_data:
        fig, ax = plt.subplots()
        cmap_binary = ListedColormap(['k', 'w']) 
        im = ax.imshow(Z, interpolation='nearest', aspect='auto', cmap=cmap_binary, extent=(0, Z.shape[1], 0, Z.shape[0]))
        ax.set_ylabel('Node', fontsize=label_fontsize)
        ax.set_xlabel('Cluster', fontsize=label_fontsize)
        ax.set_title('Partition $Z$', fontsize=title_fontsize, weight='bold')
        cbar = fig.colorbar(im, ax=ax, shrink=0.7, ticks=[0,1])
        plt.show()

        cmap = plt.cm.Purples
        fig, ax = plt.subplots(1,2)
        plt.subplots_adjust(wspace=0.3, top=1.22)
        im = ax[0].imshow(eta1, cmap=cmap)
        ax[0].set_ylabel('Cluster', fontsize=label_fontsize)
        ax[0].set_xlabel('Cluster', fontsize=label_fontsize)
        ax[0].set_title('$\eta_1$', fontsize=subtitle_fontsize, weight='bold')
        im = ax[1].imshow(eta2, cmap=cmap)
        ax[1].set_ylabel('Cluster', fontsize=label_fontsize)
        ax[1].set_xlabel('Cluster', fontsize=label_fontsize)
        ax[1].set_title('$\eta_2$', fontsize=subtitle_fontsize, weight='bold')
        cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.3)
        fig.suptitle('Cluster-link probability matrices',fontsize=title_fontsize, weight='bold')
        plt.show()

        fig, ax = plt.subplots(1,2)
        plt.subplots_adjust(wspace=0.3, top=1.1)
        im = ax[0].imshow(M1, cmap = cmap)
        ax[0].set_ylabel('Node', fontsize=label_fontsize)
        ax[0].set_xlabel('Node', fontsize=label_fontsize)
        ax[0].set_title('$M_1$', fontsize=subtitle_fontsize, weight='bold')
        im = ax[1].imshow(M2, cmap = cmap)
        ax[1].set_ylabel('Node', fontsize=label_fontsize)
        ax[1].set_xlabel('Node', fontsize=label_fontsize)
        ax[1].set_title('$M_2$', fontsize=subtitle_fontsize, weight='bold')
        cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.3)
        #fig.suptitle('$C$',fontsize=15, weight='bold')
        plt.show()

        #default_blue = '#1f77b4'
        #cmap_binary = ListedColormap(['white', default_blue]) 
        map_values1 = [0,1]
        colormap = plt.cm.Blues
        cmap1 = plt.cm.colors.ListedColormap(colormap(np.linspace(0, 1, len(map_values1))))

        fig, ax = plt.subplots(1,2)
        plt.subplots_adjust(wspace=0.3, top=1.1)
        im = ax[0].imshow(A[:,:,0],cmap=cmap1)
        ax[0].set_ylabel('Node', fontsize=label_fontsize)
        ax[0].set_xlabel('Node', fontsize=label_fontsize)
        ax[0].set_title('$A_{type1}$', fontsize=subtitle_fontsize, weight='bold')
        im = ax[1].imshow(A[:,:,-1], cmap=cmap1)
        ax[1].set_ylabel('Node', fontsize=label_fontsize)
        ax[1].set_xlabel('Node', fontsize=label_fontsize)
        ax[1].set_title('$A_{type2}$', fontsize=subtitle_fontsize , weight='bold')
        cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.3, ticks=[0, 1])
        fig.suptitle('Adjacency matrices',fontsize=title_fontsize, weight='bold')
        plt.show()
        
    return A, Z, eta1, eta2, M1, M2
    

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
 
# Function for filtering a signal given low and high cut for bandpass filter with Frequencies
# The filter is of order 10 since it is a two-pass with 5-order filter.    
def butter_bandpass(lowcut, highcut, fs, order=5):
    # Define filter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="bandpass")
    return b, a

def butter_bandpass_filter(data, lowcut=0.009, highcut=0.08, fs=1 / 2.490, order=5):
    # OBS! CHANGE FS TO THE APPROPRIATE SAMPLING RATE
    # Apply Filter
    # Changed by Anders to be correct range (from 0.01-0.1 to 0.009-0.08)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

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

'''
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
'''

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

def get_syn_nmi(exp_paths, K, Nc_type, eta_similarity, main_dir=main_dir):
    
    Z_exp = np.load(os.path.join(main_dir,'data','synthetic','Zexp_K'+str(int(K))+'_'+Nc_type+'_'+eta_similarity+'.npy'))
    nmi_list = []
    for folder in exp_paths:
        sample = np.load(os.path.join(folder, 'model_sample.npy'), allow_pickle=True).item()
        Z_MAP = sample['MAP']['Z'].T
        labels_MAP = Z_MAP.argmax(axis=1)
        labels_exp = Z_exp.argmax(axis=1)
        nmi = normalized_mutual_info_score(labels_true=labels_exp, labels_pred=labels_MAP)
        nmi_list.append(nmi)
    
    return nmi_list

def plot_par(dataset, df, par, miniter_gibbs=None, maxiter_gibbs=None, main_dir=main_dir, label_fontsize=label_fontsize, subtitle_fontsize=subtitle_fontsize, title_fontsize=title_fontsize, fig_name=None):
    
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
    
    plt.title(par + ' - ' + dataset, fontsize=title_fontsize, weight='bold')
    plt.ylabel(par, fontsize=label_fontsize)
    plt.xlabel('Gibbs iterations', fontsize=label_fontsize)
    plt.legend(loc='upper right',fontsize='small', fancybox=True, shadow=True, bbox_to_anchor=(1.4, 0.85))
    if fig_name is not None:
        plt.savefig(main_dir+'/figures/'+dataset+'_'+fig_name+'.png', bbox_inches='tight') 
    else:
        plt.savefig(main_dir+'/figures/'+dataset+'_plot_'+par+'.png', bbox_inches='tight')    

def boxplot_par(dataset, df, par, main_dir=main_dir, label_fontsize=label_fontsize, subtitle_fontsize=subtitle_fontsize, title_fontsize=title_fontsize, fig_name=None):
    
    # Input:  
    # dataset: 'hcp' or 'decnef'
    # df: dataframe with experiment overview
    # par: parameter to plot as a function of Gibbs iterations, e.g. 'logP' or 'noc'
    
    # Output: plot of parameter as a function of Gibbs iterations

    # Sort by dataset and relevant columns
    if dataset == 'hcp' or dataset == 'decnef':
        df_dataset = df.loc[df.dataset==dataset].sort_values(by=['n_rois','threshold_annealing','model_type','splitmerge','noc'], ascending=True).reset_index()
        unique_models = df_dataset[['model_type', 'threshold_annealing', 'n_rois', 'noc']].drop_duplicates()
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
        
        if model_type == 'parametric':
            model_type_short = 'p'
        elif model_type == 'nonparametric':
            model_type_short = 'np'
        else:
            print('Unknown model type')
        if dataset == 'hcp' or dataset == 'decnef':
            color_intensity = n_rois / df_dataset['n_rois'].max()  # Higher n_rois, darker intensity
            label = model_type_short+'_'+str(n_rois)+'_'+str(noc)
            #label = f"{model_type}_nrois{n_rois}_noc{noc}"
        elif dataset == 'synthetic':
            color_intensity = K / df_dataset['K'].max()  # Higher K, darker intensity
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
    ax.set_ylabel(par + ' distribution', fontsize=label_fontsize)
    ax.set_title('Boxplots of ' + par + ' - ' + dataset, fontsize=title_fontsize, weight='bold')
    ax.yaxis.grid(True)
    if fig_name is not None:
        plt.savefig(main_dir+'/figures/'+dataset+'_'+fig_name+'.png', bbox_inches='tight')    
    else:
        plt.savefig(main_dir+'/figures/'+dataset+'_boxplot_'+par+'.png', bbox_inches='tight')    

def old_plot_par(dataset, exp1_folders, exp2_folders, par, exp1_label, exp2_label, results_folder = 'results', plot1=True, plot2=True, miniter_gibbs=None, maxiter_gibbs=None, main_dir=main_dir, fig_name=None, ylim=None):
    
    # Input: 
    # exp_folder_lists: list of experiment folder names (1 is without annenaling, 2 is with annealing)
    # par: parameter to plot as a function of Gibbs iterations, e.g. 'logP' or 'noc'
    
    # Output: plot of parameter as a function of Gibbs iterations
    
    exp1_paths = [os.path.join(main_dir,results_folder,dataset,folder) for folder in exp1_folders]
    exp2_paths = [os.path.join(main_dir,results_folder,dataset,folder) for folder in exp2_folders]
    
    plt.figure()
    #for i in range(len(exp_folder_list)):
    #exp_folder = exp_folder_list[i]
    if plot1:
        _, _, mean_par, min_par, max_par = get_stats(exp1_paths, par)
        iters = range(len(mean_par))
        if miniter_gibbs is None:
            miniter_gibbs = iters[0]
        if maxiter_gibbs is None:
            maxiter_gibbs = iters[-1]
        #plt.errorbar(iters, mean_logP, yerr=max_logP-min_logP, fmt='-o', capsize=5, label=var)
        plt.plot(iters[miniter_gibbs:maxiter_gibbs], mean_par[miniter_gibbs:maxiter_gibbs],'-bo', label=exp1_label)
        plt.fill_between(iters[miniter_gibbs:maxiter_gibbs], min_par[miniter_gibbs:maxiter_gibbs], max_par[miniter_gibbs:maxiter_gibbs], alpha=0.5, color='blue')

    if plot2:
        _, _, mean_par, min_par, max_par = get_stats(exp2_paths, par)
        iters = range(len(mean_par))
        #miniter_gibbs = iters[0]
        #maxiter_gibbs = iters[-1]
        #plt.errorbar(iters, mean_logP, yerr=max_logP-min_logP, fmt='-o', capsize=5, label=var)
        plt.plot(iters[miniter_gibbs:maxiter_gibbs], mean_par[miniter_gibbs:maxiter_gibbs],'-ro', label=exp2_label)
        plt.fill_between(iters[miniter_gibbs:maxiter_gibbs], min_par[miniter_gibbs:maxiter_gibbs], max_par[miniter_gibbs:maxiter_gibbs], alpha=0.5, color='red')

    plt.title(par + ' - ' + dataset, fontsize=title_fontsize, weight='bold')
    plt.ylabel(par, fontsize=label_fontsize)
    plt.xlabel('Gibbs iterations', fontsize=label_fontsize)
    plt.legend()
    plt.ylim(ylim)
    if fig_name is not None:
        plt.savefig(main_dir+'/figures/'+dataset+'_'+fig_name+'.png', bbox_inches='tight')    
    else:
        plt.savefig(main_dir+'/figures/'+dataset+'_old_plot_'+par+'.png', bbox_inches='tight')    
    
    
def plot_eta(dataset, eta, target_type, exp_name_title=None, main_dir=main_dir, label_fontsize=label_fontsize, subtitle_fontsize=subtitle_fontsize, title_fontsize=title_fontsize):
    if dataset == 'hcp':
        eta_type1 = eta[:,:,:250] # fmri
        eta_type2 = eta[:,:,250:] # dmri
    elif dataset == 'decnef':
        if target_type == 'schizo':
            schizo_mask = np.load(main_dir+'/data/decnef/schizo_mask.npy')
            healthy_mask = (schizo_mask == False)
            eta_type1 = eta[:,:,np.nonzero(healthy_mask)[0]]
            eta_type2 = eta[:,:,np.nonzero(schizo_mask)[0]]
        elif target_type == 'male':
            male_mask = np.load(main_dir+'/data/decnef/male_mask.npy')
            female_mask = (male_mask == False)
            eta_type1 = eta[:,:,np.nonzero(female_mask)[0]]
            eta_type2 = eta[:,:,np.nonzero(male_mask)[0]]
        else:
            print('Unknown target type')
        ''' # need to rewrite plotting function to handle more than 2 types
        elif target_type == 'hand':
            hand_labels = np.load(main_dir+'/data/decnef/hand_mask.npy')
            hand0_mask = (hand_labels == 0)
            hand1_mask = (hand_labels == 1)
            hand2_mask = (hand_labels == 2)
            eta_type0 = eta[:,:,hand0_mask]
            eta_type1 = eta[:,:,hand1_mask]
            eta_type2 = eta[:,:,hand2_mask]
        elif target_type 'site':
            site_labels = np.load(main_dir+'/data/decnef/site_mask.npy')
            site0_mask = (site_labels == 0)
            site1_mask = (site_labels == 1)
            site2_mask = (site_labels == 2)
            site3_mask = (site_labels == 3)
            site4_mask = (site_labels == 4)
            site5_mask = (site_labels == 5)
            site6_mask = (site_labels == 6)
            site7_mask = (site_labels == 7)
            site8_mask = (site_labels == 8)
            site9_mask = (site_labels == 9)
            site10_mask = (site_labels == 10)
            eta_type0 = eta[:,:,site0_mask]
            eta_type1 = eta[:,:,site1_mask]
            eta_type2 = eta[:,:,site2_mask]
            eta_type3 = eta[:,:,site3_mask]
            eta_type4 = eta[:,:,site4_mask]
            eta_type5 = eta[:,:,site5_mask]
            eta_type6 = eta[:,:,site6_mask]
            eta_type7 = eta[:,:,site7_mask]
            eta_type8 = eta[:,:,site8_mask]
            eta_type9 = eta[:,:,site9_mask]
            eta_type10 = eta[:,:,site10_mask]
        '''
    elif dataset == 'synthetic':
        eta_type1 = eta[:,:,:S1]
        eta_type2 = eta[:,:,S1:S2]
    else: 
        print('unknown dataset')
    
    print(np.array_equal(eta_type2[:,:,0], eta_type2[:,:,1]))

    fig, axs = plt.subplots(2, 5, figsize=(15, 7), constrained_layout=True)
    axs = axs.ravel()
    cmap = plt.cm.Purples
    
    K = eta.shape[0]
    #max_val = np.max(eta)
    xy_ticks = range(0, K + 1, 1)

    for s in range(10):
        if s < 5:
            im = axs[s].imshow(eta_type1[:,:,s], cmap=cmap, extent=[0,K,K,0])#, vmin=0, vmax=max_val)
            #if dataset=='hcp':
            #    axs[s].set_title('Functional: '+str(s+1), fontsize=subtitle_fontsize, weight='bold')
            #else:
            #    axs[s].set_title('Healthy: '+str(s+1), fontsize=subtitle_fontsize, weight='bold')
            axs[s].set_ylabel('Cluster', fontsize=label_fontsize)
            axs[s].set_xlabel('Cluster', fontsize=label_fontsize)
            axs[s].set_yticks(xy_ticks)
            axs[s].set_yticklabels(xy_ticks)
            axs[s].set_xticks(xy_ticks)
            axs[s].set_xticklabels(xy_ticks)
        else:
            im = axs[s].imshow(eta_type2[:,:,-(s-4)], cmap=cmap, extent=[0,K,K,0])#, vmin=0, vmax=max_val)
            #if dataset=='hcp':
            #    axs[s].set_title('Structural: '+str(s-4), fontsize=subtitle_fontsize, weight='bold')
            #else:
            #    axs[s].set_title('Schizo: '+str(s-4), fontsize=subtitle_fontsize, weight='bold')
            axs[s].set_ylabel('Cluster', fontsize=label_fontsize)
            axs[s].set_xlabel('Cluster', fontsize=label_fontsize)
            axs[s].set_yticks(xy_ticks)
            axs[s].set_yticklabels(xy_ticks)
            axs[s].set_xticks(xy_ticks)
            axs[s].set_xticklabels(xy_ticks)

    if dataset=='hcp':
        axs[0].set_title('Functional', fontsize=subtitle_fontsize, weight='bold')
        axs[5].set_title('Structural', fontsize=subtitle_fontsize, weight='bold')
    else: 
        if target_type == 'schizo':
            axs[0].set_title('Healthy', fontsize=subtitle_fontsize, weight='bold')
            axs[5].set_title('Schizophrenia', fontsize=subtitle_fontsize, weight='bold')
        elif target_type == 'male':
            axs[0].set_title('Male', fontsize=subtitle_fontsize, weight='bold')
            axs[5].set_title('Female', fontsize=subtitle_fontsize, weight='bold')
        else:
            print('uknown target type')
    fig.suptitle('Cluster-link probability matrices - '+dataset+',\n Experiment: ' + exp_name_title, fontsize=title_fontsize, weight='bold')
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.95)
    plt.savefig(main_dir+'/figures/'+dataset+'_eta_types_'+target_type+'.png', bbox_inches='tight')    
   
   
def plot_circ_eta(dataset, eta, noc, metric='std', threshold=0, main_dir=main_dir, label_fontsize=label_fontsize, subtitle_fontsize=subtitle_fontsize, title_fontsize=title_fontsize):
    # threshold = link probability threshold (e.g based on np.percentile(eta_example.flatten(),99)))
    if metric == 'std':
        eta_metric = np.std(eta, axis=2)
    elif metric == 'KL_div': # KL divergence = relative entropy
        eta_metric = entropy(eta, np.ones(eta.shape), axis=2) # relative to uniform distribution

    eta_example = np.triu(eta_metric) # upper triangular matrix
    # eta_example = np.triu(eta[:,:,example_sub])

    # create a graph from the matrix
    G = nx.from_numpy_array(eta_example)

    # remove self-edges
    #G.remove_edges_from(nx.selfloop_edges(G))

    # set edge colors based on the edge weights
    edge_colors = [eta_example[i, j] for i, j in zip(*np.where(np.triu(eta_example) > threshold))]
    edge_cmap = plt.cm.Purples

    # draw the graph in a circular layout
    pos = nx.circular_layout(G)
    node_colors = sns.color_palette('hls', noc)

    # add only edges above the threshold
    G_thresh = nx.Graph()
    for i in range(noc):
        G_thresh.add_node(i)
    for i, j in zip(*np.where(np.triu(eta_example) > threshold)):
        G_thresh.add_edge(i, j, weight=eta_example[i, j])

    nx.draw_circular(G_thresh, with_labels=True, node_color=node_colors, node_size=300,
                    edge_color=edge_colors, edge_cmap=edge_cmap, width=2,
                    font_size=label_fontsize)

    # add a colorbar for the edge colors
    sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=eta_example.min(), vmax=eta_example.max()))
    sm.set_array([])
    plt.colorbar(sm)

    # show the plot
    if metric == 'std':
        plt.title('Std. of cluster-link probabilities - ' + dataset, fontsize=title_fontsize, weight='bold')
        plt.savefig(main_dir+'/figures/'+dataset+'_eta_stdcirc.png', bbox_inches='tight')
    elif metric == 'KL_div':
        plt.title('KL div. of cluster-link probabilities - ' + dataset, fontsize=title_fontsize, weight='bold')
        plt.savefig(main_dir+'/figures/'+dataset+'_eta_KLdivcirc.png', bbox_inches='tight')
    else: 
        print('unknown metric')
     
        
def plot_sortedA(dataset, Z, noc, n_rois, exp_name_title, atlas_name='schaefer', main_dir=main_dir, label_fontsize=label_fontsize, subtitle_fontsize=subtitle_fontsize, title_fontsize=title_fontsize):
    graph_no = 4 # since all experiments was run with graph_no=4
    # Unique graph values
    map_values1 = [0, 1] # 2 discrete values
    map_values2 = [0, 1, 2, 3] # 4 discrete values
    map_values3 = [0, 1, 2, 3, 4, 5, 6, 7] # 8 discrete values
    map_values4 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # 16 discrete values

    # Choose a colormap
    colormap = cm.Blues # desired colormap

    # Create colormaps for each graph density type
    cmap1 = cm.colors.ListedColormap(colormap(np.linspace(0, 1, len(map_values1))))
    cmap2 = cm.colors.ListedColormap(colormap(np.linspace(0, 1, len(map_values2))))
    cmap3 = cm.colors.ListedColormap(colormap(np.linspace(0, 1, len(map_values3))))
    cmap4 = cm.colors.ListedColormap(colormap(np.linspace(0, 1, len(map_values4))))

    if graph_no == 1:
        cmap = cmap1
    elif graph_no == 2:
        cmap = cmap2
    elif graph_no == 3:
        cmap = cmap3
    elif graph_no == 4:
        cmap = cmap4
    
    data_folder_name = atlas_name+str(n_rois)
    folder_path = os.path.join(main_dir, 'data/'+dataset+'/'+data_folder_name)

    # compute A
    A_vals_list = np.load(os.path.join(folder_path,'A'+str(graph_no)+'_vals_list.npy'))
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
    #colors = [] # cluster colors
    #for i in range(len(count)):
    #    colors.append('#%06X' % np.random.randint(0, 0xFFFFFF))
    colors = sns.color_palette('hls', noc)
    
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
    for s in range(10):
        if s < 5:
            im = axs[s].imshow(A_type1[:,:,s], cmap=cmap)
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
                axs[s].add_patch(plt.Rectangle((last_val,last_val), size, size, fc=colors[i], ec=colors[i], linewidth=2, fill=False))
                last_val = x
        else:
            im = axs[s].imshow(A_type2[:,:,-(s-4)], cmap=cmap)
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
                axs[s].add_patch(plt.Rectangle((last_val,last_val), size, size, fc=colors[i], ec=colors[i], linewidth=2, fill=False))
                last_val = x

    if dataset=='hcp':
        axs[0].set_title('Functional', fontsize=subtitle_fontsize, weight='bold')
        axs[5].set_title('Structural', fontsize=subtitle_fontsize, weight='bold')
    else: 
        axs[0].set_title('Healthy', fontsize=subtitle_fontsize, weight='bold')
        axs[5].set_title('Schizophrenia', fontsize=subtitle_fontsize, weight='bold')
        
    #fig.suptitle('Sorted adjacency matrices for ' + dataset + ' data for different graph types,\n n_rois='+str(n_rois)+', 6.25th percentile threshold', fontsize=title_fontsize, weight='bold')
    fig.suptitle('Sorted adjacency matrices wrt. partition - ' + dataset + ',\n Experiment: '+exp_name_title, fontsize=title_fontsize, weight='bold')
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.9)
    plt.savefig(main_dir+'/figures/'+dataset+'_data_sorted.png', bbox_inches='tight')
    
    
############################################################ Classification functions ############################################################
def get_X_data(eta):
    eta_triu_list = []
    for s in range(eta.shape[-1]):
        eta_triu_mat = np.triu(eta[:,:,s])
        eta_triu_vec = eta_triu_mat[np.triu_indices(eta_triu_mat.shape[-1])]
        eta_triu_list.append(eta_triu_vec)
    return np.array(eta_triu_list) # shape: n_samles x n_features

def classification_pipeline(clf, param_grid, target_type, feature_type, dataset, n_rois=None, eta=None, class_weight = 'balanced', main_dir='/work3/s174162/speciale'):
    # pipeline for nested cross validation of given classifer, target_type and feature_type
    
    atlas_name = 'schaefer'
    # create target/label data: y
    if dataset == 'hcp':
        y = np.load(os.path.join(main_dir, 'data', 'hcp', target_type+'_mask.npy'))
        if feature_type == 'demoinfo':
            demoinfo_feat = np.load(os.path.join(main_dir,'data',dataset,'demoinfo_'+target_type+'.npy'), allow_pickle=True)
            X = demoinfo_feat
        elif feature_type == 'corr':
            corr_sublist = np.load(os.path.join(main_dir,'data',dataset, atlas_name+str(n_rois),'corr_sublist.npy'), allow_pickle=True)
            if target_type == 'male':
                demoinfo_feat = np.load(os.path.join(main_dir,'data',dataset,'demoinfo_'+target_type+'.npy'), allow_pickle=True)
                corr_feat = np.concatenate((corr_sublist[:250,:],corr_sublist[:250,:]),1) # concatenated correlation matrix for structural an functional connectivity
                X = np.concatenate([demoinfo_feat, corr_feat], axis=1)
            elif target_type == 'structconn':
                X = corr_sublist # correlation matrix for both structural an functional connectivity
            else:
                print('Unknown target type')
        elif feature_type == 'eta':
            if target_type == 'male':
                eta_feat = get_X_data(eta)
                X = np.concatenate([eta_feat[:250,:], eta_feat[250:,:]], axis=1) # stacking functional and structural eta features for each subject when predicting gender
            elif target_type == 'structconn':
                X = get_X_data(eta)
            else:
                print('Unknown target type')
        else:
            print('Unknown feature type')
    elif dataset == 'decnef':
        y = np.load(os.path.join(main_dir, 'data', 'decnef', target_type+'_mask.npy')) 
        demoinfo_feat = np.load(os.path.join(main_dir,'data',dataset,'demoinfo_'+target_type+'.npy'), allow_pickle=True)
        if feature_type == 'demoinfo':
            X = demoinfo_feat
        elif feature_type == 'demoinfo_corr':  
            corr_feat = np.load(os.path.join(main_dir,'data',dataset, atlas_name+str(n_rois),'corr_sublist.npy'), allow_pickle=True)
            X = np.concatenate([demoinfo_feat, corr_feat], axis=1)
        elif feature_type == 'demoinfo_eta':
            eta_feat = get_X_data(eta)
            X = np.concatenate([demoinfo_feat, eta_feat], axis=1)
        else:
            print('Unknown feature type')
    else:
        print('dataset not found')
    #print('y shape:', y.shape)

    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) # note that we use the same scaler for test data (we don't fit to test data)

    if (target_type == 'site') or (target_type == 'hand'): # multiclass target types
        #roc_auc_weighted = make_scorer(roc_auc_score, average='weighted', multi_class='ovr')#, needs_proba=True)
        #ap_weighted = make_scorer(average_precision_score, average='weighted', needs_proba=True)
        #scoring = {'accuracy': 'accuracy', 'roc_auc': roc_auc_weighted, 'f1': 'f1_weighted'}  # scoring metrics for cross validation
        scoring = {'accuracy': 'accuracy', 'f1': 'f1_weighted'}  # scoring metrics for cross validation
    else: # binary target types
        scoring = ['accuracy','f1'] # scoring metrics for cross validation # could also evaluate average precision and auc-roc, but I just stick with accuracy and F1-score instead so its the same for all target types
    refit_metric = 'f1'
    cv_outer = StratifiedShuffleSplit(n_splits=5, test_size=test_size, random_state=0)
    cv_inner = StratifiedShuffleSplit(n_splits=3, test_size=test_size, random_state=0)

    #pipe = Pipeline([('scaler', scaler), ('clf', clf)])
    pipe = make_pipeline(scaler, clf) # also possible to use Pipeline: https://stackoverflow.com/questions/40708077/what-is-the-difference-between-pipeline-and-make-pipeline-in-scikit-learn

    # define serach
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv_inner, scoring=scoring, n_jobs=-1, refit=refit_metric) #When specifying multiple metrics, the refit parameter must be set to the metric (string) for which the best_params_ will be found and used to build the best_estimator_ on the whole dataset.

    #val_scores = cross_val_score(gs, X_train, y_train, cv=cv_outer, scoring=scoring, n_jobs=-1)
    scores = cross_validate(gs, X_train, y_train, cv=cv_outer, scoring=scoring, n_jobs=-1, return_train_score=True)
    
    # Access the best hyperparameters for each fold
    gs.fit(X_train, y_train)
    best_params = gs.best_params_ # Access the best estimator params
   
    val_accuracy = scores['test_accuracy']
    #val_AP = scores['test_average_precision']
    #val_AUC = scores['test_roc_auc']
    val_F1 = scores['test_f1']

    #return np.mean(val_accuracy), np.mean(val_AUC), np.mean(val_AP), np.mean(val_F1), best_params
    return np.mean(val_accuracy), np.mean(val_F1), best_params

def classification_results(dataset, df, target_type, feature_type, clf, clf_name, param_grid, class_weight = 'balanced', main_dir='/work3/s174162/speciale', label_fontsize=label_fontsize, subtitle_fontsize=subtitle_fontsize, title_fontsize=title_fontsize, fig_name=None):
    # Input:  
    # dataset: 'hcp' or 'decnef'
    # df: dataframe with experiment overview

    # Output: plot of parameter as a function of Gibbs iterations

    # Sort by dataset and relevant columns
    if dataset == 'hcp' or dataset == 'decnef':
        df_dataset = df.loc[df.dataset==dataset].sort_values(by=['n_rois','threshold_annealing','model_type','splitmerge','noc'], ascending=True).reset_index()
        unique_models = df_dataset[['model_type', 'threshold_annealing', 'n_rois', 'noc']].drop_duplicates()
    elif dataset == 'synthetic':
        df_dataset = df.loc[df.dataset==dataset].sort_values(by=['N','K','S1','S2','Nc_type', 'eta_similarity', 'model_type','splitmerge','noc'], ascending=True).reset_index()
        unique_models = df_dataset[['model_type', 'K', 'S1', 'Nc_type', 'eta_similarity', 'noc']].drop_duplicates()
    else:
        print('unknown dataset')

    #unique_models = unique_models.loc[[0,1]] # TESTING

    cmap = plt.get_cmap('viridis')

    noc_values = df_dataset['noc'].unique()
    num_noc = len(noc_values)

    label_list = []
    color_list = []
    acc_data = []
    #AUC_data = []
    #AP_data = []
    F1_data = []
    best_params_data = []
    for _, row in unique_models.iterrows():
        
        model_type = row['model_type']
        noc = row['noc']
        if dataset == 'hcp' or dataset == 'decnef':
            threshold_annealing = row['threshold_annealing']
            n_rois = row['n_rois']
            model_df = df_dataset.loc[(df_dataset['model_type'] == model_type) & (df_dataset['threshold_annealing'] == threshold_annealing) & (df_dataset['n_rois'] == n_rois) & (df_dataset['noc'] == noc)]
            
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
        
        if model_type == 'parametric':
            model_type_short = 'p'
        elif model_type == 'nonparametric' and threshold_annealing == 'True':
            model_type_short = 'np_anneal'
        elif model_type == 'nonparametric' and threshold_annealing == 'False':
            model_type_short = 'np'
        else:
            print('Unknown model type')
        if dataset == 'hcp' or dataset == 'decnef':
            color_intensity = n_rois / df_dataset['n_rois'].max()  # Higher n_rois, darker intensity
            #label = f"{model_type}_nrois{n_rois}_noc{noc}"
            label = model_type_short+str(int(n_rois))+'_'+str(noc)
        elif dataset == 'synthetic':
            color_intensity = K / df_dataset['K'].max()  # Higher K, darker intensity
            label = f"{model_type}_K{K}_S1{S1}_Nc_type{Nc_type}_eta_similarity{eta_similarity}_noc{noc}"
        else:
            print('unknown dataset')
        label_list.append(label)
        
        color = cmap((color_intensity + noc_color_index) / num_noc)
        color_list.append(color)
        
        ####### FOR EACH FEATURE MODEL: EXTRACT FEATURES (ETA) AND PERFORM CROSS VALIDATION TO GET BEST MODEL AND ACCURACY SCORE #######
        for _, model_row in model_df.iterrows():
            folders = model_row['exp_name_list']
            paths = [os.path.join(main_dir, 'results', dataset, folder) for folder in folders]
            acc_list = []
            #AUC_list = []
            #AP_list = []
            F1_list = []
            best_params_list = []
            for folder in paths:
                sample = np.load(os.path.join(folder, 'model_sample.npy'), allow_pickle=True).item()
                eta = sample['MAP']['eta']
                #mean_accuracy, mean_AUC, mean_AP, mean_F1, best_params = classification_pipeline(clf=clf, param_grid=param_grid, target_type=target_type, feature_type=feature_type, dataset=dataset, eta=eta, class_weight = class_weight) # mean cross-validated score of best estimater
                mean_accuracy, mean_F1, best_params = classification_pipeline(clf=clf, param_grid=param_grid, target_type=target_type, feature_type=feature_type, dataset=dataset, eta=eta, class_weight = class_weight) # mean cross-validated score of best estimater
                acc_list.append(mean_accuracy)
                #AUC_list.append(mean_AUC)
                #AP_list.append(mean_AP)
                F1_list.append(mean_F1)
                best_params_list.append(best_params) # probably not neccessary to save this since the optimal hyperparameters for different initializations of the same feature model probably is the same
            best_acc_idx = np.argmax(acc_list) # best params according to refit parameter <- WHICH ONE?
            best_params_data.append(best_params_list[best_acc_idx]) # saving optimal parameters for random initialization with highest accuracy score
            acc_data.append(acc_list)
            #AUC_data.append(AUC_list)
            #AP_data.append(AP_list)
            F1_data.append(F1_list)
            best_params_data.append(best_params)

    # Baseline results
    if dataset == 'hcp':
        corr100_results = np.load(os.path.join(main_dir,'results',dataset,target_type+'_corr100_results.npy'), allow_pickle=True).item()
        corr200_results = np.load(os.path.join(main_dir,'results',dataset,target_type+'_corr200_results.npy'), allow_pickle=True).item()
        corr300_results = np.load(os.path.join(main_dir,'results',dataset,target_type+'_corr300_results.npy'), allow_pickle=True).item()
        if target_type == 'male':
            demoinfo_results = np.load(os.path.join(main_dir,'results',dataset,target_type+'_demoinfo_results.npy'), allow_pickle=True).item()  
            mean_accuracy_demoinfo = demoinfo_results[clf_name][0]
            #mean_AUC_demoinfo = demoinfo_results[clf_name][1]
            #mean_AP_demoinfo = demoinfo_results[clf_name][2]
            mean_F1_demoinfo = demoinfo_results[clf_name][1]
    elif dataset == 'decnef':
        demoinfo_results = np.load(os.path.join(main_dir,'results',dataset,target_type+'_demoinfo_results.npy'), allow_pickle=True).item()  
        corr100_results = np.load(os.path.join(main_dir,'results',dataset,target_type+'_demoinfo_corr100_results.npy'), allow_pickle=True).item()
        corr200_results = np.load(os.path.join(main_dir,'results',dataset,target_type+'_demoinfo_corr200_results.npy'), allow_pickle=True).item()
        corr300_results = np.load(os.path.join(main_dir,'results',dataset,target_type+'_demoinfo_corr300_results.npy'), allow_pickle=True).item()
        
        mean_accuracy_demoinfo = demoinfo_results[clf_name][0]
        #mean_AUC_demoinfo = demoinfo_results[clf_name][1]
        #mean_AP_demoinfo = demoinfo_results[clf_name][2]
        mean_F1_demoinfo = demoinfo_results[clf_name][1]
    else:
        print('Unknown dataset')   
    
    mean_accuracy_corr100 = corr100_results[clf_name][0]
    #mean_AUC_corr100 = corr100_results[clf_name][1]
    #mean_AP_corr100 = corr100_results[clf_name][2]
    mean_F1_corr100 = corr100_results[clf_name][1]
    
    mean_accuracy_corr200 = corr200_results[clf_name][0]
    #mean_AUC_corr200 = corr200_results[clf_name][1]
    #mean_AP_corr200 = corr200_results[clf_name][2]
    mean_F1_corr200 = corr200_results[clf_name][1]
    
    mean_accuracy_corr300 = corr300_results[clf_name][0]
    #mean_AUC_corr300 = corr300_results[clf_name][1]
    #mean_AP_corr300 = corr300_results[clf_name][2]
    mean_F1_corr300 = corr300_results[clf_name][1]
  
    # Accuracy boxplot
    shift = 1
    rotation = 70
    ha = 'right'
    ylim_low = 0
    fig, ax = plt.subplots()
    bp = ax.boxplot(acc_data, patch_artist=True)#, boxprops=dict(facecolor=color_list))
    ax.set_xticks(np.arange(len(label_list))+shift)
    ax.set_xticklabels(label_list, rotation=rotation, ha=ha)
    for box, color in zip(bp['boxes'], color_list):
        box.set_facecolor(color)
    ax.set_ylabel('Accuracy', fontsize=label_fontsize)
    ax.set_title(dataset + ' - ' + clf_name +' - '+ target_type, fontsize=title_fontsize, weight='bold')
    ax.yaxis.grid(True)
    if dataset == 'hcp':
        ax.set_ylim(ylim_low,1) # set ylim
        if target_type == 'structconn':
            line1 = ax.axhline(y=mean_accuracy_corr100, color='darkorange', linestyle='--', label='corr100')
            line2 = ax.axhline(y=mean_accuracy_corr200, color='fuchsia', linestyle='--', label='corr200')
            line3 = ax.axhline(y=mean_accuracy_corr300, color='r', linestyle='--', label='corr300')
            ax.legend(handles=[line1,line2,line3])
        elif target_type == 'male':
            line1 = ax.axhline(y=mean_accuracy_corr100, color='darkorange', linestyle='--', label='demoinfo+corr100')
            line2 = ax.axhline(y=mean_accuracy_corr200, color='fuchsia', linestyle='--', label='demoinfo+corr200')
            line3 = ax.axhline(y=mean_accuracy_corr300, color='r', linestyle='--', label='demoinfo+corr300')
            line4 = ax.axhline(y=mean_accuracy_demoinfo, color='lightpink', linestyle='--', label='demoinfo')
            ax.legend(handles=[line1,line2,line3,line4])
        else:
            print('Unknown target type')
    elif dataset == 'decnef':
        ax.set_ylim(ylim_low,1) # set ylim
        # add baseline results
        line1 = ax.axhline(y=mean_accuracy_demoinfo, color='lightpink', linestyle='--', label='demoinfo')
        line2 = ax.axhline(y=mean_accuracy_corr100, color='darkorange', linestyle='--', label='demoinfo+corr100')
        line3 = ax.axhline(y=mean_accuracy_corr200, color='fuchsia', linestyle='--', label='demoinfo+corr200')
        line4 = ax.axhline(y=mean_accuracy_corr300, color='r', linestyle='--', label='demoinfo+corr300')
        ax.legend(handles=[line1,line2,line3,line4])
    else:
        print('Unknown dataset')
    if fig_name is not None:
        plt.savefig(main_dir+'/figures/'+dataset+'_boxplot_accuracy_'+fig_name+'.png', bbox_inches='tight')    
    else:
        plt.savefig(main_dir+'/figures/'+dataset+'_boxplot_accuracy_'+clf_name+'_'+target_type+'.png', bbox_inches='tight')
    '''
    # ROC-AUC boxplot
    fig, ax = plt.subplots()
    bp = ax.boxplot(AUC_data, patch_artist=True)#, boxprops=dict(facecolor=color_list))
    ax.set_xticks(np.arange(len(label_list))+shift)
    ax.set_xticklabels(label_list, rotation=rotation, ha=ha)
    for box, color in zip(bp['boxes'], color_list):
        box.set_facecolor(color)
    ax.set_ylabel('ROC-AUC', fontsize=label_fontsize)
    ax.set_title(dataset + ' - ' + clf_name +' - '+ target_type, fontsize=title_fontsize, weight='bold')
    ax.yaxis.grid(True)
    if dataset == 'hcp':
        ax.set_ylim(ylim_low,1) # set ylim
        # add baseline results
        if target_type == 'structconn':
            line1 = ax.axhline(y=mean_AUC_corr100, color='darkorange', linestyle='--', label='corr100')
            line2 = ax.axhline(y=mean_AUC_corr200, color='fuchsia', linestyle='--', label='corr200')
            line3 = ax.axhline(y=mean_AUC_corr300, color='r', linestyle='--', label='corr300')
            ax.legend(handles=[line1,line2,line3])
        elif target_type == 'male':
            line1 = ax.axhline(y=mean_AUC_corr100, color='darkorange', linestyle='--', label='demoinfo+corr100')
            line2 = ax.axhline(y=mean_AUC_corr200, color='fuchsia', linestyle='--', label='demoinfo+corr200')
            line3 = ax.axhline(y=mean_AUC_corr300, color='r', linestyle='--', label='demoinfo+corr300')
            line4 = ax.axhline(y=mean_AUC_demoinfo, color='lightpink', linestyle='--', label='demoinfo')
            ax.legend(handles=[line1,line2,line3,line4])
        else:
            print('Unknown target type')
    elif dataset == 'decnef':
        ax.set_ylim(ylim_low,1) # set ylim
        # add baseline results
        line1 = ax.axhline(y=mean_AUC_demoinfo, color='lightpink', linestyle='--', label='demoinfo')
        line2 = ax.axhline(y=mean_AUC_corr100, color='darkorange', linestyle='--', label='demoinfo+corr100')
        line3 = ax.axhline(y=mean_AUC_corr200, color='fuchsia', linestyle='--', label='demoinfo+corr200')
        line4 = ax.axhline(y=mean_AUC_corr300, color='r', linestyle='--', label='demoinfo+corr300')
        ax.legend(handles=[line1,line2,line3,line4])
    else:
        print('Unknown dataset')
    if fig_name is not None:
        plt.savefig(main_dir+'/figures/'+dataset+'_boxplot_AUC_'+fig_name+'.png', bbox_inches='tight')    
    else:
        plt.savefig(main_dir+'/figures/'+dataset+'_boxplot_AUC_'+clf_name+'_'+target_type+'.png', bbox_inches='tight')  
        
    # Average Precision (alternative to PR-AUC) boxplot
    fig, ax = plt.subplots()
    bp = ax.boxplot(AP_data, patch_artist=True)#, boxprops=dict(facecolor=color_list))
    ax.set_xticks(np.arange(len(label_list))+shift)
    ax.set_xticklabels(label_list, rotation=rotation, ha=ha)
    for box, color in zip(bp['boxes'], color_list):
        box.set_facecolor(color)
    ax.set_ylabel('AP', fontsize=label_fontsize)
    ax.set_title(dataset + ' - ' + clf_name +' - '+ target_type, fontsize=title_fontsize, weight='bold')
    ax.yaxis.grid(True)
    if dataset == 'hcp':
        ax.set_ylim(ylim_low,1) # set ylim
        if target_type == 'structconn':
                line1 = ax.axhline(y=mean_AP_corr100, color='darkorange', linestyle='--', label='corr100')
                line2 = ax.axhline(y=mean_AP_corr200, color='fuchsia', linestyle='--', label='corr200')
                line3 = ax.axhline(y=mean_AUC_corr300, color='r', linestyle='--', label='corr300')
                ax.legend(handles=[line1,line2,line3])
        elif target_type == 'male':
            line1 = ax.axhline(y=mean_AP_corr100, color='darkorange', linestyle='--', label='demoinfo+corr100')
            line2 = ax.axhline(y=mean_AP_corr200, color='fuchsia', linestyle='--', label='demoinfo+corr200')
            line3 = ax.axhline(y=mean_AP_corr300, color='r', linestyle='--', label='demoinfo+corr300')
            line4 = ax.axhline(y=mean_AP_demoinfo, color='lightpink', linestyle='--', label='demoinfo')
            ax.legend(handles=[line1,line2,line3,line4])
        else:
            print('Unknown target type')
    elif dataset == 'decnef':
        ax.set_ylim(ylim_low,1) # set ylim
        # add baseline results
        line1 = ax.axhline(y=mean_AP_demoinfo, color='lightpink', linestyle='--', label='demoinfo')
        line2 = ax.axhline(y=mean_AP_corr100, color='darkorange', linestyle='--', label='demoinfo+corr100')
        line3 = ax.axhline(y=mean_AP_corr200, color='fuchsia', linestyle='--', label='demoinfo+corr200')
        line4 = ax.axhline(y=mean_AP_corr300, color='r', linestyle='--', label='demoinfo+corr300')
        ax.legend(handles=[line1,line2,line3,line4])
    else:
        print('Unknown dataset')
    if fig_name is not None:
        plt.savefig(main_dir+'/figures/'+dataset+'_boxplot_AP_'+fig_name+'.png', bbox_inches='tight')    
    else:
        plt.savefig(main_dir+'/figures/'+dataset+'_boxplot_AP_'+clf_name+'_'+target_type+'.png', bbox_inches='tight')  
    '''   
    # F1-score boxplot
    fig, ax = plt.subplots()
    bp = ax.boxplot(F1_data, patch_artist=True)#, boxprops=dict(facecolor=color_list))
    ax.set_xticks(np.arange(len(label_list))+shift)
    ax.set_xticklabels(label_list, rotation=rotation, ha=ha)
    for box, color in zip(bp['boxes'], color_list):
        box.set_facecolor(color)
    ax.set_ylabel('F1', fontsize=label_fontsize)
    ax.set_title(dataset + ' - ' + clf_name +' - '+ target_type, fontsize=title_fontsize, weight='bold')
    ax.yaxis.grid(True)
    if dataset == 'hcp':
        ax.set_ylim(ylim_low,1) # set ylim
        if target_type == 'structconn':
            line1 = ax.axhline(y=mean_F1_corr100, color='darkorange', linestyle='--', label='corr100')
            line2 = ax.axhline(y=mean_F1_corr200, color='fuchsia', linestyle='--', label='corr200')
            line3 = ax.axhline(y=mean_F1_corr300, color='r', linestyle='--', label='corr300')
            ax.legend(handles=[line1,line2,line3])
        elif target_type == 'male':
            line1 = ax.axhline(y=mean_F1_corr100, color='darkorange', linestyle='--', label='demoinfo+corr100')
            line2 = ax.axhline(y=mean_F1_corr200, color='fuchsia', linestyle='--', label='demoinfo+corr200')
            line3 = ax.axhline(y=mean_F1_corr300, color='r', linestyle='--', label='demoinfo+corr300')
            line4 = ax.axhline(y=mean_F1_demoinfo, color='lightpink', linestyle='--', label='demoinfo')
            ax.legend(handles=[line1,line2,line3,line4])
        else:
            print('Unknown target type')
    elif dataset == 'decnef':
        ax.set_ylim(ylim_low,1) # set ylim
        # add baseline results
        line1 = ax.axhline(y=mean_F1_demoinfo, color='lightpink', linestyle='--', label='demoinfo')
        line2 = ax.axhline(y=mean_F1_corr100, color='darkorange', linestyle='--', label='demoinfo+corr100')
        line3 = ax.axhline(y=mean_F1_corr200, color='fuchsia', linestyle='--', label='demoinfo+corr200')
        line4 = ax.axhline(y=mean_F1_corr300, color='r', linestyle='--', label='demoinfo+corr300')
        ax.legend(handles=[line1,line2,line3,line4])
    else:
        print('Unknown dataset')
    if fig_name is not None:
        plt.savefig(main_dir+'/figures/'+dataset+'_boxplot_F1_'+fig_name+'.png', bbox_inches='tight')    
    else:
        plt.savefig(main_dir+'/figures/'+dataset+'_boxplot_F1_'+clf_name+'_'+target_type+'.png', bbox_inches='tight') 
    plt.close('all')
    
    # compute mean over random initializations and max over feature models
    mean_acc = [np.mean(acc_list) for acc_list in acc_data]
    #mean_AUC = [np.mean(AUC_list) for AUC_list in AUC_data]
    #mean_AP = [np.mean(AP_list) for AP_list in AP_data]
    mean_F1 = [np.mean(F1_list) for F1_list in F1_data]
    max_mean_acc = np.max(mean_acc)
    #max_mean_AUC = np.max(mean_AUC)
    #max_mean_AP = np.max(mean_AP)
    max_mean_F1 = np.max(mean_F1)
    best_idx = np.argmax(mean_F1) # using on refit metric F1
    #return max_mean_acc, max_mean_AUC, max_mean_AP, max_mean_F1, best_params_data[best_idx]
    return max_mean_acc, max_mean_F1, best_params_data[best_idx]

# %%
def boxplot_synpar(df, par, model_type, noc_ini, Nc_type, S1, main_dir='/work3/s174162/speciale', label_fontsize=label_fontsize, subtitle_fontsize=subtitle_fontsize, title_fontsize=title_fontsize):
    # define respective dataframe
    df_dataset = df.loc[df.dataset=='synthetic'].sort_values(by=['N','K','S1','S2','Nc_type', 'eta_similarity', 'model_type','splitmerge','noc'], ascending=True).reset_index()
    unique_models = df_dataset[['model_type', 'K', 'S1', 'Nc_type', 'eta_similarity', 'noc']].drop_duplicates()
    unique_models_select = unique_models[(unique_models.model_type==model_type) & 
                                            (unique_models.noc==noc_ini) & 
                                            (unique_models.Nc_type==Nc_type) &
                                            (unique_models.S1==S1)]

    # Compute boxplot data
    eta_similarity_list = ['comp_diff', 'part_diff', 'same']
    bp_list = []
    for eta_similarity in eta_similarity_list:
        unique_models_select_eta_similarity = unique_models_select.loc[(unique_models_select['eta_similarity']==eta_similarity)]
        MAPpar_data = []
        for _, row in unique_models_select_eta_similarity.iterrows():
            model_type = row['model_type']
            noc = row['noc'] # noc_ini
            K = row['K']
            S1 = row['S1']
            Nc_type = row['Nc_type']
            eta_similarity = row['eta_similarity']

            # specify df specific model_exp
            model_df = df_dataset.loc[(df_dataset['model_type'] == model_type) & (df_dataset['K'] == K) & (df_dataset['S1'] == S1) & 
                                        (df_dataset['Nc_type'] == Nc_type) & (df_dataset['eta_similarity'] == eta_similarity) & (df_dataset['noc'] == noc)]

            for _, model_row in model_df.iterrows():
                folders = model_row['exp_name_list']
                paths = [os.path.join(main_dir, 'results', 'synthetic', folder) for folder in folders]
            if par == 'nmi':
                MAPpar_list = get_syn_nmi(paths, K, Nc_type, eta_similarity) #output is nmi_list
            else:
                MAPpar_list, _, _, _, _ = get_stats(paths, par)  
            MAPpar_data.append(MAPpar_list)
        bp_list.append(MAPpar_data)

    # PLOT
    fig, ax = plt.subplots()
    bp0 = ax.boxplot(bp_list[0], positions=[1,5,9], widths=0.35, 
                    patch_artist=True, boxprops=dict(facecolor="C0"))
    bp1 = ax.boxplot(bp_list[1], positions=[2,6,10], widths=0.35, 
                    patch_artist=True, boxprops=dict(facecolor="C2"))
    bp2 = ax.boxplot(bp_list[2], positions=[3,7,11], widths=0.35, 
                    patch_artist=True, boxprops=dict(facecolor="C4"))

    ax.legend([bp0["boxes"][0], bp1["boxes"][0], bp2["boxes"][0]], eta_similarity_list)#, loc='upper left')
    ax.set_xticklabels(['K=2','','','','K=5','','','','K=10'])
    #ax.set_ylim(min_noc, 10+1)
    if par == 'noc':
        ax.set_ylabel('MAP noc', fontsize=label_fontsize)
        ax.plot([1,5,9],[2,5,10], color='C0', marker='x', linestyle='--', linewidth=2, alpha=0.5)
        ax.plot([2,6,10],[1,3,5], color='C2', marker='x', linestyle='--', linewidth=2, alpha=0.5)
        ax.plot([3,7,11],[1,1,1], color='C4', marker='x', linestyle='--', linewidth=2, alpha=0.5)
        if model_type == 'nonparametric':
            ax.set_ylim(0, 10+0.1)
        elif model_type == 'parametric':
            ax.set_ylim(0, noc_ini+0.1)
    elif par == 'logP_A':
        ax.set_ylabel('MAP logP_A', fontsize=label_fontsize)
    elif par == 'nmi':
        ax.set_ylabel('NMI', fontsize=label_fontsize)
        ax.set_ylim(0, 1+0.1)
    else:
        print('unknown par')
    ax.yaxis.grid(True)
    ax.set_title('Exp: '+model_type + ', initial noc='+str(noc)+', '+Nc_type+', S1='+str(int(S1)), fontsize=12, weight='bold')
    #ax.set_xlim(0,14)
    #plt.show()
    plt.savefig(main_dir+'/figures/syn_bp_'+par+'_'+model_type+str(noc)+Nc_type+str(int(S1))+'.png', bbox_inches='tight')    
            