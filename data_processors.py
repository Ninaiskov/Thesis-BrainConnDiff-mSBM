### Functions for generating or constructing data
## preamble 
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

os.environ["OMP_NUM_THREADS"] = "4"  # set number of threads to 4

############################################################ Synthetic data ############################################################
def generate_syndata(N, K, S1, S2, balance_Nc, eta_similarity, seed=0, disp_data = False):
    ## Inputs
    # N = total number of nodes (corresponds to n_rois)
    # K = Number of clusters;
    # S1 = Number of first type of graph, e.g. healthy
    # S2 = Number of second type of graph, e.g. sick
    # balance_Nc = Boolean (True/False) if size of clusters (no. of nodes in each cluster) should be balanced or not
    # eta_similarity = 'same', 'comp_diff' or 'part_diff' (how similar eta1 and eta2 should be)
    # disp_data = bool for displaying generated data
    
    # Output
    # A = adjacency matrices for all subjects
    
    # STEPS:
    # 1) compute partition (z)
        # balanced or unbalanced
    if balance_Nc:
        Nc = int(N/K)
        z = np.kron(np.eye(K),np.ones((Nc,1)))
    else: 
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
    np.random.seed(seed)
    A = np.empty((N, N, S1+S2))
    A.fill(np.nan)
    for s in range(S1+S2):
        if s <= S1-1:
            At = z @ eta1 @ z.T > np.random.rand(N, N)
            A[:,:,s] = np.triu(At, 1) + np. triu(At, 1).T
        else:
            At = z @ eta2 @ z.T > np.random.rand(N, N)
            A[:,:,s] = np.triu(At, 1) + np. triu(At, 1).T

    if disp_data:
        fig, ax = plt.subplots()
        cmap_binary = ListedColormap(['k', 'w']) 
        im = ax.imshow(z, interpolation='nearest', aspect='auto', cmap=cmap_binary, extent=(0, z.shape[1], 0, z.shape[0]))
        ax.set_title('Partition $z$', weight='bold', fontsize=15)
        ax.set_ylabel('Node', fontsize=12)
        ax.set_xlabel('Cluster', fontsize=12)
        cbar = fig.colorbar(im, ax=ax, shrink=0.7, ticks=[0,1])
        plt.show()

        fig, ax = plt.subplots(1,2)
        plt.subplots_adjust(wspace=0.3, top=1.22)
        im = ax[0].imshow(eta1)
        ax[0].set_title('$\eta_1$')
        ax[0].set_ylabel('Cluster', fontsize=12)
        ax[0].set_xlabel('Cluster', fontsize=12)
        im = ax[1].imshow(eta2)
        ax[1].set_ylabel('Cluster', fontsize=12)
        ax[1].set_xlabel('Cluster', fontsize=12)
        ax[1].set_title('$\eta_2$')
        cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.3)
        fig.suptitle('Cluster interaction matrices',fontsize=15, weight='bold')
        plt.show()

        default_blue = '#1f77b4'
        cmap_binary = ListedColormap(['white', default_blue]) 
        fig, ax = plt.subplots(1,2)
        plt.subplots_adjust(wspace=0.3, top=1.1)
        im = ax[0].spy(A[:,:,0],cmap=cmap_binary)
        ax[0].set_ylabel('Node', fontsize=12)
        ax[0].set_xlabel('Node', fontsize=12)
        ax[0].set_title('Graph type 1', weight='bold')
        im = ax[1].spy(A[:,:,-1],cmap=cmap_binary)
        ax[1].set_ylabel('Node', fontsize=12)
        ax[1].set_xlabel('Node', fontsize=12)
        ax[1].set_title('Graph type 2', weight='bold')
        cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.3)
        fig.suptitle('Adjacency matrix $A$',fontsize=15, weight='bold')
        plt.show()
        
    return A, z, eta1, eta2   
    

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

############################################################ fMRI data ##################################################################

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
        healthy_bool = np.load(os.path.join(my_dir, 'data/decnef/healthy_bool.npy'), allow_pickle=True)
        schizo_bool = np.load(os.path.join(my_dir, 'data/decnef/schizo_bool.npy'), allow_pickle=True)
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
