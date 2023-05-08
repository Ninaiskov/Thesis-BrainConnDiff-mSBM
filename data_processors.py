### Functions for generating or constructing data
## preamble 
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "4"  # set number of threads to 4

############################################################ Synthetic data ############################################################
def generate_syndata(Nc, K, S1, S2, disp_data = False):
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
    # eta2 = np.random.rand(K-1, K-1)

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
