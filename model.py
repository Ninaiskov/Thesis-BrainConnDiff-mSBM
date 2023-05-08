#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:58:41 2023

@author: Nina
"""
import os
import numpy as np
from scipy.sparse import csc_matrix #csr_matrix
from scipy.special import gammaln, gamma
import time
import scipy.io
from data_processors import generate_syndata, compute_A

os.environ["OMP_NUM_THREADS"] = "2"  # set number of threads to 4

class MultinomialSBM(object): # changed name from IRMUnipartiteMultinomial to MultinomialSBM
    # Non-parametric IRM of uni-partite undirected graphs based on collapsed Gibbs sampling
    
    # Usage model = MultinomialSBM(config)
    
    # Input: config (see description of each parameter in main.py)
    
    # Predefied in code:
    # alpha         Hyperparameter for the Dirichlet prior on eta (default: np.log(N))
    # eta0          Initial value of 1 x S vector of clustering probabilities (default: np.ones(S)), where S is total number of graphs (subjects)
    # n_graphs      Number of graphs used in threshold annealing (default: 4)
    
    # Output: model_sample.npy file containing:
    # iter          List of iterations
    # Z             Estimated clustering assignment matrix
    # noc           Estimated number of clusters (number of rows in Z)
    # logP_A        Log likelihood of P(A|Z)
    # logP_Z        Log prior probability of p(Z)
    # logP          Log posterior probability of p(Z|A)
    # eta           Estimated of noc x noc x S clustering probabilities (where noc is estimated number of clusters)
    # alpha         Estimated value of alpha (hyperparameter for the Dirichlet prior on eta)
    # eta0          Estimated value of 1 x S vector of clustering probabilities
    # MAP           MAP estimates for all 9 parameters described above
    
    # Original Matlab version of code is written by Morten Mørup (name: IRMUnipartiteMultinomial.m)
    # Python version of code and modifications is written by Nina Iskov
    
    def __init__(self, config):
        
        # Data configuration.
        self.dataset = config.dataset
            # Synthetic data configuration.
        self.Nc = config.Nc
        self.K = config.K
        self.S1 = config.S1
        self.S2 = config.S2
            # MRI data configurations (fMRI and/or dMRI)
        self.atlas_name = config.atlas_name
        self.n_rois = config.n_rois
        self.threshold_annealing = config.threshold_annealing
        self.n_graphs = 4 # total number of graphs used in threshold annealing
        
        # Model configuration. 
        self.model_type = config.model_type
        self.noc = config.noc
        self.splitmerge = config.splitmerge
        
        # Training configurations
        self.maxiter = config.maxiter_gibbs
        self.maxiter_eta0 = config.maxiter_eta0 
        self.maxiter_alpha = config.maxiter_alpha
        self.maxiter_splitmerge = config.maxiter_splitmerge 
        self.matlab_compare = config.matlab_compare
        #self.unit_test = config.unit_test
        #self.abstol = 1e-9 # absolute tolerance used for unit tests
    
        # Miscellaneous.
        self.main_dir = config.main_dir
        self.disp = config.disp
        self.sample_step = config.sample_step
        
        self.it = 0
        self.graph_count = 0 # count of graphs used in threshold annealing
        self.sample = {'iter': [], 'Z': [], 'noc': [], 'logP_A': [], 'logP_Z': [], 'logP': [], 'eta': [], 'alpha': [], 'eta0': []}
        
        # Load data (generate N x N x S adjacency matrix, A)
        self.load_data()
        
        # Initialize variables
        self.N, _, self.S = self.A.shape # shape of adjacency matrix: N = number of nodes (size), S = number of graphs/subjects
        self.alpha = np.log(self.N) # chosen heuristically (add to input later if needed)
        self.eta0 = np.ones(self.S) # default (add to input later if needed)
        self.eta = np.zeros((self.noc, self.noc, self.S))
        
        # Make sure A is unipartite and undirected
        for s in range(self.S):
            self.A[:,:,s] = np.triu(self.A[:,:,s],1)
            self.A[:,:,s] = self.A[:,:,s]+self.A[:,:,s].T # The adjacency matrix of an undirected graph is always symmetric (note setting diagonal to zero using triu)
        #self.A = np.stack([np.triu(self.A[:,:,s],1) for s in range(self.S)],axis=2) # chatGPT suggestion
        #self.A = np.stack([self.A[:,:,s]+self.A[:,:,s].T for s in range(self.S)], axis=2) # chaptGPT suggestion
        
        # Initialize Z (random clustering assignment matrix)
        if self.matlab_compare:
            ind = scipy.io.loadmat('matlab_randvar/ind.mat')['ind'].ravel() -1 # TESTING
        else:
            ind = np.random.choice(self.noc, self.N)
        self.Z = csc_matrix((np.ones(self.N), (ind, np.arange(self.N))), shape=(self.noc, self.N)).toarray()
        self.Z = self.Z[self.Z.sum(axis=1) > 0,:]
        self.Z = self.Z[np.sum(self.Z,1) > 0] # remove empty clusters
        self.sumZ = [] #np.sum(self.Z, axis=1) # no. nodes in each cluster
       
    def train(self):
        
        # Set algorithm variables
        logP = -np.inf
        logP_best = -np.inf

        if self.disp: # Display algorithm
            print('Uni-partite clustering based on the IRM model for Multinomial graphs')
            dheader = '{:<12} | {:<12} | {:<12} | {:<12} | {:<12}'.format('Iteration', 'logP', 'dlogP/|logP|', 'noc', 'time')
            dline = '-------------+--------------+--------------+--------------+--------------'
            print(dline)
            print(dheader)
            print(dline)

############################################################### Main loop ###############################################################
    
        while self.it < self.maxiter:
            self.it += 1
            start_time = time.time()
            logP_old = logP
            
            # Loading data
            if self.threshold_annealing:
                if (self.it-1) % int(self.maxiter/self.n_graphs) == 0 and self.it > 1:
                    self.load_data()
                    # Make sure A is unipartite and undirected (not really necessary since the graphs are constructed like this, but just to be sure)
                    #for s in range(self.S):
                        #self.A[:,:,s] = np.triu(self.A[:,:,s],1)
                        #self.A[:,:,s] = self.A[:,:,s]+self.A[:,:,s].T # The adjacency matrix of an undirected graph is always symmetric (note setting diagonal to zero using triu)
        
            # Gibbs sampling of Z
            if self.matlab_compare:
                JJ_list = scipy.io.loadmat('matlab_randvar/JJ.mat')['JJ_list'].ravel() 
                JJ = JJ_list[(self.it-1)*self.N:(self.it-1)*self.N+self.N] -1 # note: -1 to adjust to python indices
            else:
                JJ = np.random.permutation(self.N) # random permutation of the nodes
            
            self.Z, self.logP_A, self.logP_Z, _, _ = self.gibbs_sample_Z(self.Z, JJ, comp=[], Force=[]) # input: Z, A, eta0, alpha, N. Output: Z, logP_A, logP_Z
            if self.splitmerge:
                for _ in range(self.maxiter_splitmerge):
                    self.Z, self.logP_A, self.logP_Z, = self.splitmerge_sample_Z(self.Z, self.logP_A, self.logP_Z)
        
            self.sumZ = np.sum(self.Z, axis=1) # no. nodes in each cluster
            ind = np.argsort(-self.sumZ) # sort clusters by size (descending)
            self.Z = self.Z[ind,:] # sort partion matrix by cluster size
            self.noc = self.Z.shape[0]
            
            # Sample alpha
            self.sample_alpha() # input: Z, alpha. Output: logP_Z, alpha
            
            # Sample eta0
            self.sample_eta0() # input: A, Z, eta0. Output: logP_A, eta0
            
            # Calculate eta (we compute expected value of posterior of eta, just in case we want to say something about it later)
            self.calculate_eta() # input: Z, eta0. Output: eta
            
            # Evaluate result
            logP = self.logP_A + self.logP_Z # posterior probability (log likelihood + log prior), logP_Z|A
            dlogP = logP - logP_old
            elapsed_time = (time.time() - start_time) # elapsed time
            
            # Display iteration
            if self.it % 1 == 0 and self.disp:
                print(f"{self.it:12.0f} | {logP:12.4e} | {dlogP/np.abs(logP):12.4e} | {self.noc:12.0f} | {elapsed_time:12.4f}")

            # Store sample
            if self.it % self.sample_step == 0:
                self.sample['iter'].append(self.it) # maybe don't save
                #self.sample['Z'].append(self.Z) # maybe dont' save
                self.sample['noc'].append(self.noc)
                self.sample['logP_A'].append(self.logP_A) # logP_A|Z (log likelihood)
                self.sample['logP_Z'].append(self.logP_Z) # logP_Z (log prior)
                self.sample['logP'].append(logP) # logP_Z|A (log likelihood + log prior)
                #self.sample['eta'].append(self.eta) # maybe dont' save
                #self.sample['alpha'].append(self.alpha) # maybe dont' save
                #self.sample['eta0'].append(self.eta0) # maybe dont' save
            if logP > logP_best:
                self.sample['MAP'] = {'iter': self.it, 
                                      'Z': self.Z, 
                                      'noc': self.noc, 
                                      'logP_A': self.logP_A, 
                                      'logP_Z': self.logP_Z, 
                                      'logP': logP, 
                                      'eta': self.eta, 
                                      'alpha': self.alpha, 
                                      'eta0': self.eta0}
                logP_best = logP
         
        # Display final iteration
        print('Result of final iteration')
        print('%12s | %12s | %12s | %12s | %12s ' % ('iter', 'logP', 'dlogP/|logP|', 'noc', 'time'))
        print('%12.0f | %12.4e | %12.4e | %12.0f | %12.4f ' % (self.it, logP, dlogP/np.abs(logP), self.noc, elapsed_time))


############################################################### Gibbs sampler ###############################################################

    def gibbs_sample_Z(self, Z, JJ, comp, Force):
        logQ_trans = 0 # log of transition probability of Z (used for split-merge MH sampler step)
        if self.model_type == 'parametric':
            Force = []
            comp = []
        if self.matlab_compare:
            randval_list = scipy.io.loadmat('matlab_randvar/rand_val.mat')['randval_list'].ravel()
        
        const = self.multinomialln(self.eta0) # likelihood constant, log B(eta0)
        self.sumZ = np.sum(Z, axis=1) # number of nodes in each cluster
        self.noc = Z.shape[0] # number of clusters
        
        n_link = np.zeros((self.noc, self.noc, self.S)) # number of links between clusters
        for s in range(self.S): # for each subject
            n_link[:, :, s] = Z @ self.A[:, :, s] @ Z.T # clever way to extract all the relevant links for each pair of clusters
            n_link[:, :, s] = n_link[:, :, s] - 0.5 * np.diag(np.diag(n_link[:, :, s])) + self.eta0[s] # subtracting half of the diagonal from the diagonal (since we actually count the diognal twice in the line above) and adding the probability of linking between clusters  
        #n_link = np.stack([Z @ self.A[:, :, s] @ Z.T for s in range(self.S)],axis=2) # chatGPT suggestion
        #n_link = np.stack([n_link[:, :, s] - 0.5 * np.diag(np.diag(n_link[:, :, s])) + self.eta0[s] for s in range(self.S)], axis=2) # chaptGPT suggestion

        mult_eval = self.multinomialln(n_link) # compute (multinomial) log likelihood of number of links between clusters, log Beta(nlink+eta0)
        for i in JJ: # for each node (in random permutated order)
            # Remove effect of node i in partion, i.e. Z[:,i]
            self.sumZ -= Z[:, i]
            # Compute link contribution of node i to log likelihood
            ZAi = np.zeros((self.noc, 1, self.S))
            for s in range(self.S):
                ZAi[:, 0, s] = Z @ self.A[:, i, s] # sum of links between node i and other nodes in respective cluster/block for subject s (# partition*adjacency matrix (links) = non-binary matrix) 
            #ZAi = np.stack([Z @ self.A[:, i, s] for s in range(self.S)], axis=1) # chatGPT suggestion
            #ZAi = ZAi[:, np.newaxis, :] # chatGPT suggestion
            
            d = np.nonzero(Z[:, i])[0] # find non-empty cluster for node i (i.e. find which cluster node i is assigned to)
            if len(d) > 0: # if d is not an empty list (there exists non-empty cluster exist for node i (not really neccessary to make the if statement, since the node will always be assigned to a cluster)
                n_link[:, d, :] -= ZAi # removing link contribution of node i: (number of links between clusters and non-empty cluster d) minus (sum of links between node i and other nodes in respective cluster/block for subject s)
                n_link[d, :, :] = np.transpose(n_link[:, d, :], (1,0,2)) # making sure n_link is symmetric
                #n_link[d, :, :] = n_link[:, d, :].transpose((1,0,2)) # chatGPT suggestion
                Z[:, i] = 0 # remove cluster assignment for node i (i.e. remove it from cluster d)
            #else:
                #print('d is an empty list')

            ######### NOT in split merge sampler step (comp is empty) #########
            if len(comp) == 0: # if no components are given (i.e. if we are not in the split-merge MH sampler step)
                if self.sumZ[d] == 0: #if sum of nodes in cluster d is 0 then it means that node i was the only node in cluster d ("singleton cluster") and since we removed node i's contibution to sumZ, the cluster is now empty
                    #print('Removing singleton cluster')
                    v = np.arange(self.noc) 
                    v = v[v != d] # removing singleton cluster
                    d = [] 
                    self.noc -= 1 # reducing number of clusters by 1 
                    P = csc_matrix((np.ones(self.noc),(np.arange(self.noc),v)), shape=(self.noc, self.noc+1)).toarray() 
                    Z = P @ Z
                    ZAi = ZAi[v, :, :]
                    self.sumZ = self.sumZ[v]
                    n_link = n_link[v][:, v, :]
                    mult_eval = mult_eval[v][:, v]
                #else:
                    #print('No singleton cluster')
                # Calculate probability for existing communities as well as proposal cluster
                mult_eval[:,d] = self.multinomialln(n_link[:,d,:]) # updating likelihood given that node i is NOT in cluster d - i.e. compute multinomial likelihood for number of links between cluster d and other clusters for each subject  
                mult_eval[d,:] = mult_eval[:,d].T
                sum_mult_eval_dnoi = np.sum(mult_eval, axis=0)
                if self.model_type == 'nonparametric':
                    mult_eval_di = self.multinomialln(np.concatenate((n_link + ZAi, ZAi + self.eta0), axis=1)) # (note we use broadcasting here to add the contribution of node i to each cluster)
                    logQ = np.append(np.sum(mult_eval_di[:, :self.noc], axis=0), np.sum(mult_eval_di[:, self.noc], axis=0) - self.noc * const).T - np.append(sum_mult_eval_dnoi, 0) # note that prior is not included here since its just constant
                else:
                    mult_eval_di = self.multinomialln(n_link + ZAi)
                    #logQ = np.sum(mult_eval_di, axis=0) - sum_mult_eval_dnoi + gammaln(self.sumZ + 1 + self.alpha) - gammaln(self.sumZ + self.alpha)
                    logQ = np.sum(mult_eval_di, axis=0) - sum_mult_eval_dnoi # notice that the conditional prior is not included here, but instead implemented as weight 
                
                # Sample from posterior conditional
                QQ = np.exp(logQ - np.max(logQ)) # normalize to avoid numerical problems
                if self.model_type == 'nonparametric':
                    weight = np.append(self.sumZ, self.alpha) # alpha is the weight for the CRP prior
                else:
                    weight = self.sumZ + self.alpha
                    #weight = gamma(self.sumZ + 1 + self.alpha)/gamma(self.sumZ + self.alpha)
                    
                #if self.unit_test:
                #    self.unit_test_gibbs(logQ, weight, i)
                
                QQ = weight * QQ # compute true (weighted) pdf (weighted by the conditional prior)
                if self.matlab_compare:
                    randval = randval_list[i]
                else:
                    randval = np.random.rand()

                ind = np.argmax(randval < np.cumsum(QQ/np.sum(QQ)),axis=0) # generate random sample using cdf (inverse transform sampling)
                if ind >= self.noc: # this part is only the case for CRP prior (if self.model_type == 'nonparametric')
                        #print('ind>=noc') # actually it will just be ind == noc
                        # modifying shapes to include extra cluster
                        Z = np.concatenate((Z, np.zeros((1,self.N))), axis=0)
                        self.noc = Z.shape[0]
                        n_link = np.concatenate((n_link, np.zeros((1, self.noc-1, self.S))), axis = 0)
                        n_link = np.concatenate((n_link, np.zeros((self.noc, 1, self.S))), axis = 1)
                        mult_eval = np.concatenate((mult_eval, np.zeros((1, self.noc-1))), axis=0)
                        mult_eval = np.concatenate((mult_eval, np.zeros((self.noc, 1))), axis=1)
                        ZAi = np.concatenate((ZAi, np.zeros((1, 1, self.S))), axis=0)
                        Z[ind, i] = 1 # % updating partition: assigning node i to cluster with given index ind)
                        self.sumZ = np.append(self.sumZ, 0) # add zero nodes to the last cluster
                        #self.sumZ[self.noc, 0] = 0
                        for s in range(self.S):
                            n_link[:, ind, s] = self.eta0[s]
                            n_link[ind, :, s] = self.eta0[s]
                        #n_link[:, ind, :] = self.eta0.reshape(1, 1, -1)# chatGPT suggestion
                        #n_link[ind, :, :] = self.eta0.reshape(1, 1, -1)# chatGPT suggestion
                        mult_eval[:, ind] = 0 
                        mult_eval[ind, :] = 0
                        mult_eval_di = np.append(mult_eval_di[:, ind], const).T
                        ZAi[ind, 0, :] = 0
                else: # ind < self.noc
                    Z[ind, i] = 1 #  updating partition: assigning node i to cluster with given index ind)
                    mult_eval_di = mult_eval_di[:, ind]
                    
            else: ######### In split merge sampler step (comp is NOT empty meaning that the Gibbs sampling will be restricted to the given clusters in comp)!!! #########
                # Calculate probability for existing communities as well as proposal cluster (only for non-parametric model) - only changes for lines for computing sum_mult_eval_dnoi andd mult_eval_di
                mult_eval[:,d] = self.multinomialln(n_link[:,d,:]) # updating likelihood given that node i is NOT in cluster d - i.e. compute multinomial likelihood for number of links between cluster d and other clusters for each subject                 
                mult_eval[d,:] = mult_eval[:,d].T
                sum_mult_eval_dnoi = np.sum(mult_eval[:, comp], axis=0)
                mult_eval_di = self.multinomialln(n_link[:,comp,:] + ZAi) # (note we use broadcasting here to add the contribution of node i to each cluster)
                logQ = sum_mult_eval_dnoi + np.sum(mult_eval_di, axis=0)
                # Sample from posterior conditional
                QQ = np.exp(logQ - np.max(logQ)) # normalize to avoid numerical problems
                weight = self.sumZ[comp]
                
                #if self.unit_test:
                #    self.unit_test_splitmerge_Z(Z, comp, i, logQ, weight)
                    
                QQ = weight * QQ # compute true (weighted) pdf
                if len(Force) == 0:
                    ind = np.argmax(np.random.rand() < np.cumsum(QQ/np.sum(QQ)), axis=0) # generate random sample using cdf (inverse transform sampling)
                    #ind = np.argmax(0.7 < np.cumsum(QQ/np.sum(QQ)),axis=0) # TESTING # issue: invalid value encountered in divide
                else:
                    ind = int(Force[i])
                q_tmp = logQ - np.max(logQ) + np.log(weight)
                q_tmp -= np.log(np.sum(np.exp(q_tmp)))
                logQ_trans += q_tmp[ind]
                Z[comp[ind], i] = 1
                mult_eval_di = mult_eval_di[:, ind]
                mult_eval[d,:] = mult_eval[:,d].T
                ind = comp[ind]
                
            # Add contribution of new node i partition assignment
            self.sumZ += Z[:, i] # updating sum of nodes in each cluster, i.e. adding new node assignment (node i) to respective cluster
            n_link[:, ind, :] += ZAi[:,0,:]#.squeeze() # update af number of links
            n_link[ind, :, :] = n_link[:,ind,:].copy()#.T#np.transpose(n_link[:, ind, :], (1,0))
            mult_eval[:, ind] = mult_eval_di
            mult_eval[ind, :] = mult_eval_di.T
            
            # Remove empty clusters - necessary?
            if np.any(self.sumZ == 0): # if any empty clusters exists
                d = np.nonzero(self.sumZ == 0)[0] # find empty cluster
                if len(comp) > 0:
                    ind_d = np.nonzero(d < comp)[0] # find index of empty cluster in comp
                    comp[ind_d] = comp[ind_d] - 1 # update comp to reflect that cluster d is removed
                v = np.arange(self.noc)
                v = v[v != d]
                self.noc -= len(d)
                P = csc_matrix((np.ones(self.noc),(np.arange(self.noc), v)), shape=(self.noc, self.noc+1)).toarray()
                Z = P @ Z
                self.sumZ = self.sumZ[v]
                n_link = n_link[v][:,v,:]
                mult_eval = mult_eval[v][:,v]                
            
        # Calculate likelihood for sampled solution (after seeing all nodes and subjects)
        logP_A = np.sum(np.triu(mult_eval)) - self.noc * (self.noc + 1) / 2 * const
        if self.model_type == 'nonparametric':
            constZ = np.sum(gammaln(self.sumZ))
            logP_Z = self.noc * np.log(self.alpha) + constZ - gammaln(self.N + self.alpha) + gammaln(self.alpha)
        else:
            logP_Z = gammaln(self.noc * self.alpha) - gammaln(self.noc * self.alpha + self.N) - self.noc * gammaln(self.alpha) + np.sum(gammaln(self.sumZ + self.alpha))
        
        return Z, logP_A, logP_Z, logQ_trans, comp
    

############################################################### Metropolis-Hastings samplers ###############################################################
# Split-merge (version of MH) sampler for Z
# MH sampler for alpha
# MH sampler for eta0

# need to update Gibbs sampler to handle Force, comp, logQ_trans and take different Z and JJ as input
    def splitmerge_sample_Z(self, Z, logP_A, logP_Z):
        self.noc, self.N = Z.shape # number of clusters and number of nodes
        # choose two random nodes
        ind1 = int(np.ceil(self.N * np.random.rand()))-1
        ind2 = int(np.ceil((self.N-1) * np.random.rand()))-1
        #ind1 = int(np.ceil(self.N * 0.5))-1 # TESTING 0.4
        #ind2 = int(np.ceil((self.N-1) * 0.6))-1 # TESTING 0.2
        
        if ind1 <= ind2:
            ind2 += 1
        # find cluster for nodes: ind1 and ind2
        clust1 = np.nonzero(Z[:, ind1])[0][0] 
        clust2 = np.nonzero(Z[:, ind2])[0][0]
        
        # if the two nodes are in the same cluster, try to split
        if clust1 == clust2: # split
            setZ = np.nonzero(np.sum(Z[[clust1, clust2], :], axis=0)) # find nodes that are assigned to either clust1 or clust2
            setZ = np.setdiff1d(setZ, [ind1, ind2]) # remove nodes ind1 and ind2 from list of nodes
            n_setZ = len(setZ) # total number of nodes assigned to either clust1 or clust2 (not including ind1 and ind2)
            Z_t = Z.copy()
            # reassign the first node to the original cluster and the second node to a new cluster
            Z_t[clust1, :] = 0
            comp = [clust1, self.noc]
            Z_t[comp[0], ind1] = 1
            Z_t = np.concatenate((Z_t, np.zeros((1, self.N))), axis=0)
            Z_t[comp[1], ind2] = 1
            
            # Reassign by restricted Gibbs sampling
            JJ = setZ[np.random.permutation(n_setZ)]
            if n_setZ > 0:
                for _ in range(3): # "3 restricted gibbs sampling sweeps"
                    Z_t, logP_A_t, logP_Z_t, logQ_trans, comp = self.gibbs_sample_Z(Z_t, JJ, comp, Force=[]) # input: Z, A, eta0, alpha, N. Output: Z, logP_A, logP_Z
                    #Z_t, logP_A_t, logP_Z_t, logQ_trans, comp = self.gibbs_sample_Z(Z=Z_t, JJ=setZ, comp=comp, Force=[]) # TESTING
            else:
                logQ_trans = 0
                logP_A_t, logP_Z_t = self.evalProbs(Z_t, self.eta0, self.alpha)
                
            # Calculate Metropolis-Hastings ratio
            a_split = np.random.rand() < np.exp(logP_A_t + logP_Z_t - logP_A - logP_Z - logQ_trans) # acceptance probability for splitting cluster
            #a_split = 0.2 < np.exp(logP_A_t + logP_Z_t - logP_A - logP_Z - logQ_trans) # TESTING
            
            if a_split:
                print('Splitting cluster', str(clust1))
                logP_A = logP_A_t
                logP_Z = logP_Z_t
                Z = Z_t.copy()
        else: # merge
            Z_t = Z.copy()
            Z_t[clust1, :] = Z_t[clust1, :] + Z_t[clust2, :] # merging clusters by adding node contribution of clust2 to clust1
            setZ = np.nonzero(Z_t[clust1, :]) # find nodes assigned to clust1
            Z_t = np.delete(Z_t, clust2, axis=0) # removing clust2
            if clust2 < clust1:
                clust1_t = clust1-1 # correcting cluster index since shifted by removing clust2
            else:
                clust1_t = clust1 # if clust2 > clust1, then clust1 index is not shifted
            noc_t = self.noc-1
            
            # calculate likelihood of merged cluster
            logP_A_t, logP_Z_t = self.evalProbs(Z_t, self.eta0, self.alpha)
            
            # split merged cluster and calculate transition probabilities
            setZ = np.setdiff1d(setZ, [ind1, ind2])
            n_setZ = len(setZ)
            Z_tt = Z_t.copy()
            Z_tt[clust1_t, :] = 0 # dimension of Z_tt should be (noc_t, N) <-- check this
            comp = [clust1_t, noc_t]
            Z_tt[comp[0], ind1] = 1
            Z_tt = np.concatenate((Z_tt, np.zeros((1, self.N))), axis=0)
            Z_tt[comp[1], ind2] = 1
            
            # Reassign by restricted Gibbs sampling
            JJ = setZ[np.random.permutation(n_setZ)]
            if n_setZ > 0:
                for _ in range(2):
                    Z_tt, _, _, _, comp = self.gibbs_sample_Z(Z_tt, JJ, comp, Force=[])
                    #Z_tt, _, _, _, comp = self.gibbs_sample_Z(Z_tt, setZ, comp, Force=[]) # TESTING
                Force = np.array([0, 1]) @ Z[[clust1, clust2], :] # in matlab [1,2]
                JJ = setZ[np.random.permutation(n_setZ)]
                _, _, _, logQ_trans, _ = self.gibbs_sample_Z(Z_tt, JJ, comp, Force)
                #_, _, _, logQ_trans, _ = self.gibbs_sample_Z(Z_tt, setZ, comp, Force) # TESTING
            else:
                logQ_trans = 0
            
            # Calculate Metropolis-Hastings ratio
            a_merge = np.random.rand() < np.exp(logP_A_t + logP_Z_t - logP_A - logP_Z + logQ_trans) # acceptance probability for mergin clusters
            if a_merge:
                print('Merging clusters', str(clust1), 'and', str(clust2))
                logP_A = logP_A_t.copy()
                logP_Z = logP_Z_t.copy()
                Z = Z_t.copy()
        
        return Z, logP_A, logP_Z


    def sample_alpha(self):
        # sample hyperparameter: "concentration parameter" / "rate of generating new clusters" used in CRP dist., imposes improper uniform prior, Metropolis Hastings
        if self.matlab_compare:
            randnalpha_list = scipy.io.loadmat('matlab_randvar/randnalpha.mat')['randnalpha_list'].ravel() # TESTING
            randalpha_list = scipy.io.loadmat('matlab_randvar/randalpha.mat')['randalpha_list'].ravel() # TESTING
        
        if self.model_type == 'nonparametric':
            constZ = np.sum(gammaln(self.sumZ))
            #self.logP_Z = self.noc * np.log(self.alpha) + const - gammaln(self.N + self.alpha) + gammaln(self.alpha)  # CRP - same as self.logP_Z
        
        accept = 0
        for i in range(self.maxiter_alpha):
            if self.matlab_compare:
                randnalpha = randnalpha_list[i]
            else: 
                randnalpha = np.random.randn()
            alpha_new = np.exp(np.log(self.alpha) + 0.1 * randnalpha)  # symmetric proposal distribution in log-domain (use change of variable in acceptance rate alpha_new/alpha)
            if self.model_type == 'nonparametric':
                logP_Z_new = self.noc * np.log(alpha_new) + constZ - gammaln(self.N + alpha_new) + gammaln(alpha_new)
            else:
                logP_Z_new = gammaln(self.noc * alpha_new) - gammaln(self.noc * alpha_new + self.N) - self.noc * gammaln(alpha_new) + np.sum(gammaln(self.sumZ + alpha_new))

            #if self.unit_test:
            #    self.unit_test_MH_alpha(alpha_new=alpha_new, logP_Z_new=logP_Z_new, logP_Z=self.logP_Z) # inputs: eta0_new=None, logP_A_new=None, logP_A=None, alpha_new, logP_Z_new, logP_Z
            
            if self.matlab_compare:
                randalpha = randalpha_list[i]
            else: 
                randalpha = np.random.rand()
            if randalpha < alpha_new / self.alpha * np.exp(logP_Z_new - self.logP_Z):  # if u_k < acceptance probability A (but where is A defined as min(1,r_k))?
                self.alpha = alpha_new
                self.logP_Z = logP_Z_new
                accept += 1

        # print('accepted ' + str(accept) + ' out of ' + str(max_iter) + ' samples for alpha')
        #return logZ, alpha


    def sample_eta0(self):
        if self.matlab_compare:
            randneta0_list = scipy.io.loadmat('matlab_randvar/randneta0.mat')['randneta0_list'].ravel() # TESTING
            randeta0_list = scipy.io.loadmat('matlab_randvar/randeta0.mat')['randeta0_list'].ravel() # TESTING
        
        n_link_noeta0 = np.zeros((self.noc, self.noc, self.S))
        for s in range(self.S):
            n_link_noeta0[:,:,s] = self.Z @ self.A[:,:,s] @ self.Z.T
            n_link_noeta0[:,:,s] = n_link_noeta0[:,:,s] - 0.5 * np.diag(np.diag(n_link_noeta0[:,:,s]))

        n_link = n_link_noeta0 + self.eta0
        #const = self.multinomialln(self.eta0)
        #self.logP_A = np.triu(self.multinomialln(n_link)).sum() - self.noc*(self.noc+1)/2 * const
            
        accept = 0
        for s in range(self.S):
            for i in range(self.maxiter_eta0):
                if self.matlab_compare:
                    randneta0 = randneta0_list[i] # TESTING
                else:
                    randneta0 = np.random.randn() 
                eta_new = np.exp(np.log(self.eta0[s]) + 0.1 * randneta0)  # symmetric proposal distribution in log-domain (use change of variable in acceptance rate alpha_new/alpha)
                eta0_new = np.copy(self.eta0)
                eta0_new[s] = eta_new
                const_new = self.multinomialln(eta0_new)
                n_link_new = np.copy(n_link)
                n_link_new[:,:,s] = n_link_noeta0[:,:,s] + eta_new
                #logP_A_new = np.triu(self.multinomialln(n_link_new)).sum() - self.noc*(self.noc+1)/2 * const_new
                logP_A_new = np.sum(np.triu(self.multinomialln(n_link_new))) - self.noc*(self.noc+1)/2 * const_new
                
                #if self.unit_test:
                #    self.unit_test_MH_eta0(eta0_new = eta0_new, logP_A_new = logP_A_new, logP_A = self.logP_A) #inputs: eta0_new, logP_A_new, logP_A, alpha_new=None, logP_Z_new=None, logP_Z=None
                
                #if self.matlab_compare:
                #    randeta0 = randeta0_list[i]
                #else:
                randeta0 = np.random.rand()
                if randeta0 < (eta_new/self.eta0[s]) * np.exp(logP_A_new - self.logP_A):
                    self.eta0[s] = eta_new
                    self.logP_A = logP_A_new
                    n_link = n_link_new
                    accept += 1

        #return logP, eta0


############################################################### Data processing functions ###############################################################
    def load_data(self):
            if self.dataset == 'synthetic':
                if self.matlab_compare:
                    A_dict = scipy.io.loadmat('matlab_randvar/A.mat')
                    self.A = A_dict['A']
                else:
                    self.A = generate_syndata(self.Nc, self.K, self.S1, self.S2)
            elif self.dataset == 'hcp' or self.dataset == 'decnef':
                folder_name = self.atlas_name+str(self.n_rois)
                folder_path = os.path.join(self.main_dir, 'data/'+self.dataset+'/'+folder_name)
                if self.threshold_annealing: # incresing graph density over iterations
                    self.graph_count += 1
                    print('Annealing threshold: graph no. '+str(self.graph_count))
                    filename = 'A'+str(self.graph_count)+'_vals_list.npy'
                    A_vals_list = np.load(os.path.join(folder_path,filename))
                else: # using fixed graph density (most dense graph)
                    print('No threshold annnealing: Using fixed graph density (most dense graph)')
                    A_vals_list = np.load(os.path.join(folder_path,'A'+str(self.n_graphs)+'_vals_list.npy'))
                self.A = compute_A(A_vals_list, self.n_rois)
            else:
                print('Unknown dataset')
            
            
############################################################### Model evaluation functions ###############################################################
    def multinomialln(self, x): # maybe better to call it logbeta func 
        # Multinomial distribution (log probability)
        return np.sum(gammaln(x), axis=-1) - gammaln(np.sum(x, axis=-1))


    def calculate_eta(self):
        ZAZt = np.zeros((self.noc, self.noc, self.S))

        for s in range(self.S):
            ZAZt[:,:,s] = self.Z @ self.A[:,:,s] @ self.Z.T
            ZAZt[:,:,s] = ZAZt[:,:,s] - 0.5 * np.diag(np.diag(ZAZt[:,:,s])) + self.eta0[s]
        
        sum_ZAZt = np.sum(ZAZt, axis=2)
        #self.eta = np.divide(ZAZt, sum_ZAZt[:, :, np.newaxis]) # chatGPT suggestion
        self.eta = ZAZt/sum_ZAZt[:,:,np.newaxis] # check if this written smarter

    def evalProbs(self, Z, eta0, alpha):
        # used to evaluate the likelihood and prior probabilities of the model in unit tests
        
        sumZ = np.sum(Z, axis=1) # number of nodes in each cluster
        noc = Z.shape[0] # number of clusters
        
        n_link = np.zeros((noc, noc, self.S)) # number of links between clusters
        for s in range(self.S): # for each subject
            n_link[:, :, s] = Z @ self.A[:, :, s] @ Z.T # clever way to extract all the relevant links for each pair of clusters
            n_link[:, :, s] = n_link[:, :, s] - 0.5 * np.diag(np.diag(n_link[:, :, s])) + eta0[s] # subtracting half of the diagonal from the diagonal (since we actually count the diognal twice in the line above) and adding the probability of linking between clusters
       
        mult_eval = self.multinomialln(n_link) # compute (multinomial) log likelihood of number of links between clusters 
        const = self.multinomialln(eta0)
        
        logP_A = np.sum(np.triu(mult_eval)) - noc * (noc + 1) / 2 * const # /2 something with counting the diagnoal twice?
        if self.model_type == 'nonparametric':
            constZ = np.sum(gammaln(sumZ))
            logP_Z = noc * np.log(alpha) + constZ - gammaln(self.N + alpha) + gammaln(alpha)
        else:
            logP_Z = gammaln(noc * alpha) - gammaln(noc * alpha + self.N) - noc * gammaln(alpha) + np.sum(gammaln(sumZ + alpha))
        return logP_A, logP_Z


############################################################### Unit tests ###############################################################
    def unit_test_gibbs(self, logQ, weight, i):
        noc_tmp1 = int(np.ceil((self.Z.shape[0]-1) * np.random.rand())) # generate cluster index between 1 and noc
        Z_tmp1 = self.Z.copy()
        Z_tmp1[noc_tmp1, i] = 1
        logP_A_tmp1, logP_Z_tmp1 = self.evalProbs(Z_tmp1, self.eta0, self.alpha)
        if self.model_type == 'nonparametric':
            noc_tmp2 = int(np.ceil(self.Z.shape[0] * np.random.rand())) # generate cluster index between 1 and noc+1
            Z_tmp2 = self.Z.copy()
            if noc_tmp2 >= self.Z.shape[0]:
                Z_tmp2 = np.concatenate((Z_tmp2, np.zeros((1, self.Z.shape[1]))), axis=0)
            Z_tmp2[noc_tmp2, i] = 1
            logP_A_tmp2, logP_Z_tmp2 = self.evalProbs(Z_tmp2, self.eta0, self.alpha)
            a1 = logP_A_tmp1 + logP_Z_tmp1 - (logP_A_tmp2 + logP_Z_tmp2)
            a2 = logQ[noc_tmp1] + np.log(weight[noc_tmp1]) - (logQ[noc_tmp2] + np.log(weight[noc_tmp2]))
        else:
            noc_tmp2 = int(np.ceil((self.Z.shape[0]-1) * np.random.rand())) # generate cluster index between 1 and noc
            Z_tmp2 = self.Z.copy()
            Z_tmp2[noc_tmp2, i] = 1
            logP_A_tmp2, logP_Z_tmp2 = self.evalProbs(Z_tmp2, self.eta0, self.alpha)
            a1 = logP_A_tmp1 + logP_Z_tmp1 - (logP_A_tmp2 + logP_Z_tmp2)
            a2 = logQ[noc_tmp1] - logQ[noc_tmp2]
            
        absdiff = np.abs(a1 - a2)
        if absdiff > self.abstol:
            print('Gibbs unit test failed for node', str(i))
            print('absdiff: ', absdiff)
        #else:
        #    print('Gibbs unit test passed')
    
    def unit_test_MH_eta0(self, eta0_new, logP_A_new, logP_A, alpha_new=None, logP_Z_new=None, logP_Z=None):
        logP_A_tmp, _ = self.evalProbs(self.Z, self.eta0, self.alpha)
        logP_A_tmp_new, _ = self.evalProbs(self.Z, eta0_new, self.alpha)
        a1 = logP_A_tmp_new - logP_A_tmp
        a2 = logP_A_new - logP_A
        absdiff = np.abs(a1 - a2)
        if absdiff > self.abstol:
            print('MH eta0 unit test failed')
            print('absdiff: ', absdiff)
        #else:
        #    print('MH eta0 unit test passed')
            
    def unit_test_MH_alpha(self, alpha_new, logP_Z_new, logP_Z, eta0_new=None, logP_A_new=None, logP_A=None):
        _ , logP_Z_tmp = self.evalProbs(self.Z, self.eta0, self.alpha)
        _ , logP_Z_tmp_new = self.evalProbs(self.Z, self.eta0, alpha_new)
        a1 = logP_Z_tmp_new - logP_Z_tmp
        a2 = logP_Z_new - logP_Z
        absdiff = np.abs(a1 - a2)
        if absdiff > self.abstol:
            print('MH alpha unit test failed')
            print('absdiff: ', absdiff)
        #else:
        #    print('MH alpha unit test passed')
        
    def unit_test_splitmerge_Z(self, Z, comp, i, logQ, weight):
        noc1t = comp[0]
        noc2t = comp[1]
        Z_t = Z.copy()
        Z_t[noc1t, i] = 1
        logP_A1, logP_Z1 = self.evalProbs(Z_t, self.eta0, self.alpha)
        Z_t = Z.copy()
        logP_A2, logP_Z2 = self.evalProbs(Z_t, self.eta0, self.alpha)
        a1 = logP_A1 + logP_Z1 - (logP_A2 + logP_Z2)
        a2 = logQ[0] + np.log(weight[0])-(logQ[1]+np.log[weight[1]])
        absdiff = np.abs(a1-a2)
        if absdiff > self.abstol:
            print('MH splitmerge Z unit test failed')
            print('absdiff: ', absdiff)
        #else:
        #    print('MH splitmerge Z unit test passed')