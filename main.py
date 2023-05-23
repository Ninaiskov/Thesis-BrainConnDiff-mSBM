import os
import argparse
import time 
from datetime import datetime
import numpy as np
from model import MultinomialSBM
import cProfile
import pstats


def main(config):
    # initiate results folder and log.txt file
    if config.dataset == 'synthetic':
        exp_name = config.model_type+'_'+str(config.K)+'_'+str(config.S1)+'_'+str(config.S2)+'_'+str(config.balance_Nc)+'_'+config.eta_similarity+'_'+str(datetime.now())
        #exp_name = config.model_type+'_Nc'+str(config.Nc)+'_K'+str(config.K)+'_S1'+str(config.S1)+'_S2'+str(config.S2)+'_'+str(datetime.now())#str(uuid.uuid4())
    else:
        exp_name = config.model_type+'_'+config.atlas_name+str(config.n_rois)+'_'+str(datetime.now())#str(uuid.uuid4())
    save_dir = os.path.join(config.main_dir, 'results/'+config.dataset+'/'+exp_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    # making sure parameters make sense wrt. other parameters
    if config.dataset == 'synthetic':
        config.threshold_annealing = False # maybe we can actually create a similar method for synthetic data? just making the graph denser over iterations?
    if config.dataset == 'hcp' or config.dataset == 'decnef':
        config.matlab_compare = False
    if config.model_type == 'parametric':
        config.splitmerge = False
        config.threshold_annealing = False
    if config.threshold_annealing:
        config.maxiter_gibbs = 400
        config.use_convergence_criteria = False
        
    print(config)
        
    # log file with specifications for experiment:
    if config.dataset == 'synthetic':
        with open(os.path.join(save_dir, 'log.txt'), 'w') as f:
            f.write(f"dataset: {config.dataset}\n")
            f.write(f"exp_name: {exp_name}\n")
            f.write(f"matlab_compare: {config.matlab_compare}\n")
            f.write(f"N: {config.N}\n")
            f.write(f"K: {config.K}\n")
            f.write(f"S1: {config.S1}\n")
            f.write(f"S2: {config.S2}\n")
            f.write(f"balance_Nc: {config.balance_Nc}\n")
            f.write(f"eta_similarity: {config.eta_similarity}\n")
            f.write(f"model_type: {config.model_type}\n")
            f.write(f"splitmerge: {config.splitmerge}\n")
            f.write(f"noc: {config.noc}\n")
            f.write(f"maxiter_gibbs: {config.maxiter_gibbs}\n")
            f.write(f"maxiter_eta0: {config.maxiter_eta0}\n")
            f.write(f"maxiter_alpha: {config.maxiter_alpha}\n")
            #f.write(f"total_time_min: {elapsed_time}\n")
    elif config.dataset == 'hcp' or config.dataset == 'decnef':
        with open(os.path.join(save_dir, 'log.txt'), 'w') as f:
            f.write(f"dataset: {config.dataset}\n")
            f.write(f"exp_name: {exp_name}\n")
            f.write(f"atlas_name: {config.atlas_name}\n")
            f.write(f"n_rois: {config.n_rois}\n")
            f.write(f"threshold_annealing: {config.threshold_annealing}\n")
            f.write(f"model_type: {config.model_type}\n")
            f.write(f"splitmerge: {config.splitmerge}\n")
            f.write(f"noc: {config.noc}\n")
            f.write(f"maxiter_gibbs: {config.maxiter_gibbs}\n")
            f.write(f"maxiter_eta0: {config.maxiter_eta0}\n")
            f.write(f"maxiter_alpha: {config.maxiter_alpha}\n")
            #f.write(f"total_time_min: {elapsed_time}\n")
    else: 
        print('Unknown dataset. Please choose between synthetic, hcp or decnef.')
    
    start_time = time.time()
    
    #%% Run code
    print('Using ' + config.dataset + ' dataset')
    model = MultinomialSBM(config)

    model.train()
    # SAVE MODEL OUTPUTS
    np.save(os.path.join(save_dir,'model_sample.npy'), model.sample)

    elapsed_time = (time.time() - start_time) /60
    print('total_time_min:', elapsed_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data configuration.
    parser.add_argument('--dataset', type=str, default='synthetic', help='dataset name (synthetic, hcp, decnef)')
        # Synthetic data configuration. 
    parser.add_argument('--N', type=int, default=100, help='number of nodes in each cluster (synthetic data)')
    parser.add_argument('--K', type=int, default=5, help='number of clusters (synthetic data)')
    parser.add_argument('--S1', type=int, default=5, help='number of graphs of type 1 (synthetic data)')
    parser.add_argument('--S2', type=int, default=5, help='number of graphs of type 2 (synthetic data)')
    parser.add_argument('--balance_Nc', type=bool, default=None, help='if size of clusters (no. of nodes in each cluster) should be balanced or not (True/False)')
    parser.add_argument('--eta_similarity', type=str, default='comp_diff', help='same, comp_diff or part_diff (how similar eta1 and eta2 should be)')

        # MRI data configurations (fMRI and/or dMRI)
    parser.add_argument('--atlas_name', type=str, default='schaefer', help='atlas name (schaefer)')
    parser.add_argument('--n_rois', type=int, default=100, help='number of ROIs (hcp/decnef data): 100, 200 or 300')
    parser.add_argument('--threshold_annealing', type=bool, default=False, help='use threshold annealing (True/False). If True, threshold annealing is used to increase graph density over iterations')
    
    # Model configuration.
    parser.add_argument('--model_type', type=str, default='nonparametric', help='model type (nonparametric/parametric)')
    parser.add_argument('--noc', type=int, default=50, help='intial number of clusters')
    parser.add_argument('--splitmerge', type=bool, default=True, help='use splitmerge for nonparametric model (True/False)')
    
    # Training configuration.
    parser.add_argument('--maxiter_gibbs', type=int, default=1000, help='max number of gibbs iterations')
    parser.add_argument('--maxiter_eta0', type=int, default=10, help='max number of MH iterations for sampling eta0')
    parser.add_argument('--maxiter_alpha', type=int, default=100, help='max number of MH iterations for sampling alpha')
    parser.add_argument('--maxiter_splitmerge', type=int, default=10, help='max number of splitmerge iterations')
    parser.add_argument('--unit_test', type=bool, default=False, help='perform unit test (True/False)')
    parser.add_argument('--matlab_compare', type=bool, default=False, help='compare to matlab code (True/False). If True, random variables are initiated from the saved random variables in folder matlab_randvar')
    parser.add_argument('--use_convergence_criteria', type=bool, default=True, help='use convergence criteria (True/False). If True, the algorithm stops when the convergence criteria is met')
    
    # Miscellaneous.
    parser.add_argument('--main_dir', type=str, default='/work3/s174162/speciale/', help='main directory')
    parser.add_argument('--disp', type=bool, default=True, help='display iteration results (True/False)')
    parser.add_argument('--sample_step', type=int, default=1, help='number of iterations between each saved/logged sample')

    config = parser.parse_args()
    main(config)
    #cProfile.run('main(config)')