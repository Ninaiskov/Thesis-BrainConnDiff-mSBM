import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nilearn as ni
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
import nilearn.connectome
from nilearn import plotting as nplt
import scipy.io
from helper_functions import butter_bandpass_filter

################################################## INITIALIZATION #######################################################
dataset = 'decnef' # dataset to generate graphs from: 'hcp' or 'decnef

# atlas_name = name of nilearn atlas for brain parcellation
# n_rois = number of ROIs in atlas (default n_rois = 400 for schaefer)
# threshold = used for thresholding correlations to create binary adjacency matrices (0 is chosen based on histogram, but not optimal)

atlas_name = 'schaefer'
n_rois = 300

start_time = time.time()

my_dir = '/work3/s174162/speciale/'

if dataset == 'decnef':
    data_dir = '/work3/khma/SRPBS_OPEN/BIDS/derivs/fmriprep/'
    
    # create subject list with healthy and schizophrenia subjects
    healthy_ids = np.load(os.path.join(my_dir, 'data/healthy_ids.npy'), allow_pickle=True)
    schizo_ids = np.load(os.path.join(my_dir, 'data/schizo_ids.npy'), allow_pickle=True)
    sublist = np.concatenate((healthy_ids, schizo_ids)).tolist()
    #sublist.sort() # sort alphabetically (or keep order as healthy subs first and then schizo subs)
    
    # define atlas
    atlas = ni.datasets.fetch_atlas_schaefer_2018(n_rois=n_rois)
    labels = atlas.labels

    # connectivity measure
    conn = ni.connectome.ConnectivityMeasure(kind='correlation')
elif dataset == 'hcp':
    data_dir = my_dir+'data/hcp/'
    sublist = np.arange(500) # 250 subjects with 2 scans each (fmri and dmri)
else:
    print('Unknown dataset')
save_folder = atlas_name+str(n_rois)

save_dir = os.path.join(my_dir, 'data', dataset, save_folder)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
        
################################################## FUNCTIONS #######################################################
def get_corr(dataset, data_dir, sub, atlas, conn):
    #print('Computing correlations')
    # function for loading fmri data and extracting correlation matrix. Output is vector containing values from upper triangle of correlation matrix (excluding diagonal)
    
    if dataset == 'decnef':
        # extract functional and anatomical images (fmri and anat)
        fmri = nib.load(data_dir+sub+'/func/'+sub+'_task-rest_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
        #anat = nib.load(data_dir+sub+'/anat/'+sub+'_desc-preproc_T1w.nii.gz')
        confounds = pd.read_csv(data_dir+sub+'/func/'+sub+'_task-rest_run-1_desc-confounds_timeseries.tsv', sep = '\t')
        global_signal = confounds.global_signal.values
        
        # define masker and fit to fmri data (i.e. use atlas to parcelate data)
        masker = NiftiLabelsMasker(atlas['maps'], standardize=True, verbose=5, detrend=True) # error if using memory='nilearn_cache'
        masker.fit(fmri)
        
        # extract time series: masker.transform(fmri)
        time_series = masker.transform(fmri, confounds = global_signal)
        
        # high pass filter time series using butter bandpass filter
        tr = fmri.header.get_zooms()[-1] # resolution is in header (last dimension is "tr = time resolution": in seconds)
        time_series_filt = butter_bandpass_filter(time_series.T, fs=1/tr).T
        
        # compute correlation between timeseries
        corr_mat = conn.fit_transform([time_series_filt]) # correlation matrix
        corr_mat = np.triu(corr_mat, k=1)[0] # correlations from upper triangular matrix (including zeroes below diagonal)
        
    elif dataset == 'hcp':
        if sub < 250: # load fmri
            corr_cell = scipy.io.loadmat(data_dir+atlas_name+str(n_rois)+'/corr'+str(n_rois)+'_fmri.mat')['A'][0]
            corr_mat = corr_cell[sub].toarray()
        else: # load dmri
            corr_cell = scipy.io.loadmat(data_dir+atlas_name+str(n_rois)+'/corr'+str(n_rois)+'_dmri.mat')['A'][0]
            corr_mat = corr_cell[sub-250].toarray()
        corr_mat = np.triu(corr_mat, k=1) # correlations from upper triangular matrix (including zeroes below diagonal)
    else: 
        print('Unknown dataset')
    # save only upper triangular part of correlation matrix since it is symmetric (excluding main diagonal)
    corr_vec = corr_mat[np.triu_indices(corr_mat.shape[-1],1)] # correlations from upper triangular matrix (without zeroes below diagonal)
    return corr_vec

def compute_percentiles(corr):
    #print('Computing percentiles')
    lowest_percentile = 6.25
        
    percentiles = np.percentile(corr, np.arange(lowest_percentile, 100, lowest_percentile))
    # percentile:      [ 6.25, 12.5 , 18.75, 25.  , 31.25, 37.5 , 43.75, 50., 56.25, 62.5 , 68.75, 75.  , 81.25, 87.5 , 93.75]
    # percentile idx:  [ 0   ,  1   ,   2  , 3    ,  4   ,   5  ,  6   , 7  ,  8   ,   9  ,  10  , 11   , 12   , 13   , 14] 
    
    # threshold Acor to get integer-weighted adjacency matrix (using percentiles as threshold)
    # adjacency matrix values stored as vector
    
    # first graph - thresholding with 50th percentile
    conditions1 = [corr <= percentiles[7], corr > percentiles[7]]
    values1 = [0, 1]
    A1_vals = np.select(conditions1, values1, default=np.nan)

    # second graph - thresholding with 25h percentile
    conditions2 = [corr <= percentiles[3], # np.percentile(corr, 25)
                    (corr > percentiles[3]) & (corr <= percentiles[7]), #np.percentile(corr, 25) and np.percentile(corr, 50)
                    (corr > percentiles[7]) & (corr <= percentiles[11]), #np.percentile(corr, 50) and np.percentile(corr, 75)
                    corr > percentiles[11]] # np.percentile(corr, 75)]
    values2 = [0, 1, 2, 3]
    A2_vals = np.select(conditions2, values2, default=np.nan)
    
    # third graph - thresholding with 12.5th percentile
    conditions3 = [corr <= percentiles[1], # np.percentile(corr, 12.5)
                    (corr > percentiles[1]) & (corr <= percentiles[3]), # np.percentile(corr, 12.5) and np.percentile(corr, 25)
                    (corr > percentiles[3]) & (corr <= percentiles[5]), # np.percentile(corr, 25) and np.percentile(corr, 37.5)
                    (corr > percentiles[5]) & (corr <= percentiles[7]), # np.percentile(corr, 37.5) and np.percentile(corr, 50)
                    (corr > percentiles[7]) & (corr <= percentiles[9]), # np.percentile(corr, 50) and np.percentile(corr, 62.5)
                    (corr > percentiles[9]) & (corr <= percentiles[11]), # np.percentile(corr, 62.5) and np.percentile(corr, 75)
                    (corr > percentiles[11]) & (corr <= percentiles[13]), # np.percentile(corr, 75) and np.percentile(corr, 87.5)
                    corr > percentiles[13]] # np.percentile(corr, 87.5)]
    values3 = [0, 1, 2, 3, 4, 5, 6, 7]
    A3_vals = np.select(conditions3, values3, default=np.nan)

    # fourth graph - thresholding with 6.25th percentile
    conditions4 = [corr <= percentiles[0], # np.percentile(corr, 6.25)
                    (corr > percentiles[0]) & (corr <= percentiles[1]), # np.percentile(corr, 6.25) and np.percentile(corr, 12.5)
                    (corr > percentiles[1]) & (corr <= percentiles[2]), # np.percentile(corr, 12.5) and np.percentile(corr, 18.75)
                    (corr > percentiles[2]) & (corr <= percentiles[3]), # np.percentile(corr, 18.75) and np.percentile(corr, 25)
                    (corr > percentiles[3]) & (corr <= percentiles[4]), # np.percentile(corr, 25) and np.percentile(corr, 31.25)
                    (corr > percentiles[4]) & (corr <= percentiles[5]), # np.percentile(corr, 31.25) and np.percentile(corr, 37.5)
                    (corr > percentiles[5]) & (corr <= percentiles[6]), # np.percentile(corr, 37.5) and np.percentile(corr, 43.75)
                    (corr > percentiles[6]) & (corr <= percentiles[7]), # np.percentile(corr, 43.75) and np.percentile(corr, 50)
                    (corr > percentiles[7]) & (corr <= percentiles[8]), # np.percentile(corr, 50) and np.percentile(corr, 56.25)
                    (corr > percentiles[8]) & (corr <= percentiles[9]), # np.percentile(corr, 56.25) and np.percentile(corr, 62.5)
                    (corr > percentiles[9]) & (corr <= percentiles[10]), # np.percentile(corr, 62.5) and np.percentile(corr, 68.75)
                    (corr > percentiles[10]) & (corr <= percentiles[11]), # np.percentile(corr, 68.75) and np.percentile(corr, 75)
                    (corr > percentiles[11]) & (corr <= percentiles[12]), # np.percentile(corr, 75) and np.percentile(corr, 81.25)
                    (corr > percentiles[12]) & (corr <= percentiles[13]), # np.percentile(corr, 81.25) and np.percentile(corr, 87.5)
                    (corr > percentiles[13]) & (corr <= percentiles[14]), # np.percentile(corr, 87.5) and np.percentile(corr, 93.75)
                    corr > percentiles[14]] # np.percentile(corr, 93.75)]
    values4 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    A4_vals = np.select(conditions4, values4, default=np.nan)
    
    return percentiles, A1_vals, A2_vals, A3_vals, A4_vals


################################################## MAIN LOOP #######################################################

## loop over subjects
failed_subs = [] # list of subjects that failed during load
corr_sublist = [] # list of correlation vectors for each subject
percentiles_sublist = [] # list of list of percentiles for each subject
A1_vals_list = [] # upper triangular adjacency matrix values for all subjects (stored as vectors for each subject)
A2_vals_list = []
A3_vals_list = []
A4_vals_list = []
for sub in sublist:
    if dataset == 'decnef':
        if os.path.exists(data_dir+sub+'/func/'):
            corr = get_corr(dataset, data_dir, sub, atlas, conn) # corr as vector
            percentiles, A1_vals, A2_vals, A3_vals, A4_vals = compute_percentiles(corr)
        else: 
            failed_subs.append(sub) # sub-0889
    elif dataset == 'hcp':
        corr = get_corr(dataset, data_dir, sub, atlas=None, conn=None) # corr as vector
        percentiles, A1_vals, A2_vals, A3_vals, A4_vals = compute_percentiles(corr)
    else:
        print('Unknown dataset')
    corr_sublist.append(corr)
    percentiles_sublist.append(percentiles)
    A1_vals_list.append(A1_vals)
    A2_vals_list.append(A2_vals)
    A3_vals_list.append(A3_vals)
    A4_vals_list.append(A4_vals)
        
        
## output 
# graphs (adjacency matrices) for each subject: numpy arrays   
np.save(os.path.join(save_dir,'corr_sublist.npy'), corr_sublist)
np.save(os.path.join(save_dir,'percentiles_sublist.npy'), percentiles_sublist)
np.save(os.path.join(save_dir,'A1_vals_list.npy'), A1_vals_list) # for all subjects
np.save(os.path.join(save_dir,'A2_vals_list.npy'), A2_vals_list)
np.save(os.path.join(save_dir,'A3_vals_list.npy'), A3_vals_list)
np.save(os.path.join(save_dir,'A4_vals_list.npy'), A4_vals_list)

elapsed_time = (time.time() - start_time) /60

# log file with specifications for graph construction:
with open(os.path.join(save_dir, 'log.txt'), 'w') as f:
    f.write(f"Dataset: {dataset}\n")
    f.write(f"Atlas name: {atlas_name}\n")
    f.write(f"Number of ROIs: {n_rois}\n")
    f.write(f"Total number of subjects: {len(sublist)-len(failed_subs)}\n")
    f.write(f"Time [min] spent: {elapsed_time}\n")
    f.write(f"Subject list:\n")
    for sub in sublist:
        f.write(f'{sub}\n')
    f.write(f"Subjects that failed:\n")
    for sub in failed_subs:
        f.write(f'{sub}\n')
            
        
