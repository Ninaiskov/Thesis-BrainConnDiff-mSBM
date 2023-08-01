# Master thesis (speciale): Modeling Individual Differences in Functional Brain Connectivity using Stochastic Block Models

### STRUCTURE (Filename and description)

- main.py: Main script for running model
- model.py: Multinomial Stochastic Block Model (mSBM) class
- generate_graphs.py: Generate adjacency matrices (graphs) from fMRI and dMRI images  
- helper_functions.py: Helper functions
- final_results.py: Produce final classification results for report
- run_fmri_batchjobs.sh: Submit multiple batchjobs (fMRI data experiments)
- run_syn_batchjobs.sh: Submit multiple batchjobs (synthetic data experiments)
- submit_big.sh: Submit single batchjobs to BIG cluster
- submit_hpc.sh: Submit single batchjobs to HPC cluster
- visualize.ipynb: Visualize data and different model outputs
- clf_featureprep.ipynb: Preparing classification features
- misc_code.ipynb: Miscallaneous code for experiment overview, cleaning up error files, etc.
- speciale.yml: Conda environment
