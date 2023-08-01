import numpy as np
import pandas as pd
import os
from ast import literal_eval
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from helper_functions import classification_pipeline, classification_results

main_dir = '/work3/s174162/speciale'
df = pd.read_csv(os.path.join(main_dir, 'results', 'experiment_overview.csv'), header=0, sep=',', converters={'n_rois': literal_eval,
                                                                                                        'exp_name_list': literal_eval,
                                                                                                        'N': literal_eval,
                                                                                                        'K': literal_eval,
                                                                                                        'S1': literal_eval,
                                                                                                        'S2': literal_eval})

# Define the classifiers and their corresponding hyperparameters
class_weight = 'balanced'

#classifiers = {
#    'LR': (LogisticRegression(max_iter=1000, class_weight=class_weight), {'logisticregression__C': [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]}),
#    'RF': (RandomForestClassifier(class_weight=class_weight), {'randomforestclassifier__n_estimators': [10, 20, 50, 75, 100, 150, 200, 250, 300, 400]}), #max_depth: None for unlimited depth.
#    'KNN': (KNeighborsClassifier(), {'kneighborsclassifier__n_neighbors': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]}), # KNN has no class weight parameter
#    'SVM': (SVC(class_weight=class_weight), {'svc__C': [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]})#, 'svc__kernel': ['linear', 'rbf']})
#}
classifiers = {
    'LR': (LogisticRegression(max_iter=1000, class_weight=class_weight), {'logisticregression__C': [0.1, 1, 2, 5, 10]}),
    'RF': (RandomForestClassifier(class_weight=class_weight), {'randomforestclassifier__n_estimators': [5, 10, 20, 50, 100]}), #max_depth: None for unlimited depth.
    'KNN': (KNeighborsClassifier(), {'kneighborsclassifier__n_neighbors': [3, 5, 10, 15, 20]}), # KNN has no class weight parameter
    'SVM': (SVC(class_weight=class_weight), {'svc__C': [0.1, 1, 2, 5, 10]})#, 'svc__kernel': ['linear', 'rbf']})
}

#for dataset in ['decnef', 'hcp']: 
dataset = 'decnef'
print(dataset)

##### Running results for feature type: demographic info
if dataset == 'decnef':
    target_type_list = ['schizo', 'male', 'site', 'hand']
    feature_type = 'demoinfo'
    results = {}
    for target_type in target_type_list:
        for i, (clf_name, (clf, param_grid)) in enumerate(classifiers.items()):
            mean_accuracy, mean_F1, best_params = classification_pipeline(clf=clf, 
                                                                                        param_grid=param_grid, 
                                                                                        target_type=target_type, 
                                                                                        feature_type=feature_type, 
                                                                                        dataset=dataset)
            results[clf_name] = [mean_accuracy, mean_F1, best_params]
            np.save(os.path.join(main_dir,'results',dataset,target_type+'_'+feature_type+'_results.npy'), results)
elif dataset == 'hcp':
    target_type = 'male' # only for target_type male - doesn't makes sense to predict structconn from demoinfo (age)
    feature_type = 'demoinfo'
    results = {}
    for i, (clf_name, (clf, param_grid)) in enumerate(classifiers.items()):
        mean_accuracy, mean_F1, best_params = classification_pipeline(clf=clf, 
                                                                                    param_grid=param_grid, 
                                                                                    target_type=target_type, 
                                                                                    feature_type=feature_type, 
                                                                                    dataset=dataset)
        results[clf_name] = [mean_accuracy, mean_F1, best_params]
        np.save(os.path.join(main_dir,'results',dataset,target_type+'_'+feature_type+'_results.npy'), results)

##### Running results for feature type: Demographic info + correlation matrix
if dataset == 'hcp':
    feature_type = 'corr'
    target_type_list = ['structconn', 'male']
elif dataset == 'decnef':
    feature_type = 'demoinfo_corr'
    target_type_list = ['schizo', 'male', 'site', 'hand']
else:
    print('Unknown dataset')

results = {}
for n_rois in [100, 200, 300]:
    for target_type in target_type_list:
        for i, (clf_name, (clf, param_grid)) in enumerate(classifiers.items()):
            mean_accuracy, mean_F1, best_params = classification_pipeline(clf=clf, 
                                                                                            param_grid=param_grid, 
                                                                                            target_type=target_type, 
                                                                                            feature_type=feature_type, 
                                                                                            dataset=dataset, 
                                                                                            n_rois=n_rois)
            results[clf_name] = [mean_accuracy, mean_F1, best_params]
            np.save(os.path.join(main_dir,'results',dataset,target_type+'_'+feature_type+str(n_rois)+'_results.npy'), results)

##### Running results for feature type: Demographic info + eta
if dataset == 'hcp':
    feature_type = 'eta'
elif dataset == 'decnef':
    feature_type = 'demoinfo_eta'
else:
    print('Unknown dataset')

results = {}
for target_type in target_type_list:
    for i, (clf_name, (clf, param_grid)) in enumerate(classifiers.items()):
        max_mean_accuracy, max_mean_F1, best_params = classification_results(dataset=dataset, 
                                                                                df=df, 
                                                                                target_type=target_type, 
                                                                                feature_type=feature_type, 
                                                                                clf=clf, 
                                                                                clf_name=clf_name, 
                                                                                param_grid=param_grid)
        results[clf_name] = [max_mean_accuracy, max_mean_F1, best_params]
        np.save(os.path.join(main_dir,'results',dataset,target_type+'_'+feature_type+'_results.npy'), results)