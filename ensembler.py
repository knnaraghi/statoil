import os
import pandas as pd
import numpy as np

## This ensemble method is adapted from this Kaggle Kernel found @ 
## https://www.kaggle.com/submarineering/submarineering-best-public-score-until-now
## I used mean ensembling for similar models and the min-max with best-base strategy outlined in the link for dissimillar models
## Make sure to edit path and best_base if necessary for your local environment 

def ensemble(strategy='mean', best_base=None):
    
    path = ""
    all_files = os.listdir(path)

    output = [pd.read_csv(os.path.join(path, f), index_col=0) for f in all_files]
    concat_sub = pd.concat(output, axis=1)
    cols = list(map(lambda x: "is_iceberg_" + str(x), range(len(concat_sub.columns))))
    end_idx = len(cols) + 1
    concat_sub.columns = cols
    concat_sub.reset_index(inplace=True)
    
    ## Check correlation of models
    print(concat_sub.corr())
    
    concat_sub['is_iceberg_max'] = concat_sub.iloc[:, 1:end_idx].max(axis=1)
    concat_sub['is_iceberg_min'] = concat_sub.iloc[:, 1:end_idx].min(axis=1)
    concat_sub['is_iceberg_mean'] = concat_sub.iloc[:, 1:end_idx].mean(axis=1)
    concat_sub['is_iceberg_median'] = concat_sub.iloc[:, 1:end_idx].median(axis=1)

    cutoff_lo = 0.8
    cutoff_hi = 0.2
    
    if strategy=='best_base':
        best_base = pd.read_csv(best_base)
        concat_sub['is_iceberg_base'] = best_base['is_iceberg']
        concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:end_idx] > cutoff_lo, axis=1), 
                                    concat_sub['is_iceberg_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:end_idx] < cutoff_hi, axis=1),
                                             concat_sub['is_iceberg_min'], 
                                             concat_sub['is_iceberg_base']))
        concat_sub['is_iceberg'] = np.clip(concat_sub['is_iceberg'].values, 0.001, 0.999)
        concat_sub[['id', 'is_iceberg']].to_csv('submission.csv', index=False, float_format='%.6f')
    else:
        concat_sub['is_iceberg'] = concat_sub['is_iceberg_mean']
        concat_sub[['id', 'is_iceberg']].to_csv('stack_mean.csv', index=False, float_format='%.6f')
        
ensemble(strategy='mean', best_base='')