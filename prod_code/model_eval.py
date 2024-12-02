#####################################################################################
#
#                          Prod Code - Model Evaluation
#
#####################################################################################


from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import os
import pickle
import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
from multiprocessing import Pool

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/homes/dkurtenb/projects/russat/output/anom_df_combat_all_sats_0.csv')
save_dir = '/homes/dkurtenb/projects/russat/output'

cols = ['inclination','ra_of_asc_node', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 'mean_motion']

def calculate_outlier(df, column_nm, threshold = 2, method = 'iqr'):
    z_scores = np.abs((df[column_nm] - df[column_nm].mean())/df[column_nm].std())
    df[f'outlier_{column_nm}'] = (z_scores > threshold).astype(int)

    if method=='iqr':
        # IQR Method returned ~ 2% outlier
        q1 = df[column_nm].quantile(0.25)
        q3 = df[column_nm].quantile(0.75)
        iqr = q3 - q1
        df[f'outlier_{column_nm}'] = ((df[column_nm] < (q1 - 1.5 * iqr)) | (df[column_nm] > (q3 + 1.5 *iqr))).astype(int)
    
    elif method =='zscore':
        # ~ 5% outlier with threshold=2
        z_scores = np.abs((df[column_nm] - df[column_nm].mean())/df[column_nm].std())
        df[f'outlier_{column_nm}'] = (z_scores > threshold).astype(int)

    elif method =='mad':
        #~9% outliers with threshold=2 
        median = df[column_nm].median()
        mad = median_abs_deviation(df[column_nm])
        modified_z = 0.6745 * (df[column_nm] - median) / mad
        df[f'outlier_{column_nm}'] = (abs(modified_z) > threshold).astype(int)

    else:
        print('Not a valid outlier detection method')

    df['outlier'] = (df.iloc[:, -6:] == 1).any(axis=1).astype(int)

count = 0 
tle_df = pd.DataFrame()

for id in df['NORAD_CAT_ID'].unique():
    print(f'Working in space object: {id}')
    bysat_df = df[df['NORAD_CAT_ID']==id]
    for c in cols:
        calculate_outlier(bysat_df,c, threshold = 2, method = 'mad')
    tle_df = pd.concat([tle_df, bysat_df], axis=0)
    count += 1
    progress = (count/len(df['NORAD_CAT_ID'].unique())) * 100
    print(f"Outlier calculation is {progress} percent complete")

tle_df['anomaly_ind'] = (tle_df['anom_count'] >= 1).astype(int)

acc_lst = []
prec_lst =[]
recall_lst = []
f1_lst =[]
count = 0 

for id in tle_df['NORAD_CAT_ID'].unique():
   print(f'Evaluating model for {id}') 
   samp_df = tle_df[tle_df['NORAD_CAT_ID']==id]
   prediction_columns = ['anomaly_ind']
   gt_column = 'outlier'

   for col in prediction_columns:
      correct = (tle_df[col] == tle_df['outlier']).mean()
      acc_lst.append(correct)
      #print(f"\nAccuracy for {col}: {correct:.5f}")

      precision = precision_score(tle_df['outlier'], tle_df[col], average='weighted')
      prec_lst.append(precision)
      #print(f"Precision for {col}: {precision:.3f}")
      
      recall = recall_score(tle_df['outlier'], tle_df[col], average='weighted')
      recall_lst.append(recall)
      #print(f"Recall for {col}: {recall:.3f}")

      f1 = f1_score(tle_df['outlier'], tle_df[col], average='weighted')
      f1_lst.append(f1)
      #print(f"F1 score for {col}: {f1:.3f}")

   
metrics_df = pd.DataFrame({
    'NORAD_CAT_ID': tle_df['NORAD_CAT_ID'].unique(),
    'accuracy': acc_lst,
    'precision': prec_lst,
    'recall': recall_lst,
    'f1_score': f1_lst
})


metrics_df.to_csv(os.path.join(save_dir,'metrics_results.csv'), index=False)

print('^^^^^^^^^^^^^^^^^^ DONE ^^^^^^^^^^^^^^^^^^^^^^^^') 

