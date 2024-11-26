import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter
import os
import joblib
from scipy.stats import median_abs_deviation
from skyfield.api import load, EarthSatellite
from datetime import datetime, timezone
import numpy as np
import sys
sys.path.append('/homes/dkurtenb/projects/russat/prod_code')
from inf_helpers import *

##########################################################################################################
#
#                                      COMBAT
#
##########################################################################################################


import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('/homes/dkurtenb/projects/russat/output/MODEL_combat_tle_data.parquet')

def build_anom_model(NORAD_ID_NUM, type):
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 
        torch.backends.cudnn.benchmark = True
    
    os.makedirs(f"/homes/dkurtenb/projects/russat/output/{type}_plots/plots_{type}_{NORAD_ID_NUM}", exist_ok=True)

    samp_df = df[df['NORAD_CAT_ID']==NORAD_ID_NUM]
    samp_df = samp_df.sort_values(by='datetime', ascending=False)
    orb_df = samp_df[['datetime','inclination','ra_of_asc_node', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 'mean_motion']]
    orb_df = orb_df.set_index('datetime', drop = True)
    
    plot_save_dir = f"/homes/dkurtenb/projects/russat/output/{type}_plots/plots_{type}_{NORAD_ID_NUM}"
    
    feature_names = list(orb_df)
    
    detector, anomalies, explanations, timestamps, anomaly_details = run_anomaly_detection_pipeline(
        orb_df,
        feature_names=feature_names,
        model_path="/homes/dkurtenb/projects/russat/output/anomaly_model",
        should_train=False,
        NORAD_ID_NUM=NORAD_ID_NUM,  
        plot_save_dir=plot_save_dir  
    )
    
    return orb_df, detector, anomalies, explanations, samp_df   

unique_ids = df['NORAD_CAT_ID'].unique()
total_sats = len(unique_ids)

all_orbital_features = ['inclination', 'ra_of_asc_node', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 'mean_motion']
anom_columns = [f'anom_{feat}' for feat in all_orbital_features]

type_nm = 'combat'

all_dfs = []

for count, x in enumerate(unique_ids, 1):
    orb_df, detector, anomalies, explanations, samp_df = build_anom_model(x,type_nm)
    
    anom_dict = {exp['sample_index']: [feat['feature'] for feat in exp['anomalous_features']] for exp in explanations}  

    full_df = samp_df.copy(deep=False)
    full_df.reset_index(inplace=True, drop = True)

    all_features = set().union(*[set(features) for features in anom_dict.values()]) 
    
    anom_df = pd.DataFrame(0, 
                        index=anom_dict.keys(),
                        columns=anom_columns)


    for key, features in anom_dict.items():
        anom_df.loc[key, [f'anom_{feat}' for feat in features]] = 1
    

    full_df = full_df.join(anom_df, how='left')

    for col in anom_columns:
        if col not in full_df.columns:
            full_df[col] = 0

    full_df['anom_count'] = full_df.filter(like='anom_').sum(axis=1)
    full_df = full_df.fillna(0)

    all_dfs.append(full_df)

    if count % 100 == 0:  # Adjust this number based on your memory constraints
        intermediate_df = pd.concat(all_dfs, ignore_index=True)
        intermediate_df.to_parquet(f'/homes/dkurtenb/projects/russat/output/anom_df_{type_nm}_all_sats_intermediate.parquet')
        all_dfs = [intermediate_df]

    progress = count/df['NORAD_CAT_ID'].nunique()*100
    print(f"\nModel number {count} out of {df['NORAD_CAT_ID'].nunique()} complete. Progress: {progress:.3f}% done\n")
    
    del full_df, orb_df, detector, anomalies, explanations, samp_df
    del anom_df, anom_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

final_df = pd.concat(all_dfs, ignore_index=True)
final_df.to_parquet(f'/homes/dkurtenb/projects/russat/output/anom_df_{type_nm}_all_sats.parquet')

import os
intermediate_file = f'/homes/dkurtenb/projects/russat/output/anom_df_{type_nm}_all_sats_intermediate.parquet'
if os.path.exists(intermediate_file):
    os.remove(intermediate_file)