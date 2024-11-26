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
#                                      INFERENCE/COMBAT 
#                                 CHANGE type_nm and refire
#
##########################################################################################################


import warnings
warnings.filterwarnings('ignore')

type_nm = 'inference' # inference or combat
refire = False 

df = pd.read_parquet(f'/homes/dkurtenb/projects/russat/output/MODEL_{type_nm}_tle_data.parquet')

def build_anom_model(NORAD_ID_NUM, type_nm):
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 
        torch.backends.cudnn.benchmark = True
    
    os.makedirs(f"/homes/dkurtenb/projects/russat/output/{type_nm}_plots/plots_{type_nm}_{NORAD_ID_NUM}", exist_ok=True)

    samp_df = df[df['NORAD_CAT_ID']==NORAD_ID_NUM]
    samp_df = samp_df.sort_values(by='datetime', ascending=False)
    orb_df = samp_df[['datetime','inclination','ra_of_asc_node', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 'mean_motion']]
    orb_df = orb_df.set_index('datetime', drop = True)
    
    plot_save_dir = f"/homes/dkurtenb/projects/russat/output/{type_nm}_plots/plots_{type_nm}_{NORAD_ID_NUM}"
    
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

###############################################################################################################################################################
files = os.listdir("/homes/dkurtenb/projects/russat/output/{type_nm}_plots")

trained = list({int(f.split("_")[-1].strip("'\"")) for f in files})
sat = df['NORAD_CAT_ID'].unique().tolist()

sats_to_train = list(set(sat) - set(trained))

sats_to_train.insert(0,28908)
###############################################################################################################################################################

all_dfs = []

if refire:
    id_lst = sats_to_train
else:
    id_lst = unique_ids

for count, x in enumerate(id_lst, 1):
    print(f'\n\n\nWorking on space object number {x}\n\n\n')
    orb_df, detector, anomalies, explanations, samp_df = build_anom_model(x, type_nm)
    
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

    mode = 'w' if count == 1 else 'a'
    header = count == 1
    full_df.to_csv(f'/homes/dkurtenb/projects/russat/output/anom_df_{type_nm}_all_sats_0.csv', mode=mode, header=header, index=False)
    
    progress = count/df['NORAD_CAT_ID'].nunique()*100
    print(f"\nModel number {count} out of {df['NORAD_CAT_ID'].nunique()} complete. Progress: {progress:.3f}% done\n")

    del full_df, orb_df, detector, anomalies, explanations, samp_df
    del anom_df, anom_dict

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
