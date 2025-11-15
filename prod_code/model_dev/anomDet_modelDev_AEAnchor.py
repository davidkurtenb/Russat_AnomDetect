#########################################################
#          PROD CODE - AE Anchor Model Dev
#########################################################

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(r'C:\Users\dk412\Desktop\David\Python Projects\RusSat\prod_code\model_dev\helpers')
from helpers import *

import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_parquet(r'C:\Users\dk412\Desktop\David\Python Projects\RusSat\data\model_dev\MODEL_training_tle_data_outlier.parquet')
train_df = train_df[(train_df['datetime'] >= '2017-08-24') & (train_df['datetime'] <= '2021-08-24')]

def build_anom_model(NORAD_ID_NUM, test_size=0.2, random_state=42):

    if torch.cuda.is_available():
        torch.cuda.empty_cache() 
        torch.backends.cudnn.benchmark = True

    os.makedirs(f"/homes/dkurtenb/projects/russat/output/model_training/kan/plots/plots_training_{NORAD_ID_NUM}", exist_ok=True)

    samp_df = train_df[train_df['NORAD_CAT_ID']==NORAD_ID_NUM]
    samp_df = samp_df.sort_values(by='datetime', ascending=False)
    orb_df = samp_df[['datetime','inclination','ra_of_asc_node', 'eccentricity','arg_of_perigee', 'mean_anomaly', 'mean_motion']]
    orb_df = orb_df.set_index('datetime', drop=True)

    labels = samp_df['outlier'].values
    
    train_size = int((1 - test_size) * len(orb_df))
    X_train = orb_df.iloc[:train_size]
    X_test = orb_df.iloc[train_size:]
    y_train = labels[:train_size]
    y_test = labels[train_size:]

    X_train_size = X_train.shape[0]
    X_test_size = X_test.shape[0]
    train_cnt_outlier = np.sum(y_train > 0) 
    test_cnt_outlier = np.sum(y_test > 0)

    plot_save_dir = f"/homes/dkurtenb/projects/russat/output/model_training/kan/plots/plots_training_{NORAD_ID_NUM}"
    feature_names = list(orb_df)
    
    detector, anomalies, explanations, timestamps, anomaly_details, losses = run_kan_anomaly_detection(
        X_train,
        feature_names=feature_names,
        model_path="/homes/dkurtenb/projects/russat/output/model_training/kan/anomaly_model_KAN_4year",
        should_train=True,
        NORAD_ID_NUM=NORAD_ID_NUM,
        plot_save_dir=plot_save_dir,
        print_plots = True
    )
    
    test_anomalies, test_anomaly_details = detector.kan_detect_anomalies(X_test.values)
    
    metrics = {
        'accuracy': accuracy_score(y_test, test_anomalies),
        'precision': precision_score(y_test, test_anomalies, zero_division=0),
        'recall': recall_score(y_test, test_anomalies, zero_division=0),
        'f1': f1_score(y_test, test_anomalies, zero_division=0)
    }
    
    return {
        'train_data': X_train,
        'test_data': X_test,
        'train_labels': y_train,
        'test_labels': y_test,
        'test_predictions': test_anomalies,
        'metrics': metrics,
        'detector': detector,
        'anomalies': anomalies,
        'explanations': explanations,
        'timestamps': timestamps,
        'anomaly_details': anomaly_details,
        'full_sample': samp_df,
        'x_train_size': X_train_size,
        'x_test_size': X_test_size,
        'train_cnt_outlier': train_cnt_outlier,
        'test_cnt_outlier':test_cnt_outlier
    }

metrics_df = pd.DataFrame(columns=['NORAD_ID', 'accuracy', 'precision', 'recall', 'f1', 'x_train_size', 'x_test_size', 'train_cnt_outlier','test_cnt_outlier'])

unique_ids = train_df['NORAD_CAT_ID'].unique()[:5]
total_sats = len(unique_ids)

all_orbital_features = ['inclination', 'ra_of_asc_node', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 'mean_motion']
anom_columns = [f'anom_{feat}' for feat in all_orbital_features]

for count, x in enumerate(unique_ids, 1):
    print(f'\n***********  Developing model for NORAD ID: {x}  **************\n')
    results = build_anom_model(x)

    metrics_row = {
        'NORAD_ID': x,
        'accuracy': results['metrics']['accuracy'],
        'precision': results['metrics']['precision'],
        'recall': results['metrics']['recall'],
        'f1': results['metrics']['f1'],
        'x_train_size': results['x_train_size'], 
        'x_test_size': results['x_test_size'], 
        'train_cnt_outlier': results['train_cnt_outlier'],
        'test_cnt_outlier': results['test_cnt_outlier']
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics_row])], ignore_index=True)

    metrics_df.to_csv("/homes/dkurtenb/projects/russat/output/model_training/kan/model_metrics.csv", index=False)
    
    anom_dict = {exp['sample_index']: [feat['feature'] for feat in exp['anomalous_features']] 
                for exp in results['explanations']}    

    full_df = results['full_sample'].copy(deep=False)
    full_df.reset_index(inplace=True, drop=True)

    all_features = set().union(*[set(features) for features in anom_dict.values()]) 
    
    anom_df = pd.DataFrame(0, 
                        index=anom_dict.keys(),
                        columns=anom_columns)

    for key, features in anom_dict.items():
        anom_df.loc[key, [f'anom_{feat}' for feat in features]] = 1

    full_df = full_df.join(anom_df, how='left')
    full_df['anom_count'] = full_df.filter(like='anom_').sum(axis=1)
    full_df = full_df.fillna(0)

    for col in anom_columns:
        if col not in full_df.columns:
            full_df[col] = 0

    mode = 'w' if count == 1 else 'a'
    header = count == 1
    full_df.to_csv('/homes/dkurtenb/projects/russat/output/model_training/kan/anom_df_TRAIN_subset.csv', mode=mode, header=header, index=False)
    
    progress = count/train_df['NORAD_CAT_ID'].nunique()*100
    print(f"\nModel number {count} out of {train_df['NORAD_CAT_ID'].nunique()} complete. Progress: {progress:.3f}% done")
    
    del full_df, anom_df, anom_dict, results
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()