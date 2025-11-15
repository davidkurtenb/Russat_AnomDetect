############################################################################ 
#                        PROD CODE - KAN Model Dev
############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.append('/homes/dkurtenb/projects/russat/prod_code/archive')
from helpers_KANAD import * 

#model_eval_sats = [41579, 26384, 36358, 37806]
#df = train_df[train_df['NORAD_CAT_ID'].isin(model_eval_sats)].reset_index(drop=True)

parquet_file = "/homes/dkurtenb/projects/russat/data/MODEL_training_tle_data_outlier.parquet"
#train_df = pd.read_parquet(parquet_file,filters=[('NORAD_CAT_ID', 'in', model_eval_sats)]).reset_index(drop=True)
train_df = pd.read_paraquet(parquet_file)

def extract_orb_ele(df, data_col, label_col):
    time_series = np.array(df[data_col])
    labels = np.array(df[label_col])

    return time_series, labels

orb_elements_lst = ['eccentricity',
                    'mean_anomaly',
                    'inclination',
                    'mean_motion',
                    'ra_of_asc_node',
                    'arg_of_perigee']

class Args:
    #path = "./data/"
    dropout = 0.2
    hidden_size = 64
    grid_size = 25
    n_layers = 3
    epochs = 150
    early_stopping = 30
    seed = 42
    lr = 0.001
    window_size = 15
    step_size = 8
    batch_size = 32
    anomaly_fraction = 0.1
    base_save_dir = '/homes/dkurtenb/projects/russat/output/model_training/kan/'


args = Args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_smote = True

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

result_anom_df = pd.DataFrame()

for idx,i in enumerate(train_df['NORAD_CAT_ID'].unqiue()):
    sat_df = train_df[train_df['NORAD_CAT_ID']==i]
    print(f'TOTAL NUMBER OF NULL VALUES: {sat_df["outlier_iqr_mean_anomaly"].isnull().sum()}')
    sat_save_dir = os.path.join(args.base_save_dir, f'anomaly_model/{i}_kan_model')
    os.makedirs(sat_save_dir, exist_ok=True)   

    for e in orb_elements_lst:
        print(f'Assessing satellite NORAD ID #: {i}')
        print(sat_df[f'outlier_iqr_{e}'].value_counts())
        time_series, labels = extract_orb_ele(sat_df, e, f'outlier_iqr_{e}')
        
        time_series_null_count = sum(np.isnan(x).sum() for x in time_series)
        print(f"\nTotal nulls in time sereis: {time_series_null_count}")    # Count total nulls in X_train

        dataset = TimeSeriesAnomalyDataset(time_series, labels, window_size=args.window_size, step_size=args.step_size)

        train_loader, val_loader, test_loader, balanced_train_dataset,val_dataset, train_dataset, test_dataset, test_indices = data_loader_build(dataset, args, sat_df, use_smote, e)

        #pos_weight = len([y for y in y_resampled if y == 0]) / len([y for y in y_resampled if y == 1])
        #print(f"Using pos_weight: {pos_weight:.4f}")

        criterion = FocalLoss(alpha=0.25, gamma=2)
      
        model = KAN(in_feat=1, hidden_feat=args.hidden_size, out_feat=1, grid_feat=args.grid_size, num_layers=args.n_layers, use_bias=True, dropout=args.dropout).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=2000, mode='triangular', cycle_momentum=False)

        best_val_f1 = -1
        patience = args.early_stopping
        patience_counter = 0
        optimal_threshold = 0.5  

        model,preds = train_kan_model(args,                       
                                model, 
                                train_loader,
                                best_val_f1, 
                                optimizer, 
                                criterion,
                                scheduler,
                                balanced_train_dataset,
                                val_loader,
                                val_dataset,
                                patience,
                                sat_save_dir,
                                i,
                                e)

        #model.load_state_dict(torch.load(rf"C:\Users\dk412\Desktop\David\Python Projects\RusSat\output\model_training\kan\anomaly_model\{i}_kan_model.pth"))

        all_true_test, all_preds_test, all_probs_test = model_eval(model, 
                                                                    test_loader,
                                                                    args,
                                                                    criterion,
                                                                    optimal_threshold,
                                                                    test_dataset)
        

        test_sample_indices = [dataset.sample_indices[i] for i in test_indices]
        aggregated_preds = aggregate_predictions(
            test_sample_indices, all_preds_test, args.window_size, len(time_series)
        )

        sat_df[f'anom_{e}'] = aggregated_preds


        test_start = min(test_sample_indices)
        test_end = max(test_sample_indices) + args.window_size
        plot_anomalies(time_series, labels, aggregated_preds, i, e, args.base_save_dir, start=test_start, end=test_end)

        plot_metrics(all_true_test, all_probs_test, i, e, args.base_save_dir)
    
    result_anom_df = pd.concat([result_anom_df,sat_df], ignore_index = True)
    result_anom_df['anom_count'] = result_anom_df[['anom_eccentricity', 'anom_mean_anomaly', 'anom_inclination', 'anom_mean_motion', 'anom_ra_of_asc_node', 'anom_arg_of_perigee']].sum(axis=1)

print(result_anom_df.shape)
result_anom_df.to_csv(os.path.join(args.base_save_dir,'result_KAN_anom_detection_training_output.csv'),index=False)

print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DONE')


y_true = (result_anom_df[['outlier_iqr_eccentricity', 'outlier_iqr_mean_anomaly', 'outlier_iqr_inclination', 
                'outlier_iqr_mean_motion', 'outlier_iqr_ra_of_asc_node', 'outlier_iqr_arg_of_perigee']].sum(axis=1) > 0).astype(int)
y_pred = (result_anom_df['anom_count'] > 0).astype(int)

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f" Accuracy={accuracy:.3f}, F1={f1:.3f}")