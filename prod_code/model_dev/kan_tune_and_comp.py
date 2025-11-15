import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

sys.path.append(r'C:\Users\dk412\Desktop\David\Python Projects\RusSat\prod_code\model_dev\helpers')
from helpers_KANAD import *

class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    
class ImprovedKAN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feat * 15, 64),  # Flatten the window
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(), 
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x.view(x.size(0), -1)).squeeze()

import random
from sklearn.model_selection import ParameterGrid

def improved_hyperparam_tune_kan_model(df, outlier_type, sample_size=1000, orbital_subset=None):

    kan_params = {
        'in_feat': [6],
        'hidden_feat': [32, 64],  
        'out_feat': [1],
        'grid_feat': [25, 50],      
        'num_layers': [2, 3],     
        'epochs': [100],          
        'batch_size': [32],
        'dropout': [0.3],         
        'learning_rate': [0.001, 0.01, 0.1], 
        'window_size': [15],
        'step_size': [8],
        'seed': [42],
        'use_smote': [True]       
    }

    orbital_elements = ['inclination', 'ra_of_asc_node', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 'mean_motion']
    
    if orbital_subset:
        orbital_elements = [el for el in orbital_elements if el in orbital_subset]

    test_sats = df['NORAD_CAT_ID'].unique()
    best_params = {}
    best_score = 0

    full_grid = list(ParameterGrid(kan_params))
    if sample_size < len(full_grid):
        param_grid = random.sample(full_grid, sample_size)
    else:
        param_grid = full_grid

    print(f"Tuning over {len(param_grid)} param sets for {len(test_sats)} satellites "
          f"and {len(orbital_elements)} elements")

    for param_idx, params in enumerate(param_grid, start=1):
        print(f"\n[{param_idx}/{len(param_grid)}] Testing params: {params}")

        scores = []

        for sat_id in test_sats:
            sat_df = df[df['NORAD_CAT_ID'] == sat_id].sort_values('datetime')
            if len(sat_df) < 50:
                continue

            split = int(len(sat_df) * 0.7)
            train = sat_df.iloc[:split]
            X_train = train[orbital_elements].values
            y_train = train[f'outlier_{outlier_type}'].values

            for element in orbital_elements:
                args = Args(
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    seed=params['seed'],
                    learning_rate=params['learning_rate']
                )

                dataset = TimeSeriesAnomalyDataset(
                    X_train, y_train,
                    window_size=params['window_size'],
                    step_size=params['step_size']
                )
                if len(dataset) == 0:
                    continue

                temp_df = train.copy()
                temp_df[f'outlier_{outlier_type}_{element}'] = y_train

                train_loader, val_loader, test_loader, balanced_train_dataset, \
                val_dataset, train_dataset, test_dataset, test_indices = \
                    data_loader_build(dataset, args, temp_df, params['use_smote'], element)

                model = ImprovedKAN(
                    in_feat=params['in_feat'],
                    hidden_feat=params['hidden_feat'],
                    out_feat=params['out_feat'],
                    grid_feat=params['grid_feat'],
                    num_layers=params['num_layers'],
                    dropout=params['dropout']
                ).to(args.device)

                optimizer = torch.optim.AdamW(model.parameters(),
                                              lr=params['learning_rate'],
                                              weight_decay=1e-3)
                #criterion = nn.BCEWithLogitsLoss() CHANGE Dropped this line added crieterion below
                #criterion = FocalLoss(alpha=2, gamma=2)                
                criterion = FocalLoss(alpha=10, gamma=1) # CHANGE alpha/gamma and loss function

                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

                temp_dir = '/tmp/kan_tune'
                os.makedirs(temp_dir, exist_ok=True)

                model, _ = train_kan_model(
                    args=args,
                    model=model,
                    train_loader=train_loader,
                    best_val_f1=-1,
                    optimizer=optimizer,
                    criterion=criterion,
                    scheduler=scheduler,
                    balanced_train_dataset=balanced_train_dataset,
                    val_loader=val_loader,
                    val_dataset=val_dataset,
                    patience=3,  
                    sat_save_dir=temp_dir,
                    i=sat_id,
                    x=element
                )

                all_true_test, all_preds_test, all_probs_test = model_eval(
                    model=model,
                    test_loader=test_loader,
                    args=args,
                    criterion=criterion,
                    optimal_threshold=0.5,
                    test_dataset=test_dataset
                )

                if len(all_true_test) > 0 and len(set(all_true_test)) > 1:
                    score = f1_score(all_true_test, all_preds_test,
                                     average='weighted', zero_division=0)
                    scores.append(score)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if scores:
            avg_score = np.mean(scores)
            print(f"Avg F1 score: {avg_score:.4f}")
            if avg_score > best_score:
                best_score = avg_score
                best_params['kan'] = params
                print(f"New best params with F1={best_score:.4f}")

    print("\nBest parameters found:")
    print(best_params)
    return best_params


def run_improved_kan_comparison_with_best_params(df, best_params,outlier_type):
    ORBITAL_FEATURES = ['inclination', 'ra_of_asc_node', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 'mean_motion']
    
    for element in ORBITAL_FEATURES:
        df[f'kan_anom_{element}'] = 0
    
    if 'kan' not in best_params:
        print("No KAN parameters found, skipping KAN evaluation")
        return df
    
    kan_params = best_params['kan']
    
    for sat_id in df['NORAD_CAT_ID'].unique():
        print(f"Processing satellite {sat_id} with Improved KAN model")
        
        sat_df = (df[df['NORAD_CAT_ID'] == sat_id]
                 .sort_values('datetime')[['datetime'] + ORBITAL_FEATURES + [f'outlier_{outlier_type}']]
                 .set_index('datetime'))
        
        for col in ORBITAL_FEATURES:
            sat_df[col] = pd.to_numeric(sat_df[col], errors='coerce')
        sat_df = sat_df.dropna()
        
        if len(sat_df) < kan_params['window_size'] * 2:
            continue
        
        data = sat_df[ORBITAL_FEATURES].values
        labels = sat_df[f'outlier_{outlier_type}'].values
        
        for element in ORBITAL_FEATURES:
            try:
                args = Args(
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    epochs=kan_params['epochs'],
                    batch_size=kan_params['batch_size'],
                    seed=kan_params['seed'],
                    learning_rate=kan_params['learning_rate']
                )
                
                split_idx = int(len(data) * 0.7)
                X_train = data[:split_idx]
                y_train = labels[:split_idx]
                
                train_dataset = TimeSeriesAnomalyDataset(
                    X_train, y_train,
                    window_size=kan_params['window_size'],
                    step_size=kan_params['step_size']
                )
                
                if len(train_dataset) == 0:
                    continue
                
                temp_df = sat_df.reset_index().iloc[:split_idx].copy()
                temp_df[f'outlier_{outlier_type}_{element}'] = y_train
                
                train_loader, val_loader, test_loader, balanced_train_dataset, val_dataset, train_dataset_subset, test_dataset, test_indices = data_loader_build(
                    train_dataset, args, temp_df, kan_params['use_smote'], element
                )
                
                model = ImprovedKAN(
                    in_feat=kan_params['in_feat'],
                    hidden_feat=kan_params['hidden_feat'],
                    out_feat=kan_params['out_feat'],
                    grid_feat=kan_params['grid_feat'],
                    num_layers=kan_params['num_layers'],
                    dropout=kan_params['dropout']
                ).to(args.device)
                
                optimizer = torch.optim.AdamW(model.parameters(), lr=kan_params['learning_rate'], weight_decay=1e-4)
                #criterion = nn.BCEWithLogitsLoss()
                #criterion = FocalLoss(alpha=2, gamma=2)
                criterion = FocalLoss(alpha=10, gamma=1) # CHANGE alpha/gamma and loss function
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
                
                temp_dir = '/tmp/kan_eval'
                os.makedirs(temp_dir, exist_ok=True)
                
                model, _ = train_kan_model(
                    args=args,
                    model=model,
                    train_loader=train_loader,
                    best_val_f1=0.0,
                    optimizer=optimizer,
                    criterion=criterion,
                    scheduler=scheduler,
                    balanced_train_dataset=balanced_train_dataset,
                    val_loader=val_loader,
                    val_dataset=val_dataset,
                    patience=5,
                    sat_save_dir=temp_dir,
                    i=sat_id,
                    x=element
                )
                
                all_true_test, all_preds_test, all_probs_test = model_eval(
                    model=model,
                    test_loader=test_loader,
                    args=args,
                    criterion=criterion,
                    optimal_threshold=0.5,
                    test_dataset=test_dataset
                )
                
                if len(all_preds_test) > 0:
                    test_start_idx = split_idx
                    test_end_idx = min(test_start_idx + len(all_preds_test), len(sat_df))
                    test_timestamps = sat_df.index[test_start_idx:test_end_idx]
                    
                    mask = (df['NORAD_CAT_ID'] == sat_id) & (df['datetime'].isin(test_timestamps))
                    preds_to_assign = all_preds_test[:mask.sum()]
                    df.loc[mask, f'kan_anom_{element}'] = preds_to_assign
            
            except Exception as e:
                print(f"Error processing satellite {sat_id}, element {element}: {e}")
                continue
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return df

def evaluate_improved_kan_model_performance(results_df, time_period, outlier_type, save_path=None):

    orbital_elements = ['inclination', 'ra_of_asc_node', 'eccentricity',
                        'arg_of_perigee', 'mean_anomaly', 'mean_motion']
    model_cols = [f'kan_anom_{element}' for element in orbital_elements]

    for col in model_cols:
        if col not in results_df.columns:
            results_df[col] = 0
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0).astype(int)

    results_df['anom'] = results_df[model_cols].max(axis=1)

    results_df = results_df.dropna(subset=[f'outlier_{outlier_type}', 'anom'])
    results_df[f'outlier_{outlier_type}'] = results_df[f'outlier_{outlier_type}'].astype(int)

    if len(results_df) == 0:
        print(f"[{time_period}] No valid data to evaluate.")
        metrics = {
            'time_period': time_period,
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1_score': None
        }
    else:
        accuracy = accuracy_score(results_df[f'outlier_{outlier_type}'], results_df['anom'])
        precision = precision_score(results_df[f'outlier_{outlier_type}'], results_df['anom'], zero_division=0)
        recall = recall_score(results_df[f'outlier_{outlier_type}'], results_df['anom'], zero_division=0)
        f1 = f1_score(results_df[f'outlier_{outlier_type}'], results_df['anom'], zero_division=0)

        metrics = {
            'time_period': time_period,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        print(f"[{time_period}] Aggregated: "
              f"Acc={accuracy:.3f}, F1={f1:.3f}, "
              f"Precision={precision:.3f}, Recall={recall:.3f}")

    if save_path:
        pd.DataFrame([metrics]).to_csv(save_path, index=False)

    return metrics

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model_eval_sats = [41579, 26384, 36358, 37806]
    #parquet_file = "/homes/dkurtenb/projects/russat/data/MODEL_training_tle_data_all_outlier.parquet"
    parquet_file = "/homes/dkurtenb/projects/russat/data/MODEL_trainig_tle_data_all_outlier.parquet"
    df = pd.read_parquet(parquet_file, filters=[('NORAD_CAT_ID', 'in', model_eval_sats)])
    outlier_type = 'iqr'
    #outlier_type = 'mad'
    #outlier_type = 'zscore'
    
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n\tStarting KAN hyperparameter tuning with {outlier_type}\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ")
    best_params = improved_hyperparam_tune_kan_model(df,outlier_type)
    
    print("\n" + "="*80)
    print("STARTING TIME PERIOD EVALUATION WITH IMPROVED KAN MODEL")
    print("="*80)
    
    df_train_5year = df.copy(deep=False)
    df_train_4year = df[(df['datetime'] >= '2017-08-24') & (df['datetime'] <= '2021-08-24')]
    df_train_3year = df[(df['datetime'] >= '2018-08-24') & (df['datetime'] <= '2021-08-24')]
    df_train_2year = df[(df['datetime'] >= '2019-08-24') & (df['datetime'] <= '2021-08-24')]
    df_train_1year = df[(df['datetime'] >= '2020-08-24') & (df['datetime'] <= '2021-08-24')]
    
    dfs = [df_train_5year, df_train_4year, df_train_3year, df_train_2year, df_train_1year]
    nms = ['5_year', '4_year', '3_year', '2_year', '1_year']
    

    accuracy_comp = []

    for df_period, nm in zip(dfs, nms):
        print(f"\nProcessing {nm} dataset with Improved KAN ***********************************")

        results_df = run_improved_kan_comparison_with_best_params(df_period.copy(), best_params, outlier_type)
        results_path = f'/homes/dkurtenb/projects/russat/output/improved_kan_model_results_{nm}.csv'
        results_df.to_csv(results_path, index=False)

        metrics_save_path = f'/homes/dkurtenb/projects/russat/output/improved_kan_model_aggregated_metrics_{nm}.csv'
        model_results = evaluate_improved_kan_model_performance(results_df, nm, outlier_type, save_path=metrics_save_path)
        accuracy_comp.append(model_results)

    summary_df = pd.DataFrame(accuracy_comp)
    summary_df.to_csv('/homes/dkurtenb/projects/russat/output/improved_kan_model_aggregated_summary.csv', index=False)

    print("\nAggregated Summary Results:")
    print(summary_df)
