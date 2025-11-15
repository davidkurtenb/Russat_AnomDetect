#############################################################################
#                       PROD CODE - Tune & Comp
#############################################################################
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score
import sys
sys.path.append('/homes/dkurtenb/projects/russat/prod_code/archive')
from helpers import *
import pandas as pd
import numpy as np
import torch
import os

def hyperparam_tune_models(df):
    
    ae_params = {'latent_dim': [2, 3, 4, 5],
                    'epochs': [100, 150, 200], 
                    'batch_size': [16, 32, 64],
                    'threshold_sigma': [1.5, 2.0, 2.5]}
                
    vae_params = {'hidden_dim': [16, 32, 64],
        'latent_dim': [2, 3, 4, 5],
        'epochs': [100, 150, 200],
        'batch_size': [16, 32, 64], 
        'threshold_sigma': [1.5, 2.0, 2.5]}
    
    aeanchor_params = {'hidden_dim': [16, 32, 64],
        'latent_dim': [2, 3, 4, 5],
        'epochs': [100, 150, 200],
        'batch_size': [16, 32, 64],
        'threshold_sigma': [1.5, 2.0, 2.5]}
    
    
    if_params = {'contamination': [0.05, 0.1, 0.15, 'auto'],
        'n_estimators': [50, 100, 200],
        'max_samples': [0.5, 0.8, 1.0]}
    
    
    test_sats = df['NORAD_CAT_ID'].unique()
    #test_sats = df['NORAD_CAT_ID'].unique()[:3]
    
    best_params = {}
    
    print("Tuning Autoencoder***************************")
    best_score = 0
    for params in ParameterGrid(ae_params):
        scores = []
        for sat_id in test_sats:
            sat_df = df[df['NORAD_CAT_ID'] == sat_id].sort_values('datetime')
            if len(sat_df) < 50:
                continue
                
            split = int(len(sat_df) * 0.7)
            train = sat_df.iloc[:split]
            test = sat_df.iloc[split:]
            
            X_train = train[['inclination', 'ra_of_asc_node', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 'mean_motion']].values
            X_test = test[['inclination', 'ra_of_asc_node', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 'mean_motion']].values
            y_test = test['outlier'].values
            
            detector = TLEAnomalyDetector(input_dim=6, latent_dim=params['latent_dim'])
            detector.train(X_train, epochs=params['epochs'], batch_size=params['batch_size'])
            anomalies, _ = detector.detect_anomalies(X_test, threshold_sigma=params['threshold_sigma'])
            score = f1_score(y_test, anomalies.astype(int), average='weighted', zero_division=0)
            scores.append(score)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params['autoencoder'] = params
    
    print("Tuning VAE***************************")
    best_score = 0
    for params in ParameterGrid(vae_params):
        scores = []
        for sat_id in test_sats:
            sat_df = df[df['NORAD_CAT_ID'] == sat_id].sort_values('datetime')
            if len(sat_df) < 50:
                continue
                
            split = int(len(sat_df) * 0.7)
            train = sat_df.iloc[:split]
            test = sat_df.iloc[split:]
            
            X_train = train[['inclination', 'ra_of_asc_node', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 'mean_motion']].values
            X_test = test[['inclination', 'ra_of_asc_node', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 'mean_motion']].values
            y_test = test['outlier'].values
            
            detector = TLEVariationalAnomalyDetector(input_dim=6, hidden_dim=params['hidden_dim'], latent_dim=params['latent_dim'])
            detector.train(X_train, epochs=params['epochs'], batch_size=params['batch_size'])
            anomalies, _ = detector.vae_detect_anomalies(X_test, threshold_sigma=params['threshold_sigma'])
            score = f1_score(y_test, anomalies.astype(int), average='weighted', zero_division=0)
            scores.append(score)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params['vae'] = params
    
    print("Tuning AE Anchor***************************")
    best_score = 0
    for params in ParameterGrid(aeanchor_params):
        scores = []
        for sat_id in test_sats:
            sat_df = df[df['NORAD_CAT_ID'] == sat_id].sort_values('datetime')
            if len(sat_df) < 50:
                continue
                
            split = int(len(sat_df) * 0.7)
            train = sat_df.iloc[:split]
            test = sat_df.iloc[split:]
            
            X_train = train[['inclination', 'ra_of_asc_node', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 'mean_motion']].values
            X_test = test[['inclination', 'ra_of_asc_node', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 'mean_motion']].values
            y_test = test['outlier'].values
            
            detector = TLEAEAnchorDetector(input_dim=6, hidden_dim=params['hidden_dim'], latent_dim=params['latent_dim'])
            detector.train(X_train, epochs=params['epochs'], batch_size=params['batch_size'])
            anomalies, _ = detector.aeanchor_detect_anomalies(X_test, threshold_sigma=params['threshold_sigma'])
            score = f1_score(y_test, anomalies.astype(int), average='weighted', zero_division=0)
            scores.append(score)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params['aeanchor'] = params
    
    print("Tuning IsoFor***************************")
    best_score = 0
    for params in ParameterGrid(if_params):
        scores = []
        for sat_id in test_sats:
            sat_df = df[df['NORAD_CAT_ID'] == sat_id].sort_values('datetime')
            if len(sat_df) < 50:
                continue
                
            split = int(len(sat_df) * 0.7)
            train = sat_df.iloc[:split]
            test = sat_df.iloc[split:]
            
            X_train = train[['inclination', 'ra_of_asc_node', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 'mean_motion']].values
            X_test = test[['inclination', 'ra_of_asc_node', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 'mean_motion']].values
            y_test = test['outlier'].values
            
        detector = TLEIsolationForest(
            contamination=params['contamination'],
            n_estimators=params['n_estimators'],
            max_samples=params['max_samples'])
        
        detector.train(X_train)
        anomalies, _ = detector.detect_anomalies(X_test)
        score = f1_score(y_test, anomalies.astype(int), average='weighted', zero_division=0)
        scores.append(score)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params['isolation_forest'] = params

    print("Best parameters found:")
    for model, params in best_params.items():
        print(f"{model}: {params}")
    
    return best_params

def run_comparison_with_best_params(df, best_params):
    
    ORBITAL_FEATURES = ['inclination', 'ra_of_asc_node', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 'mean_motion']
    
    df['ae_anom'] = 0
    df['vae_anom'] = 0
    df['aeanchor_anom'] = 0
    df['isofor_anom'] = 0
    
    for sat_id in df['NORAD_CAT_ID'].unique():
        print(f"Processing satellite {sat_id}")
        
        sat_df = (df[df['NORAD_CAT_ID'] == sat_id]
                 .sort_values('datetime')[['datetime'] + ORBITAL_FEATURES]
                 .set_index('datetime'))
        
        for col in ORBITAL_FEATURES:
            sat_df[col] = pd.to_numeric(sat_df[col], errors='coerce')
        sat_df = sat_df.dropna()
        
        if len(sat_df) < 10:
            continue
        
        data = sat_df.values
        
        if 'autoencoder' in best_params:
            p = best_params['autoencoder']
            detector = TLEAnomalyDetector(input_dim=6, latent_dim=p['latent_dim'])
            detector.train(data, epochs=p['epochs'], batch_size=p['batch_size'])
            anomalies, _ = detector.detect_anomalies(data, threshold_sigma=p['threshold_sigma'])
            mask = (df['NORAD_CAT_ID'] == sat_id) & (df['datetime'].isin(sat_df.index))
            df.loc[mask, 'ae_anom'] = pd.Series(anomalies, index=sat_df.index).reindex(df.loc[mask, 'datetime']).values
        
        if 'vae' in best_params:
            p = best_params['vae']
            detector = TLEVariationalAnomalyDetector(input_dim=6, hidden_dim=p['hidden_dim'], latent_dim=p['latent_dim'])
            detector.train(data, epochs=p['epochs'], batch_size=p['batch_size'])
            anomalies, _ = detector.vae_detect_anomalies(data, threshold_sigma=p['threshold_sigma'])
            mask = (df['NORAD_CAT_ID'] == sat_id) & (df['datetime'].isin(sat_df.index))
            df.loc[mask, 'vae_anom'] = pd.Series(anomalies, index=sat_df.index).reindex(df.loc[mask, 'datetime']).values
        
        if 'aeanchor' in best_params:
            p = best_params['aeanchor']
            detector = TLEAEAnchorDetector(input_dim=6, hidden_dim=p['hidden_dim'], latent_dim=p['latent_dim'])
            detector.train(data, epochs=p['epochs'], batch_size=p['batch_size'])
            anomalies, _ = detector.aeanchor_detect_anomalies(data, threshold_sigma=p['threshold_sigma'])
            mask = (df['NORAD_CAT_ID'] == sat_id) & (df['datetime'].isin(sat_df.index))
            df.loc[mask, 'aeanchor_anom'] = pd.Series(anomalies, index=sat_df.index).reindex(df.loc[mask, 'datetime']).values
        
        if 'isolation_forest' in best_params:
            p = best_params['isolation_forest']
            detector = TLEIsolationForest(contamination=p['contamination'],n_estimators=p['n_estimators'],max_samples=p['max_samples'])
            detector.train(data)
            anomalies, _ = detector.detect_anomalies(data)
            mask = (df['NORAD_CAT_ID'] == sat_id) & (df['datetime'].isin(sat_df.index))
            df.loc[mask, 'isofor_anom'] = pd.Series(anomalies, index=sat_df.index).reindex(df.loc[mask, 'datetime']).values
    
    return df

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    #train_df = pd.read_parquet("/homes/dkurtenb/projects/russat/data/MODEL_training_tle_data_outlier.parquet")
    #model_eval_sats = [41579, 38707, 39727, 40699, 26384, 32399, 29672, 28707, 36358, 37806]
    model_eval_sats = [41579, 26384, 36358, 37806]
    #df = train_df[train_df['NORAD_CAT_ID'].isin(model_eval_sats)].reset_index(drop=True)
    
    parquet_file = "/homes/dkurtenb/projects/russat/data/MODEL_training_tle_data_outlier.parquet"
    df = pd.read_parquet(parquet_file,filters=[('NORAD_CAT_ID', 'in', model_eval_sats)])

    best_params = hyperparam_tune_models(df)
    
    df_train_5year = df.copy(deep=False)
    df_train_4year = df[(df['datetime'] >= '2017-08-24') & (df['datetime'] <= '2021-08-24')]
    df_train_3year = df[(df['datetime'] >= '2018-08-24') & (df['datetime'] <= '2021-08-24')]
    df_train_2year = df[(df['datetime'] >= '2019-08-24') & (df['datetime'] <= '2021-08-24')]
    df_train_1year = df[(df['datetime'] >= '2020-08-24') & (df['datetime'] <= '2021-08-24')]
    
    dfs = [df_train_5year, df_train_4year, df_train_3year, df_train_2year, df_train_1year]
    nms = ['5_year', '4_year', '3_year', '2_year', '1_year']
    #latent_dims = [2, 3, 4]  
    
    accuracy_comp = []
    
    for df_period, nm in zip(dfs, nms):
        #for latent_dim in latent_dims:
        print(f"\nProcessing {nm} dataset ***********************************")
        
        tuned_params = best_params.copy()
        for model in ['autoencoder', 'vae', 'aeanchor']:
            if model in tuned_params:
                tuned_params[model] = tuned_params[model].copy()
                #tuned_params[model]['latent_dim'] = latent_dim
        
        results_df = run_comparison_with_best_params(df_period.copy(), tuned_params)
        results_df.to_csv(f'/homes/dkurtenb/projects/russat/output//uned_model_results_{nm}.csv', index=False)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        model_results = {}
        models = ['ae_anom', 'vae_anom', 'aeanchor_anom', 'isofor_anom']
        
        for model in models:
            clean_df = results_df.dropna(subset=['outlier', model])
            if len(clean_df) > 0:
                clean_df['outlier'] = clean_df['outlier'].astype(int)
                clean_df[model] = clean_df[model].astype(int)
                
                accuracy = accuracy_score(clean_df['outlier'], clean_df[model])
                precision = precision_score(clean_df['outlier'], clean_df[model], average='weighted', zero_division=0)
                recall = recall_score(clean_df['outlier'], clean_df[model], average='weighted', zero_division=0)
                f1 = f1_score(clean_df['outlier'], clean_df[model], average='weighted', zero_division=0)
                
                model_results[model] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                print(f"{model}: Acc={accuracy:.3f}, F1={f1:.3f}")
        
        model_results['time_period'] = nm
        #model_results['latent_dim'] = latent_dim
        accuracy_comp.append(model_results)

    flattened = []
    for d in accuracy_comp:
        row = {
            'time_period': d['time_period'],
            'latent_dim': d['latent_dim']
        }
        for model in ['ae_anom', 'vae_anom', 'aeanchor_anom', 'isofor_anom']:
            if model in d:
                row[f'{model}_accuracy'] = d[model]['accuracy']
                row[f'{model}_f1'] = d[model]['f1_score']
        flattened.append(row)
    
    results_df = pd.DataFrame(flattened)
    results_df.to_csv('/homes/dkurtenb/projects/russat/output/tuned_model_comparison_summary.csv', index=False)
    
    print("\nFinal Results Summary:")
    print(results_df)
    print(f"\nBest parameters used as base: {best_params}")