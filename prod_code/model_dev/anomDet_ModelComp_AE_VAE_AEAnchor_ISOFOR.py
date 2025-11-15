#####################################################################################################################
#                   Prod Code - Model Comparison, Training Data Range, and Latent Space
#####################################################################################################################

import pandas as pd
import sys
sys.path.append(r'C:\Users\dk412\Desktop\David\Python Projects\RusSat\prod_code\model_dev\helpers')
from helpers import *
import numpy as np
import sys
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

model_eval_sats = [41579, 38707, 39727, 40699, 26384, 32399, 29672, 28707, 36358, 37806, 
                   38734, 22594, 44387, 24764, 22671, 35697, 41106, 36401, 39375, 20196]

train_df = pd.read_parquet(r"C:\Users\dk412\Desktop\David\Python Projects\RusSat\data\model_dev\MODEL_training_tle_data_outlier.parquet")

df=train_df[train_df['NORAD_CAT_ID'].isin(model_eval_sats)].reset_index(drop=True)

df_train_5year = df.copy(deep=False)
df_train_4year = df[(df['datetime'] >= '2017-08-24') & (df['datetime'] <= '2021-08-24')]
df_train_3year = df[(df['datetime'] >= '2018-08-24') & (df['datetime'] <= '2021-08-24')]
df_train_2year = df[(df['datetime'] >= '2019-08-24') & (df['datetime'] <= '2021-08-24')]
df_train_1year = df[(df['datetime'] >= '2020-08-24') & (df['datetime'] <= '2021-08-24')]


def run_anomaly_detection_models(df: pd.DataFrame, output_dir: str, latent_dim: int = 4) -> pd.DataFrame:
    ORBITAL_FEATURES = ['inclination', 'ra_of_asc_node', 'eccentricity', 
                       'arg_of_perigee', 'mean_anomaly', 'mean_motion']
    MODEL_COLUMNS = ['ae_anom', 'vae_anom', 'kan_anom', 'isofor_anom']
    
    def process_satellite(sat_id: int, df: pd.DataFrame) -> dict:
        sat_df = (df[df['NORAD_CAT_ID'] == sat_id]
                 .sort_values('datetime', ascending=False)
                 [['datetime'] + ORBITAL_FEATURES]
                 .set_index('datetime'))
        
        for col in sat_df.columns:
            sat_df[col] = pd.to_numeric(sat_df[col], errors='coerce')
        sat_df = sat_df.dropna()
        
        print(f"\nProcessing satellite {sat_id} with latent_dim={latent_dim}")
        
        ae_detector = TLEAnomalyDetector(input_dim=len(ORBITAL_FEATURES), latent_dim=latent_dim)
        vae_detector = TLEVariationalAnomalyDetector(input_dim=len(ORBITAL_FEATURES), latent_dim=latent_dim)
        kan_detector = TLEKanDetector(input_dim=len(ORBITAL_FEATURES), latent_dim=latent_dim)
        iforest_detector = TLEIsolationForest()

        results_dict = {}
        
        #Autoencoder
        _ = ae_detector.train(sat_df.values)
        anomalies, _ = ae_detector.detect_anomalies(sat_df.values)
        results_dict['ae_anom'] = pd.Series(anomalies, index=sat_df.index)
        
        #VAE
        _ = vae_detector.train(sat_df.values)
        anomalies, _ = vae_detector.vae_detect_anomalies(sat_df.values)
        results_dict['vae_anom'] = pd.Series(anomalies, index=sat_df.index)
        
        #KAN
        _ = kan_detector.train(sat_df.values)
        anomalies, _ = kan_detector.kan_detect_anomalies(sat_df.values)
        results_dict['kan_anom'] = pd.Series(anomalies, index=sat_df.index)
        
        #Isolation Forest
        _ = iforest_detector.train(sat_df.values)
        anomalies, _ = iforest_detector.detect_anomalies(sat_df.values)
        results_dict['isofor_anom'] = pd.Series(anomalies, index=sat_df.index)
        
        return results_dict

    for model in MODEL_COLUMNS:
        df[model] = 0
    
    results_by_sat = {
        sat_id: process_satellite(sat_id, df) 
        for sat_id in df['NORAD_CAT_ID'].unique()
    }
    
    for sat_id, results in results_by_sat.items():
        for model in MODEL_COLUMNS:
            mask = (df['NORAD_CAT_ID'] == sat_id) & (df['datetime'].isin(results[model].index))
            df.loc[mask, model] = results[model].reindex(df.loc[mask, 'datetime']).values
    
    return df

def evaluate_models(df: pd.DataFrame) -> dict:
    df_clean = df.copy()
    df_clean['outlier'] = pd.to_numeric(df_clean['outlier'], errors='coerce')
    df_clean = df_clean.dropna(subset=['outlier'])
    df_clean['outlier'] = df_clean['outlier'].astype(int)
    
    model_results = {}  
    
    for model in ['ae_anom', 'vae_anom', 'kan_anom', 'isofor_anom']:
        df_clean[model] = df_clean[model].astype(int)
        #accuracy = df_clean[model].sum()/df_clean['outlier'].sum()
        accuracy = accuracy_score(df_clean['outlier'], df_clean[model])
        print(f"\n{model}:")
        print(f"Accuracy: {accuracy:.3f}")
        
        model_results[model] = {
            'accuracy': accuracy
        }
  
    return model_results

# Main execution
accuracy_comp = []
dfs = [df_train_5year, df_train_4year, df_train_3year, df_train_2year, df_train_1year]
nms = ['5_year', '4_year', '3_year', '2_year', '1_year']
latent_dims = [2, 3, 4]

output_dir = r'C:\Users\dk412\Desktop\David\Python Projects\RusSat\output\comp_plots'

for df, nm in zip(dfs, nms):
    for latent_dim in latent_dims:
        print(f"\nProcessing {nm} dataset with latent_dim={latent_dim}")
        processed_df = run_anomaly_detection_models(df, output_dir, latent_dim)
        processed_df.to_csv(output_dir + rf'\model_comp_results_{nm}_latent{latent_dim}.csv')
        
        model_results = evaluate_models(processed_df)
        model_results['time_period'] = nm
        model_results['latent_dim'] = latent_dim
        accuracy_comp.append(model_results)

flattened = [{
    'ae_anom': d['ae_anom']['accuracy'],
    'vae_anom': d['vae_anom']['accuracy'],
    'kan_anom': d['kan_anom']['accuracy'],
    'isofor_anom': d['isofor_anom']['accuracy'],
    'time_period': d['time_period'],
    'latent_dim': d['latent_dim']
} for d in accuracy_comp]

results_df = pd.DataFrame(flattened)

print("\nFinal Results Summary:")
print(results_df)

results_df.to_csv(output_dir + r'\model_comparison_summary.csv', index=False)