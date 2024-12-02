import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import os
import pickle
from multiprocessing import Pool

import warnings
warnings.filterwarnings('ignore')

inf_df = pd.read_parquet("/homes/dkurtenb/projects/russat/output/anom_df_inference_all_sats_0.parquet")
train_df = pd.read_parquet("/homes/dkurtenb/projects/russat/output/anomaly_output_data_TRAIN_all_sats.parquet")
#combat_df = pd.read_parquet(r'C:\Users\dk412\Desktop\David\Python Projects\RusSat\results\anom_df_combat_all_sats_0.parquet')

baseline_period = train_df.copy(deep=False)
leadup_period = inf_df.copy(deep=False)

def get_anomaly_rate(df):
    anom_cols = df.columns.str.startswith('anom')
    anom_data = df.iloc[:, anom_cols]
    df['anom_count'] = anom_data.sum(axis=1)
    df['anomaly_ind'] = (df['anom_count'] >= 1).astype(int)
    total_observations = len(df)
    anomaly_count = len(df[df['anomaly_ind'] == 1])
    return anomaly_count / total_observations if total_observations > 0 else 0

baseline_rate = get_anomaly_rate(baseline_period)
leadup_rate = get_anomaly_rate(leadup_period)

from scipy import stats

def perform_chi_square_test(period1, period2):
    table = [
        [len(period1[period1['anomaly_ind'] == 1]), len(period1[period1['anomaly_ind'] == 0])],
        [len(period2[period2['anomaly_ind'] == 1]), len(period2[period2['anomaly_ind'] == 0])]
    ]
    
    chi2, p_value = stats.chi2_contingency(table)[:2]
    return chi2, p_value

chi2_stat, p_value = perform_chi_square_test(baseline_period, leadup_period)

print(f"Chi-square statistic: {chi2_stat:.6f}")
print(f"P-value: {p_value:.6f}")
print(f"Baseline anomaly rate: {baseline_rate:.4f}")
print(f"Lead-up anomaly rate: {leadup_rate:.4f}")