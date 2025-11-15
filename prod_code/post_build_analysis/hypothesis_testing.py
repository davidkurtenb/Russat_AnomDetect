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

combat_df = pd.read_csv("/homes/dkurtenb/projects/russat/output/inference/data/anom_df_ae_anchor_combat_all_sats_0.csv")
combat_df['datetime'] = pd.to_datetime(combat_df['datetime'])

baseline_df = combat_df[(combat_df['datetime']>='02-24-2021') & (combat_df['datetime']<'08-24-2021')]
print(f"Sum of anom_cnt for leadup_df: {baseline_df['anom_count'].sum()}")
print(f"Date Ragne for leadup_df: {baseline_df['datetime'].min()} to {baseline_df['datetime'].max()}")
print("Training Data Read In")


leadup_df = combat_df[(combat_df['datetime']>='08-24-2021') & (combat_df['datetime']<='02-25-2022')]
print(f"Sum of anom_cnt for leadup_df: {leadup_df['anom_count'].sum()}")
print(f"Date Ragne for leadup_df: {leadup_df['datetime'].min()} to {leadup_df['datetime'].max()}")
print("Inf Data Read In")


baseline_period = baseline_df.copy(deep=False)
leadup_period = leadup_df.copy(deep=False)

def get_anomaly_rate(df):
    anom_cols = df.columns.str.startswith('anom')
    anom_data = df.iloc[:, anom_cols]
    #df['anom_count'] = anom_data.sum(axis=1)
    df['anomaly_ind'] = (df['anom_count'] >= 1).astype(int)
    #print(f'Total Anomalous Observations for {df} is {df['anomaly_ind'].sum()}')
    total_observations = len(df)
    anomaly_count = len(df[df['anomaly_ind'] == 1])
    return anomaly_count / total_observations if total_observations > 0 else 0

print("Step 1: Get Anomaly Rates")
baseline_rate = get_anomaly_rate(baseline_period)
leadup_rate = get_anomaly_rate(leadup_period)

from scipy import stats

print("Step 2: Statistical Eval")
def perform_chi_square_test(period1, period2):
    table = [
        [len(period1[period1['anomaly_ind'] == 1]), len(period1[period1['anomaly_ind'] == 0])],
        [len(period2[period2['anomaly_ind'] == 1]), len(period2[period2['anomaly_ind'] == 0])]
    ]

    table = np.array(table) + 0.5    

    chi2, p_value = stats.chi2_contingency(table)[:2]
    return chi2, p_value

chi2_stat, p_value = perform_chi_square_test(baseline_period, leadup_period)

print(f"Chi-square statistic: {chi2_stat:.6f}")
print(f"P-value: {p_value:.10e}")
print(f"Baseline anomaly rate: {baseline_rate:.4f}")
print(f"Lead-up anomaly rate: {leadup_rate:.4f}")


table = [
    [len(baseline_period[baseline_period['anomaly_ind'] == 1]), len(baseline_period[baseline_period['anomaly_ind'] == 0])],
    [len(leadup_period[leadup_period['anomaly_ind'] == 1]), len(leadup_period[leadup_period['anomaly_ind'] == 0])]
]
print("Contingency table:")
print(f"Baseline: Anomalies: {table[0][0]}, No Anomalies: {table[0][1]}")
print(f"Lead-up:  Anomalies: {table[1][0]}, No Anomalies: {table[1][1]}")