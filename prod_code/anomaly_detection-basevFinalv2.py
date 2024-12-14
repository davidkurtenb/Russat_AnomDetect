#!/usr/bin/env python
# coding: utf-8

# # Applying Deep Learning to Pattern Mining of Sequential Data and Anomaly Detection of Russian Satellite Activity Prior to Military Action 
# 
# By: David Kurtenbach, Megan Manly, and Zach Metzinger

# ## Section 1: Load and Filter the Data to relevant Russian Satellites

# In[23]:


import pandas as pd
import pickle
import matplotlib.pyplot as plt
import concurrent.futures
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from datetime import timedelta
import seaborn as sns
from joblib import Parallel, delayed

#Cluster Method
import optuna
from sklearn.cluster import OPTICS
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist
from scipy.stats import chi2_contingency, median_abs_deviation
from sklearn.decomposition import PCA

#RAE with Clustering Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


# In[3]:


pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)  


# In[4]:


df_dev = pd.read_parquet('DEV_training_tle_df.parquet')
df_inf = pd.read_parquet('DEV_inference_tle_df.parquet')
df_com = pd.read_parquet('DEV_combat_tle_df.parquet')

print(df_dev.shape)
print(df_inf.shape)
print(df_com.shape)


# In[5]:


print(df_dev.info())
print(df_inf.info())
print(df_com.info())


# In[6]:


# Join the dataframes to create a single dataframe excluding the caresian_cord due to it being in a list or array
combined_df = pd.concat([df_dev.drop(columns='cartesian_cords'), 
                         df_inf.drop(columns='cartesian_cords'),
                         df_com.drop(columns='cartesian_cords')], 
                        ignore_index=True).drop_duplicates()
combined_df.info()


# In[7]:


# Find the earliest and latest datetime
earliest_datetime = combined_df['datetime'].min()
latest_datetime = combined_df['datetime'].max()

print(f"Date Range: {earliest_datetime} to {latest_datetime}")


# **Feilds within the data:**
# 
# 1. line1 and line2: These are the two lines of the satellite’s Two-Line Element (TLE) set, a standard format representing the satellite’s orbital parameters. TLEs are commonly used to track satellites' positions and trajectories.
# 
# 2. catalog_number: A unique identification number for each satellite in the catalog, often referred to as the NORAD catalog number.
# 
# 3. classification: Specifies the classification of the satellite, such as "U" for unclassified, indicating the public availability or restricted nature of the satellite's data.
# 
# 4. launch_year: The year the satellite was launched, given as a string.
# 
# 5. launch_number: The number assigned to the launch event for that year, identifying the specific launch sequence.
# 
# 6. launch_piece: This indicates the particular piece of the launch, as multiple objects are often deployed from a single launch.
# 
# 7. epoch_year: The year for which the TLE data is valid, indicating the reference time for the satellite's orbital data.
# 
# 8. epoch_day: The day of the year (with decimals) representing the precise time of the TLE epoch.
# 
# 9. mean_motion_dot: The rate of change of mean motion, indicating how the satellite's speed is increasing or decreasing over time, often affected by atmospheric drag.
# 
# 10. mean_motion_ddot: The second derivative of mean motion, showing changes in the satellite’s acceleration due to factors like gravitational perturbations.
# 
# 11. bstar: The drag term, accounting for atmospheric drag on the satellite. It’s an estimate of the drag-related decay.
# 
# 12. ephemeris_type: An integer representing the type of ephemeris model used, typically used in calculations by specific orbital propagation models.
# 
# 13. element_number: This number increments with each update to the TLE, serving as a version identifier for the TLE data.
# 
# 14. satellite_number: A unique identifier for the satellite, often matching the catalog number.
# 
# 15. inclination: The angle of the satellite’s orbit relative to Earth’s equatorial plane, measured in degrees.
# 
# 16. ra_of_asc_node: The right ascension of the ascending node (RAAN), showing the orientation of the orbit in relation to Earth's equator.
# 
# 17. eccentricity: The shape of the orbit, where 0 indicates a circular orbit and values close to 1 indicate more elliptical orbits.
# 
# 18. arg_of_perigee: The angle within the orbital plane between the ascending node and the orbit's closest point to Earth (perigee).
# 
# 19. mean_anomaly: An angle that represents the satellite’s position along its orbit at the epoch, assuming a uniform circular motion.
# 
# 20. mean_motion: The number of orbits the satellite completes per day, directly related to its altitude.
# 
# 21. rev_at_epoch: The satellite’s revolution count at the time of the epoch, marking how many times it has orbited Earth since launch.
# 
# 22. datetime: The datetime of the epoch, specifying the exact time for the provided TLE data.
# 
# 23. x, y, z: Cartesian coordinates representing the satellite’s position in space relative to Earth, typically in kilometers.
# 
# 24. cartesian_cords: Likely represents the x, y, and z coordinates as a combined object or structure, providing the satellite’s position in a single field.

# In[8]:


# Load the CIS_satcat.pkl dataset
satcat_path = 'CIS_satcat.pkl'

with open(satcat_path, 'rb') as f:
    satcat = pickle.load(f)

# Convert the satcat data into a DataFrame for easy manipulation
satcat_df = pd.DataFrame(satcat)

# Load the UCS-Satellite-Database 5-1-2023.xlsx dataset
file_path_ucs = 'UCS-Satellite-Database 5-1-2023.xlsx'
ucs_df = pd.read_excel(file_path_ucs)

# Ensure relevant ID fields are of the same type for merging
satcat_df['NORAD_CAT_ID'] = satcat_df['NORAD_CAT_ID'].astype(int)
combined_df['catalog_number'] = combined_df['catalog_number'].astype(int)
ucs_df['NORAD Number'] = ucs_df['NORAD Number'].astype(int)

# Merge combined_df with additional data from satcat_df and ucs_df
# We only bring in relevant columns to avoid redundant data
merged_df = pd.merge(combined_df, satcat_df[['NORAD_CAT_ID', 'COUNTRY', 'OBJECT_TYPE']], 
                     how='left', left_on='catalog_number', right_on='NORAD_CAT_ID')
merged_df = pd.merge(merged_df, 
                     ucs_df[['NORAD Number', 
                             'Country of Operator/Owner', 
                             'Users', 
                             'Purpose', 
                             'Date of Launch', 
                             'Class of Orbit', 
                             ]],
                     how='left', left_on='catalog_number', right_on='NORAD Number')

# Convert 'Date of Launch' to datetime
merged_df['Date of Launch'] = pd.to_datetime(merged_df['Date of Launch'], errors='coerce')

# Filter the merged DataFrame based on criteria
filtered_df = merged_df[
    (merged_df['COUNTRY'] == 'CIS') & 
    (merged_df['OBJECT_TYPE'] == 'PAYLOAD') &
    (merged_df['Country of Operator/Owner'] == 'Russia')
]

# Remove duplicates to get unique satellite entries based on the criteria
filtered_df = filtered_df.drop_duplicates()

# Display the filtered data to ensure it worked
filtered_df.head()


# In[9]:


filtered_df.info()


# In[10]:


# How many unique satellite numbers there are
filtered_df['catalog_number'].value_counts().count()


# In[11]:


filtered_df['Country of Operator/Owner'].value_counts()


# In[12]:


# Get unique catalog numbers
unique_catalog_numbers = filtered_df['catalog_number'].unique()

# Convert to a DataFrame and export to CSV
pd.DataFrame(unique_catalog_numbers, columns=['catalog_number']).to_csv('CIS_sat_num.csv', index=False)


# ## Regular Analytics Method

# In[15]:


def calculate_learning_baseline(data, features, baseline_start, baseline_end, rate_of_change_multiplier=2.0, fallback_days=360):
    
    baseline_stats = {}
    rate_of_change_thresholds = {}

    def calculate_rate_of_change(group, feature, time_column='datetime'):
        delta_feature = group[feature].diff().abs()
        delta_time = group[time_column].diff().dt.total_seconds() / (3600 * 24)  # Convert to days
        delta_time[delta_time == 0] = 1e-9  # Avoid division by zero
        return delta_feature / delta_time

    grouped = data.groupby('satellite_number')
    for sat_no, sat_group in grouped:
        sat_group = sat_group.sort_values('datetime')

        # Determine the learning period
        learning_data = sat_group[
            (sat_group['datetime'] >= baseline_start) & (sat_group['datetime'] <= baseline_end)
        ]

        if learning_data.empty:
            print(f"No data for Satellite {sat_no} in the original learning period. Using fallback period.")
            fallback_end = sat_group['datetime'].min() + pd.Timedelta(days=fallback_days)
            learning_data = sat_group[sat_group['datetime'] <= fallback_end]

        baseline_stats[sat_no] = {}
        rate_of_change_thresholds[sat_no] = {}

        for feature in features:
            if feature in learning_data.columns:
                baseline_stats[sat_no][feature] = {
                    'mean': learning_data[feature].mean(),
                    'std': learning_data[feature].std()
                }

                # Calculate rate of change during the learning period
                learning_data_sorted = learning_data.sort_values('datetime')
                rate_of_change = calculate_rate_of_change(learning_data_sorted, feature)
                rate_of_change_thresholds[sat_no][feature] = rate_of_change.mean() + rate_of_change_multiplier * rate_of_change.std()

    return baseline_stats, rate_of_change_thresholds

def detect_and_visualize_anomalies_hybrid(data, baseline_stats, rate_of_change_thresholds, window_size=7, zscore_threshold=2.0, percentile_threshold=0.99):
    
    def safe_diff(series):
        """Handles differences safely to avoid invalid or zero values."""
        diff = series.diff()
        diff[diff == 0] = 1e-9  # Avoid division by zero
        return diff

    def calculate_rate_of_change(group, feature, time_column='datetime'):
        delta_feature = safe_diff(group[feature].abs())
        delta_time = group[time_column].diff().dt.total_seconds() / (3600 * 24)  # Convert to days
        delta_time[delta_time == 0] = 1e-9  # Avoid division by zero
        return delta_feature / delta_time

    features_to_plot = ['inclination', 'mean_motion', 'eccentricity', 
                        'ra_of_asc_node', 'arg_of_perigee', 'mean_anomaly']

    data.loc[:, 'statistical_thresholding_anomaly'] = False

    grouped = data.groupby('satellite_number')
    all_anomalies = []

    for sat_no, sat_group in grouped:
        print(f"Processing satellite {sat_no}")

        sat_group = sat_group.sort_values('datetime')
        sat_baseline_stats = baseline_stats.get(sat_no, {})
        sat_rate_of_change_thresholds = rate_of_change_thresholds.get(sat_no, {})

        for feature in features_to_plot:
            if feature in sat_group.columns:
                sat_group[f'{feature}_smoothed'] = (
                    sat_group[feature]
                    .rolling(window=window_size, min_periods=1)
                    .mean()
                    .fillna(0)
                )

        for feature in features_to_plot:
            if feature in sat_group.columns:
                sat_group[f'{feature}_rate_of_change'] = calculate_rate_of_change(sat_group, f'{feature}_smoothed')

        for feature in features_to_plot:
            if f'{feature}_rate_of_change' not in sat_group.columns:
                continue

            baseline_mean = sat_baseline_stats.get(feature, {}).get('mean', 0)
            baseline_std = max(sat_baseline_stats.get(feature, {}).get('std', 1e-5), 1e-5)
            feature_rate_threshold = sat_rate_of_change_thresholds.get(feature, 0.1)

            sat_group[f'{feature}_zscore'] = (sat_group[f'{feature}_rate_of_change'] - baseline_mean) / baseline_std

            combined_anomalies = sat_group[
                (sat_group[f'{feature}_zscore'].abs() > zscore_threshold) &
                (sat_group[f'{feature}_rate_of_change'] > feature_rate_threshold)
            ]

            data.loc[combined_anomalies.index, 'statistical_thresholding_anomaly'] = True

            if not combined_anomalies.empty:
                all_anomalies.append(combined_anomalies)

        fig, axes = plt.subplots(len(features_to_plot), 1, figsize=(12, 18), sharex=True)
        fig.suptitle(f"Anomalies for Satellite {sat_no} - Potential Maneuvers")

        for idx, feature in enumerate(features_to_plot):
            if feature not in sat_group.columns:
                continue

            axes[idx].plot(sat_group['datetime'], sat_group[f'{feature}_smoothed'], label=f'Smoothed {feature}', color='blue')
            anomaly_points = combined_anomalies['datetime'][combined_anomalies[f'{feature}_smoothed'].notna()]
            anomaly_values = combined_anomalies[f'{feature}_smoothed'][combined_anomalies[f'{feature}_smoothed'].notna()]
            axes[idx].scatter(anomaly_points, anomaly_values, color='red', label=f'Anomalous {feature}', s=5)
            axes[idx].set_ylabel(f'{feature}')
            axes[idx].legend()

        plt.xlabel('Datetime (UTC)')
        plt.show()

    if all_anomalies:
        combined_anomalies = pd.concat(all_anomalies, axis=0)
        print(f"Total anomalies detected: {combined_anomalies.shape[0]}")
    else:
        combined_anomalies = pd.DataFrame()
        print("No anomalies detected.")

    return combined_anomalies

# Define features to analyze
features_to_plot = ['inclination', 'mean_motion', 'eccentricity', 
                    'ra_of_asc_node', 'arg_of_perigee', 'mean_anomaly']

# Calculate baseline and thresholds using the original learning period or fallback to the first 120 days
baseline_stats, rate_of_change_thresholds = calculate_learning_baseline(
    filtered_df, 
    features_to_plot, 
    baseline_start='2016-08-24', 
    baseline_end='2021-08-23', 
    fallback_days=360
)

# Detect and visualize anomalies
combined_anomalies = detect_and_visualize_anomalies_hybrid(
    filtered_df, 
    baseline_stats=baseline_stats, 
    rate_of_change_thresholds=rate_of_change_thresholds, 
    window_size=14, 
    zscore_threshold=2.0, 
    percentile_threshold=0.99
)


# In[16]:


# Ensure the 'datetime' column is in datetime format
filtered_df['datetime'] = pd.to_datetime(filtered_df['datetime'], errors='coerce')

# Extract month for grouping
filtered_df['month'] = filtered_df['datetime'].dt.to_period('M')

# Filter the DataFrame for rows where the anomaly is True
anomalies = filtered_df[filtered_df['statistical_thresholding_anomaly'] == True].copy()

# Group anomalies by month and count occurrences
monthly_anomaly_counts = anomalies.groupby('month').size().reset_index(name='count')

# Convert the period index to datetime for plotting
monthly_anomaly_counts['month'] = monthly_anomaly_counts['month'].dt.to_timestamp()

# Interactive Plotly figure
fig = go.Figure()

# Trace for overall anomalies by month
fig.add_trace(go.Scatter(
    x=monthly_anomaly_counts['month'],
    y=monthly_anomaly_counts['count'],
    mode='lines+markers',
    name='Overall Anomalies'
))

# Plot layout
fig.update_layout(
    title="Count of Anomalies by Month",
    xaxis=dict(
        title="Month",
        showgrid=True,
        rangeslider=dict(visible=True),  # Add a range slider for zooming
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all")
            ]
        )
    ),
    yaxis=dict(
        title="Count of Anomalies",
        showgrid=True
    ),
    legend_title="Anomalies",
    template="plotly_white",
    hovermode="x unified"
)

# Show the plot
fig.show()


# ## OPTICS

# ### Hyperparameter tuning for all Russian Satellites

# In[134]:


# Define a custom scoring function for OPTICS using silhouette score
def optics_silhouette_scorer(X, labels):
    if len(set(labels)) > 1:
        return silhouette_score(X, labels)
    else:
        return -1

# Preprocess the data
def preprocess_data(df, features):
    imputer = SimpleImputer(strategy='mean')
    imputed_data = imputer.fit_transform(df[features])
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(imputed_data)
    return scaled_data

# Define features to process
features_to_include = [
    'inclination', 'mean_motion', 'eccentricity',
    'ra_of_asc_node', 'arg_of_perigee', 'mean_anomaly'
]

# Perform cross-validation
def cross_validate_optics(params, data, n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in kf.split(data):
        train_data = data[train_index]
        val_data = data[val_index]

        optics = OPTICS(
            min_samples=params['min_samples'],
            max_eps=params['max_eps'],
            metric=params['metric'],
            cluster_method="dbscan",
            p=params['p']
        )

        labels = optics.fit_predict(train_data)
        val_labels = optics.fit_predict(val_data)

        # Evaluate using silhouette score on validation set
        score = optics_silhouette_scorer(val_data, val_labels)
        scores.append(score)
    
    return np.mean(scores)

# Function to tune OPTICS for a specific satellite with cross-validation
def tune_optics_for_satellite(satellite_id, df, params, n_trials=50):
    print(f"\n--- Processing Satellite {satellite_id} ---")
    satellite_df = df[df['satellite_number'] == satellite_id].copy()
    
    # Limit data to the first 2000 observations
    satellite_df = satellite_df.head(2000)
    
    if satellite_df.empty:
        print(f"No data available for Satellite {satellite_id} after filtering.")
        return satellite_id, {"error": "No data after filtering"}

    global_data = preprocess_data(satellite_df, features_to_include)
    print(f"Data preprocessing completed for Satellite {satellite_id}. Shape: {global_data.shape}")

    def objective(trial):
        min_samples = trial.suggest_int("min_samples", *params['min_samples_range'])
        max_eps = trial.suggest_float("max_eps", *params['max_eps_range'], step=0.01)
        metric = trial.suggest_categorical("metric", ["minkowski", "manhattan", "euclidean", "chebyshev"])
        p = trial.suggest_int("p", max(1, params['p_range'][0]), min(5, params['p_range'][1]))

        param_set = {
            "min_samples": min_samples,
            "max_eps": max_eps,
            "metric": metric,
            "p": p
        }

        return cross_validate_optics(param_set, global_data)

    print(f"Starting Optuna hyperparameter tuning with cross-validation for Satellite {satellite_id}...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    print(f"Hyperparameter tuning completed for Satellite {satellite_id}.")

    best_params = study.best_params
    print(f"Best hyperparameters for Satellite {satellite_id}: {best_params}")
    
    optics = OPTICS(
        min_samples=best_params["min_samples"],
        max_eps=best_params["max_eps"],
        metric=best_params["metric"],
        cluster_method="dbscan",
        p=best_params["p"]
    )
    labels = optics.fit_predict(global_data)
    if len(set(labels)) <= 1:
        print(f"No valid clusters generated for Satellite {satellite_id}.")
        return satellite_id, {"error": "No valid clusters"}

    best_silhouette_score = optics_silhouette_scorer(global_data, labels)
    print(f"Silhouette score with best parameters for Satellite {satellite_id}: {best_silhouette_score}")

    return satellite_id, {"params": best_params, "silhouette_score": best_silhouette_score}

# Tuning ranges
tuning_params = {
    'min_samples_range': (5, 50),
    'max_eps_range': (0.01, 5.0),
    'p_range': (1, 5),
}

# List of unique satellite IDs
satellite_ids = filtered_df['satellite_number'].unique()

print("Starting parallel hyperparameter tuning for all satellites...")
results = Parallel(n_jobs=-1)(
    delayed(tune_optics_for_satellite)(satellite_id, filtered_df, tuning_params, n_trials=50) for satellite_id in satellite_ids
)

# Separate successful results from errors
best_params_dict = {satellite_id: params for satellite_id, params in results if "params" in params}
skipped_satellites = {satellite_id: params for satellite_id, params in results if "error" in params}

# Print the best parameters and silhouette scores for all satellites
print("\nBest parameters and silhouette scores per satellite:")
for satellite_id, result in best_params_dict.items():
    print(f"Satellite {satellite_id}: {result['params']} (Silhouette Score: {result['silhouette_score']})")

# Print skipped satellites and reasons
print("\nSkipped satellites and reasons:")
for satellite_id, reason in skipped_satellites.items():
    print(f"Satellite {satellite_id}: {reason['error']}")


# ### Russian Satellite Clustering

# In[143]:


unique_satellites = filtered_df['satellite_number'].unique()

# Calculate the average of available hyperparameters
def calculate_average_hyperparameters(best_params_dict):
    min_samples_vals = []
    max_eps_vals = []
    metric_vals = []
    p_vals = []

    for params in best_params_dict.values():
        if "params" in params:
            min_samples_vals.append(params["params"]["min_samples"])
            max_eps_vals.append(params["params"]["max_eps"])
            metric_vals.append(params["params"]["metric"])
            p_vals.append(params["params"]["p"])
    
    avg_params = {
        "min_samples": int(np.mean(min_samples_vals)),
        "max_eps": float(np.mean(max_eps_vals)),
        "metric": max(set(metric_vals), key=metric_vals.count),  # Most frequent metric
        "p": int(np.mean(p_vals))
    }
    return avg_params

# Apply OPTICS clustering function
def apply_optics_clustering_satellite(satellite_id, df, best_params_dict, avg_params):
    print(f"Processing satellite {satellite_id}...")

    # Filter data for the specific satellite
    satellite_df = df[df['satellite_number'] == satellite_id].copy()
    if satellite_df.empty:
        print(f"No data available for satellite {satellite_id}.")
        return None

    # Access the saved hyperparameters for the satellite or use average
    if satellite_id not in best_params_dict:
        print(f"No best parameters found for satellite {satellite_id}. Using average parameters.")
        params = avg_params
    else:
        params = best_params_dict[satellite_id]["params"]

    # Preprocess data
    imputer = SimpleImputer(strategy='mean')
    scaler = RobustScaler()

    satellite_df[features_to_cluster] = imputer.fit_transform(satellite_df[features_to_cluster])
    satellite_df[features_to_cluster] = scaler.fit_transform(satellite_df[features_to_cluster])

    # Initialize OPTICS with the parameters
    optics_model = OPTICS(
        min_samples=params['min_samples'],
        max_eps=params['max_eps'],
        metric=params['metric'],
        cluster_method="dbscan",
        p=params['p']
    )

    # Apply clustering
    labels = optics_model.fit_predict(satellite_df[features_to_cluster])
    if labels is None or len(labels) == 0:
        print(f"No clusters generated for satellite {satellite_id}. Skipping.")
        return None

    # Assign the cluster labels
    satellite_df['Satellite_OPTICS_cluster'] = labels

    # Identify the largest cluster (baseline cluster)
    cluster_counts = pd.Series(labels).value_counts()
    largest_cluster_label = cluster_counts.idxmax()  # Label of the largest cluster (baseline)
    baseline_centroid = satellite_df[satellite_df['Satellite_OPTICS_cluster'] == largest_cluster_label][features_to_cluster].mean().values

    # Calculate distances and update Satellite_OPTICS_anomaly
    satellite_df['distance_from_baseline'] = satellite_df[features_to_cluster].apply(
        lambda row: cdist([row], [baseline_centroid], metric='euclidean')[0][0], axis=1
    )

    # Exclude the noise points (those with OPTICS label -1) from anomalies
    satellite_df['Satellite_OPTICS_anomaly'] = (satellite_df['Satellite_OPTICS_cluster'] != -1) & (satellite_df['distance_from_baseline'] > 5)

    # Print cluster distances for valid clusters (not noise)
    unique_clusters = satellite_df[satellite_df['Satellite_OPTICS_cluster'] != -1]['Satellite_OPTICS_cluster'].unique()
    for cluster in unique_clusters:
        cluster_points = satellite_df[satellite_df['Satellite_OPTICS_cluster'] == cluster][features_to_cluster].values
        if len(cluster_points) > 0:
            distances = cdist(cluster_points, [baseline_centroid], metric='euclidean')
            print(f"Cluster {cluster}: Mean Distance from Baseline = {distances.mean()}")

    # Output counts
    print(f"Cluster counts for satellite {satellite_id}:\n{cluster_counts}")
    print(f"Largest cluster (baseline): {largest_cluster_label}")
    print(f"Number of anomalies: {satellite_df['Satellite_OPTICS_anomaly'].sum()}")

    return satellite_df

# Calculate average hyperparameters
avg_params = calculate_average_hyperparameters(best_params_dict)

# Apply clustering for each satellite with average parameters fallback
print("Starting parallel processing for each satellite...")
results = Parallel(n_jobs=-1)(
    delayed(apply_optics_clustering_satellite)(satellite_id, filtered_df, best_params_dict, avg_params)
    for satellite_id in unique_satellites
)

# Combine the results from all satellites into a single DataFrame
final_clustered_df = pd.concat([df for df in results if df is not None], ignore_index=True)

# Display a few rows of the final DataFrame
print("Final clustered data:")
print(final_clustered_df.head())

# Output overall anomaly and cluster counts
overall_cluster_counts = final_clustered_df['Satellite_OPTICS_cluster'].value_counts()
overall_anomaly_count = final_clustered_df['Satellite_OPTICS_anomaly'].sum()
print(f"Overall cluster counts:\n{overall_cluster_counts}")
print(f"Total anomalies detected: {overall_anomaly_count}")


# In[144]:


# Define a function to calculate outliers using different methods
def calculate_outliers(df, column_nm, threshold=2, method='mad'):
    """Identify outliers using a specified method."""
    if method == 'zscore':
        z_scores = np.abs((df[column_nm] - df[column_nm].mean()) / df[column_nm].std())
        df[f'outlier_{column_nm}'] = (z_scores > threshold).astype(int)
    elif method == 'iqr':
        q1 = df[column_nm].quantile(0.25)
        q3 = df[column_nm].quantile(0.75)
        iqr = q3 - q1
        df[f'outlier_{column_nm}'] = ((df[column_nm] < (q1 - 1.5 * iqr)) | (df[column_nm] > (q3 + 1.5 * iqr))).astype(int)
    elif method == 'mad':
        median = df[column_nm].median()
        mad = median_abs_deviation(df[column_nm])
        modified_z = 0.6745 * (df[column_nm] - median) / (mad + 1e-9)  # Avoid division by zero
        df[f'outlier_{column_nm}'] = (np.abs(modified_z) > threshold).astype(int)
    else:
        raise ValueError("Invalid outlier detection method. Choose from 'zscore', 'iqr', or 'mad'.")

# Split the data into training and testing periods
train_data = final_clustered_df[(final_clustered_df['datetime'] >= '2016-08-24') & (final_clustered_df['datetime'] <= '2021-08-23')].copy()
test_data = final_clustered_df[(final_clustered_df['datetime'] >= '2021-08-24') & (final_clustered_df['datetime'] <= '2022-02-24')].copy()

# Identify outliers across all features in the test data
features_to_include = ['inclination', 'mean_motion', 'eccentricity', 
                       'ra_of_asc_node', 'arg_of_perigee', 'mean_anomaly']

test_data['outlier'] = False  # Initialize the outlier column
progress = 0
total_features = len(features_to_include)

for feature in features_to_include:
    if feature in test_data.columns:
        calculate_outliers(test_data, feature, threshold=2, method='mad')  # Use 'mad' as default
        test_data['outlier'] |= test_data[f'outlier_{feature}']  # Combine outliers across all features
    progress += 1
    print(f"Outlier calculation progress: {progress / total_features * 100:.2f}%")

# Compare OPTICS-detected anomalies with outliers
def evaluate_optics_performance(test_data, optics_anomaly_column='Satellite_OPTICS_anomaly'):
    """Evaluate the performance of OPTICS-detected anomalies."""
    if optics_anomaly_column not in test_data.columns:
        raise ValueError(f"{optics_anomaly_column} column not found in the test dataset.")

    y_true = test_data['outlier'].astype(int)
    y_pred = test_data[optics_anomaly_column].astype(int)

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred)
    }
    return metrics

# Evaluate OPTICS performance
performance_metrics = evaluate_optics_performance(test_data)

# Print performance metrics
print("\n### OPTICS Performance Metrics ###")
for metric, value in performance_metrics.items():
    print(f"{metric}: {value:.4f}")


# In[145]:


final_clustered_df['Satellite_OPTICS_anomaly'].value_counts()


# In[147]:


# Function to plot PCA for each orbit class
def plot_pca_by_orbit_class(df, features, orbit_classes, cluster_column):
    for orbit_class in orbit_classes:
        # Filter data for the specific orbit class
        orbit_df = df[df['Class of Orbit'] == orbit_class]
        
        if orbit_df.empty:
            print(f"No data available for {orbit_class}. Skipping.")
            continue

        # Perform PCA
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(orbit_df[features])
        
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], 
                               c=orbit_df[cluster_column], cmap='viridis', s=10)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f"PCA Cluster Visualization for {orbit_class}")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid(True)
        plt.show()

# Define orbit classes and clustering column
orbit_classes = final_clustered_df['Class of Orbit'].unique()
features_to_cluster = ['inclination', 'mean_motion', 'eccentricity', 
                       'ra_of_asc_node', 'arg_of_perigee', 'mean_anomaly']

# Call the function using final_clustered_df
plot_pca_by_orbit_class(final_clustered_df, features_to_cluster, orbit_classes, 'Satellite_OPTICS_cluster')


# In[148]:


cluster_stats = final_clustered_df.groupby('Satellite_OPTICS_cluster')[features_to_cluster].agg(['mean', 'std', 'min', 'max'])
cluster_stats


# In[149]:


# Calculate centroid of cluster 0 (baseline cluster)
baseline_centroid = final_clustered_df[final_clustered_df['Satellite_OPTICS_cluster'] == 0][features_to_cluster].mean().values

# Iterate over all unique clusters and calculate distances from the baseline centroid
unique_clusters = final_clustered_df['Satellite_OPTICS_cluster'].unique()

for cluster in unique_clusters:
    cluster_points = final_clustered_df[final_clustered_df['Satellite_OPTICS_cluster'] == cluster][features_to_cluster].values
    
    # Skip empty clusters (just in case)
    if len(cluster_points) == 0:
        print(f"Cluster {cluster} has no points.")
        continue
    
    # Calculate distances
    distances = cdist(cluster_points, [baseline_centroid], metric='euclidean')
    mean_distance = distances.mean()
    
    print(f"Cluster {cluster}: Mean Distance from Baseline = {mean_distance}")


# In[150]:


for feature in features_to_cluster:
    plt.figure(figsize=(35, 6))
    sns.boxplot(x='Satellite_OPTICS_cluster', y=feature, data=final_clustered_df)
    plt.title(f'{feature} Distribution Across Clusters')
    plt.show()


# In[151]:


# Filter the DataFrame to only include anomalies
anomaly_data = final_clustered_df[final_clustered_df['Satellite_OPTICS_anomaly']].copy()

# Group by month to calculate the total count of anomalies
anomaly_data['month'] = anomaly_data['datetime'].dt.to_period('M')
monthly_anomaly_counts = anomaly_data.groupby('month').size().reset_index(name='count')

# Convert month to datetime for plotting
monthly_anomaly_counts['month'] = monthly_anomaly_counts['month'].dt.to_timestamp()

# Create the interactive plot
fig = go.Figure()

# A single trace for the total anomaly counts
fig.add_trace(go.Scatter(
    x=monthly_anomaly_counts['month'],
    y=monthly_anomaly_counts['count'],
    mode='lines+markers',
    name='Total Anomalies',
    marker=dict(size=6)
))

# Plot layout
fig.update_layout(
    title="Monthly Anomaly Counts",
    xaxis_title="Month",
    yaxis_title="Anomaly Count",
    template="plotly_white",
    xaxis=dict(
        showgrid=True,
        rangeslider=dict(visible=True),  # Add the range slider (zoom bar)
        rangeselector=dict(              # Add range selector buttons
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all")
            ]
        )
    ),
    yaxis=dict(showgrid=True),
    hovermode="x unified"  # Show all traces' values on hover
)

# Show the interactive plot
fig.show()


# In[152]:


# Filter for anomalies only and create a copy to avoid warnings
anomaly_data = final_clustered_df[final_clustered_df['Satellite_OPTICS_anomaly']].copy()

# Group by month and Purpose to calculate counts of anomalies
anomaly_data['month'] = anomaly_data['datetime'].dt.to_period('M')  # Safely modify after .copy()
anomaly_counts = (
    anomaly_data
    .groupby(['month', 'Purpose'])
    .size()
    .reset_index(name='count')
)

# Pivot data for plotting (Purpose as columns)
anomaly_pivot = anomaly_counts.pivot(index='month', columns='Purpose', values='count').fillna(0)

# Convert month to datetime for plotting
anomaly_pivot.index = anomaly_pivot.index.to_timestamp()

# Create the plot
fig, ax = plt.subplots(figsize=(14, 8))

# Plot each Purpose with a distinct color
for purpose in anomaly_pivot.columns:
    ax.plot(
        anomaly_pivot.index,
        anomaly_pivot[purpose],
        marker='o',
        label=purpose
    )

# Customize the plot
ax.set_title("Monthly Anomaly Counts by Purpose (OPTICS Anomalies Only)", fontsize=16)
ax.set_xlabel("Month", fontsize=14)
ax.set_ylabel("Count of Anomalies", fontsize=14)
ax.grid(True)
ax.legend(title="Purpose", fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

# Filter for anomalies only and create a copy to avoid warnings
anomaly_data = final_clustered_df[final_clustered_df['statistical_thresholding_anomaly']].copy()

# Group by month and Purpose to calculate counts of anomalies
anomaly_data['month'] = anomaly_data['datetime'].dt.to_period('M')  # Safely modify after .copy()
anomaly_counts = (
    anomaly_data
    .groupby(['month', 'Purpose'])
    .size()
    .reset_index(name='count')
)

# Pivot data for plotting (Purpose as columns)
anomaly_pivot = anomaly_counts.pivot(index='month', columns='Purpose', values='count').fillna(0)

# Convert month to datetime for plotting
anomaly_pivot.index = anomaly_pivot.index.to_timestamp()

# Create the plot
fig, ax = plt.subplots(figsize=(14, 8))

# Plot each Purpose with a distinct color
for purpose in anomaly_pivot.columns:
    ax.plot(
        anomaly_pivot.index,
        anomaly_pivot[purpose],
        marker='o',
        label=purpose
    )

# Customize the plot
ax.set_title("Monthly Anomaly Counts by Purpose (statistical_thresholding_anomaly)", fontsize=16)
ax.set_xlabel("Month", fontsize=14)
ax.set_ylabel("Count of Anomalies", fontsize=14)
ax.grid(True)
ax.legend(title="Purpose", fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

# Filter for rows where both anomalies are True and create a copy to avoid warnings
anomaly_data = final_clustered_df[
    (final_clustered_df['Satellite_OPTICS_anomaly']) & (final_clustered_df['statistical_thresholding_anomaly'])
].copy()

# Group by month and Purpose to calculate counts of anomalies
anomaly_data['month'] = anomaly_data['datetime'].dt.to_period('M')  # Safely modify after .copy()
anomaly_counts = (
    anomaly_data
    .groupby(['month', 'Purpose'])
    .size()
    .reset_index(name='count')
)

# Pivot data for plotting (Purpose as columns)
anomaly_pivot = anomaly_counts.pivot(index='month', columns='Purpose', values='count').fillna(0)

# Convert month to datetime for plotting
anomaly_pivot.index = anomaly_pivot.index.to_timestamp()

# Create the plot
fig, ax = plt.subplots(figsize=(14, 8))

# Plot each Purpose with a distinct color
for purpose in anomaly_pivot.columns:
    ax.plot(
        anomaly_pivot.index,
        anomaly_pivot[purpose],
        marker='o',
        label=purpose
    )

# Customize the plot
ax.set_title("Monthly Anomaly Counts by Purpose (OPTICS & Statistical Thresholding Anomalies Only)", fontsize=16)
ax.set_xlabel("Month", fontsize=14)
ax.set_ylabel("Count of Anomalies", fontsize=14)
ax.grid(True)
ax.legend(title="Purpose", fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()



# In[155]:


final_clustered_df.to_csv('RU_SAT_Final.csv', index=False)


# In[ ]:


# Filter the data for OPTICS anomalies within the specified date range
anomaly_data = final_clustered_df[final_clustered_df['Satellite_OPTICS_anomaly']].copy()
anomaly_data = anomaly_data[(anomaly_data['datetime'] >= pd.Timestamp('2021-04-01')) &
                             (anomaly_data['datetime'] < pd.Timestamp('2022-04-01'))]

# Ensure the datetime is formatted correctly for display
anomaly_data['anomaly_date'] = anomaly_data['datetime'].dt.strftime('%Y-%m-%d')

# Combine satellite_number and date for hover information
anomaly_data['hover_info'] = anomaly_data['satellite_number'].astype(str) + " | Date: " + anomaly_data['anomaly_date']

# Check for necessary columns
required_columns = {'x', 'y', 'Purpose', 'Users', 'hover_info'}
if not required_columns.issubset(anomaly_data.columns):
    raise ValueError(f"The DataFrame must contain these columns for mapping: {required_columns}")

# Create a scattergeo plot for Purpose
fig = px.scatter_geo(
    anomaly_data,
    lon='x',  
    lat='y',  
    color='Purpose',  
    hover_name='hover_info',  
    hover_data={'x': False, 'y': False, 'Purpose': True}, 
    title="World Map of OPTICS Anomalies by Purpose (April 2021 to April 2022)",
    template='plotly_white'
)

fig.update_geos(
    scope='world',
    projection_type='orthographic',  
    showland=True,  
    landcolor="rgb(243, 243, 243)",
    oceancolor="rgb(230, 250, 255)",
    showocean=True,
    subunitcolor="rgb(217, 217, 217)",
    showcoastlines=True,
    coastlinecolor="rgb(0, 0, 0)", 
    showcountries=True, 
    countrycolor="rgb(0, 0, 0)" 
)

# Show the plot
fig.show()

# Create a scattergeo plot for Users
fig = px.scatter_geo(
    anomaly_data,
    lon='x',  
    lat='y',  
    color='Users', 
    hover_name='hover_info',  
    hover_data={'x': False, 'y': False, 'Users': True}, 
    title="World Map of OPTICS Anomalies by Users (April 2021 to April 2022)",
    template='plotly_white'
)

fig.update_geos(
    scope='world',
    projection_type='orthographic',  
    showland=True,  
    landcolor="rgb(243, 243, 243)",
    oceancolor="rgb(230, 250, 255)",
    showocean=True,
    subunitcolor="rgb(217, 217, 217)",
    showcoastlines=True,
    coastlinecolor="rgb(0, 0, 0)", 
    showcountries=True,  
    countrycolor="rgb(0, 0, 0)" 
)

# Show the plot
fig.show()


# ### Stastical Test

# In[157]:


def get_anomaly_rate(df, anomaly_column=None):
    """Calculate the anomaly rate for a specific anomaly column or combined anomalies."""
    if anomaly_column:
        # Use only the specified anomaly column
        if anomaly_column not in df.columns:
            print(f"Anomaly column {anomaly_column} not found in DataFrame.")
            return 0
        df['anomaly_ind'] = df[anomaly_column].astype(int)
    else:
        # Combined logic: BOTH methods must flag as an anomaly
        if 'statistical_thresholding_anomaly' not in df.columns or 'Satellite_OPTICS_anomaly' not in df.columns:
            print("Required anomaly columns are missing from the DataFrame.")
            return 0
        df['anomaly_ind'] = (df['statistical_thresholding_anomaly'] & df['Satellite_OPTICS_anomaly']).astype(int)

    total_observations = len(df)
    anomaly_count = df['anomaly_ind'].sum()
    
    if total_observations == 0:
        print("No observations in the DataFrame for calculating anomaly rate.")
        return 0

    return anomaly_count / total_observations

def perform_chi_square_test(baseline_df, leadup_df):
    """Perform a chi-square test of independence for two datasets."""
    baseline_anomalies = baseline_df['anomaly_ind'].sum()
    baseline_non_anomalies = len(baseline_df) - baseline_anomalies

    leadup_anomalies = leadup_df['anomaly_ind'].sum()
    leadup_non_anomalies = len(leadup_df) - leadup_anomalies

    # Construct contingency table
    contingency_table = [
        [baseline_anomalies, baseline_non_anomalies],
        [leadup_anomalies, leadup_non_anomalies]
    ]

    # Check for valid contingency table
    if any(sum(row) == 0 for row in contingency_table):
        print("Invalid contingency table for chi-square test:", contingency_table)
        return None, None

    # Perform chi-square test
    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
    return chi2_stat, p_value

def hypoth_test(df, col, anomaly_column=None):
    """Perform hypothesis testing for anomalies based on a specific column."""
    chi_square_lst = []
    p_value_lst = []
    base_anom_rate_lst = []
    lead_anom_rate_lst = []
    group_lst = []

    print(f"Starting hypothesis testing for column: {col}")
    unique_groups = df[col].dropna().unique()
    if len(unique_groups) == 0:
        print(f"No valid groups found for column: {col}")
        return pd.DataFrame()

    for group in unique_groups:
        group_df = df[df[col] == group]

        # Split into baseline and leadup periods
        baseline_period = group_df[group_df['datetime'] <= '2021-08-24']
        leadup_period = group_df[(group_df['datetime'] > '2021-08-24') & (group_df['datetime'] < '2022-02-25')]

        if baseline_period.empty or leadup_period.empty:
            print(f"Skipping group {group}: No data in baseline or leadup period.")
            continue

        baseline_rate = get_anomaly_rate(baseline_period, anomaly_column)
        leadup_rate = get_anomaly_rate(leadup_period, anomaly_column)

        if baseline_rate == 0 and leadup_rate == 0:
            print(f"Skipping group {group}: No anomalies detected.")
            continue

        chi2_stat, p_value = perform_chi_square_test(baseline_period, leadup_period)

        if chi2_stat is None or p_value is None:
            print(f"Skipping group {group}: Invalid chi-square test results.")
            continue

        group_lst.append(group)
        chi_square_lst.append(float(chi2_stat))
        p_value_lst.append(float(p_value))
        base_anom_rate_lst.append(float(baseline_rate))
        lead_anom_rate_lst.append(float(leadup_rate))

    if not group_lst:
        print("No valid groups processed for hypothesis testing.")
        return pd.DataFrame(columns=[
            'chi_square', 'p_value', 'baseline_anomaly_rate',
            'leadup_anomaly_rate', 'statistical_significant'
        ])

    sig_results_df = pd.DataFrame({
        'chi_square': chi_square_lst,
        'p_value': p_value_lst,
        'baseline_anomaly_rate': base_anom_rate_lst,
        'leadup_anomaly_rate': lead_anom_rate_lst
    }, index=group_lst)

    sig_results_df['statistical_significant'] = (sig_results_df['p_value'] <= 0.05).astype(int)
    return sig_results_df

# Perform tests for 'Purpose'
print("### Hypothesis Test Results for Purpose ###")
purpose_stats_anomaly = hypoth_test(final_clustered_df, 'Purpose', anomaly_column='statistical_thresholding_anomaly')
purpose_optics_anomaly = hypoth_test(final_clustered_df, 'Purpose', anomaly_column='Satellite_OPTICS_anomaly')
purpose_combined_anomaly = hypoth_test(final_clustered_df, 'Purpose')  # BOTH methods must agree

print("\nStatistical Thresholding Anomaly:")
print(purpose_stats_anomaly)
print("\nSatellite OPTICS Anomaly:")
print(purpose_optics_anomaly)
print("\nCombined Anomaly (BOTH methods must agree):")
print(purpose_combined_anomaly)

# Perform tests for 'Class of Orbit'
print("\n### Hypothesis Test Results for Class of Orbit ###")
orbit_stats_anomaly = hypoth_test(final_clustered_df, 'Class of Orbit', anomaly_column='statistical_thresholding_anomaly')
orbit_optics_anomaly = hypoth_test(final_clustered_df, 'Class of Orbit', anomaly_column='Satellite_OPTICS_anomaly')
orbit_combined_anomaly = hypoth_test(final_clustered_df, 'Class of Orbit')  # BOTH methods must agree

print("\nStatistical Thresholding Anomaly:")
print(orbit_stats_anomaly)
print("\nSatellite OPTICS Anomaly:")
print(orbit_optics_anomaly)
print("\nCombined Anomaly (BOTH methods must agree):")
print(orbit_combined_anomaly)


# In[158]:


# Add a column for the method in each DataFrame
purpose_stats_anomaly['method'] = 'statistical_thresholding'
purpose_optics_anomaly['method'] = 'optics'
purpose_combined_anomaly['method'] = 'combined'

# Combine all results vertically
combined_purpose_results = pd.concat(
    [purpose_stats_anomaly, purpose_optics_anomaly, purpose_combined_anomaly],
    axis=0
).reset_index()

# Display the combined DataFrame
print("\n### Combined Results for Purpose ###")
combined_purpose_results


# In[159]:


# Perform the unique satellite ID count grouped by Purpose and Class of Orbit
unique_satellite_counts = (
    final_clustered_df.groupby(['Purpose', 'Class of Orbit'])['satellite_number']
    .nunique()
    .reset_index()
)

# Rename the column for clarity
unique_satellite_counts.rename(columns={'satellite_number': 'unique_satellite_count'}, inplace=True)

# Merge the unique satellite counts with the combined_purpose_results dataframe
# Explicitly specify suffixes to avoid conflicts
combined_purpose_results = combined_purpose_results.merge(
    unique_satellite_counts,
    how='left',
    left_on=['index'], 
    right_on=['Purpose'],
    suffixes=('', '_new') 
)

# Remove unnecessary '_new' suffix columns
for col in ['Class of Orbit_new', 'unique_satellite_count_new']:
    if col in combined_purpose_results.columns:
        combined_purpose_results.rename(columns={col: col.replace('_new', '')}, inplace=True)

# Drop redundant 'Purpose' column after the merge
combined_purpose_results.drop(columns=['Purpose'], errors='ignore', inplace=True)

# Display the resulting DataFrame
print("\n### Combined Purpose Results with Unique Satellite Counts ###")
combined_purpose_results


# In[160]:


# Separate anomaly rates for Statistical Thresholding, OPTICS, and Combined methods
methods = ['Statistical Thresholding', 'OPTICS', 'Combined']
anomaly_columns = ['statistical_thresholding_anomaly', 'Satellite_OPTICS_anomaly', None]  # None for combined

for method, anomaly_column in zip(methods, anomaly_columns):
    # Calculate the grouped mean anomaly rates for the method
    combined_purpose_results[f'{method}_baseline_rate'] = combined_purpose_results['baseline_anomaly_rate']
    combined_purpose_results[f'{method}_leadup_rate'] = combined_purpose_results['leadup_anomaly_rate']

    # Plotting
    plt.figure(figsize=(14, 7))
    
    x_labels = combined_purpose_results['index'] + " - " + combined_purpose_results['Class of Orbit']
    x_indices = np.arange(len(x_labels))  # Numerical indices for the x-axis
    
    bar_width = 0.4  # Width of each bar

    # Plot baseline and leadup bars side by side
    baseline_bars = plt.bar(x_indices - bar_width / 2, combined_purpose_results[f'{method}_baseline_rate'], 
                            width=bar_width, label='Baseline Anomaly Rate', alpha=0.7)
    leadup_bars = plt.bar(x_indices + bar_width / 2, combined_purpose_results[f'{method}_leadup_rate'], 
                          width=bar_width, label='Leadup Anomaly Rate', alpha=0.7)

    # Add numbers at the top of each bar with angled text
    for bar in baseline_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', 
                 fontsize=8, rotation=45)

    for bar in leadup_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', 
                 fontsize=8, rotation=45)

    # Customizations
    plt.xticks(ticks=x_indices, labels=x_labels, rotation=45, ha='right')
    plt.xlabel("Purpose - Orbit Class")
    plt.ylabel("Anomaly Rate")
    plt.title(f"Anomaly Rates: {method} Method by Purpose and Orbit Class")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ## VAE Model with OPTICS in latent space

# In[17]:


# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[22]:


# Define the date ranges for different periods
baseline_start = pd.Timestamp('2016-08-24')
baseline_end = pd.Timestamp('2021-08-23')
leadup_start = pd.Timestamp('2021-08-24')
leadup_end = pd.Timestamp('2022-02-23')
post_invasion_start = pd.Timestamp('2022-02-24')
post_invasion_end = pd.Timestamp('2024-02-24')

features_to_use = ['inclination', 'mean_motion', 'eccentricity', 
                   'ra_of_asc_node', 'arg_of_perigee', 'mean_anomaly']

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=4):
        super(VAE, self).__init__()
        # Encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.mu = nn.Linear(16, latent_dim)
        self.logvar = nn.Linear(16, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, input_dim)
        )

    def encode(self, x):
        h = self.shared_encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar

# Dataset class
class TelemetryDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# VAE loss function without KL divergence
def vae_loss_function(reconstructed, original):
    recon_loss = nn.functional.mse_loss(reconstructed, original, reduction='sum')
    return recon_loss

# Extract latent space representation
def extract_latent(loader, vae, device):
    latent_representations = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            mu, logvar = vae.encode(batch)
            z = vae.reparameterize(mu, logvar)
            latent_representations.append(z.cpu().numpy())
    return np.vstack(latent_representations)

# Suggest dynamic range for max_eps
def suggest_max_eps_range(latent_space):
    dist_std = np.std(latent_space)
    return max(0.05, dist_std / 20), min(5.0, dist_std * 3)  # Adjusted range for OPTICS

# Optimize OPTICS hyperparameters
def optimize_optics(latent_space):
    def objective(trial):
        max_eps_min, max_eps_max = suggest_max_eps_range(latent_space)
        max_eps = trial.suggest_float('max_eps', max_eps_min, max_eps_max, step=0.05)
        min_samples = trial.suggest_int('min_samples', 5, max(10, len(latent_space) // 10))
        optics = OPTICS(min_samples=min_samples, max_eps=max_eps, metric='euclidean')
        labels = optics.fit_predict(latent_space)
        if len(set(labels)) > 1:
            return silhouette_score(latent_space, labels)
        else:
            return -1

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)  # Increased trials for better tuning
    return study.best_params

# Assign labels safely
def assign_optics_labels(data, labels, column_name):
    data = data.copy()
    if len(labels) > 0:
        data[column_name] = labels
    else:
        data[column_name] = -1
    return data

# Train VAE with Early Stopping
def train_vae_with_early_stopping(vae, baseline_loader, val_loader, optimizer, scheduler, device, max_epochs=200, patience=10):
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(max_epochs):
        if early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

        vae.train()
        train_loss = 0
        for batch in baseline_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed, _, _ = vae(batch)
            loss = vae_loss_function(reconstructed, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(baseline_loader)

        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstructed, _, _ = vae(batch)
                loss = vae_loss_function(reconstructed, batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop = True

    return vae

# Main processing loop
all_results = []
satellite_ids = filtered_df['satellite_number'].unique()
best_params_per_satellite = {}

for sat_id in satellite_ids:
    print(f"Processing satellite: {sat_id}")
    
    satellite_data = filtered_df[filtered_df['satellite_number'] == sat_id]
    
    baseline_data = satellite_data[(satellite_data['datetime'] >= baseline_start) &
                                   (satellite_data['datetime'] <= baseline_end)]
    leadup_data = satellite_data[(satellite_data['datetime'] >= leadup_start) &
                                 (satellite_data['datetime'] <= leadup_end)]
    post_invasion_data = satellite_data[(satellite_data['datetime'] >= post_invasion_start) &
                                        (satellite_data['datetime'] <= post_invasion_end)]
    
    if len(baseline_data) < 10:
        print(f"Skipping satellite {sat_id} due to insufficient data.")
        continue

    scaler = RobustScaler()
    baseline_features = scaler.fit_transform(baseline_data[features_to_use].dropna())
    leadup_features = scaler.transform(leadup_data[features_to_use].dropna())
    post_invasion_features = scaler.transform(post_invasion_data[features_to_use].dropna())

    train_size = int(0.8 * len(baseline_features))
    val_size = len(baseline_features) - train_size
    train_dataset, val_dataset = random_split(
        TelemetryDataset(baseline_features), [train_size, val_size]
    )

    baseline_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    leadup_loader = DataLoader(TelemetryDataset(leadup_features), batch_size=64, shuffle=False)
    post_invasion_loader = DataLoader(TelemetryDataset(post_invasion_features), batch_size=64, shuffle=False)

    vae = VAE(input_dim=len(features_to_use), latent_dim=8).to(device)
    optimizer = optim.AdamW(vae.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    vae = train_vae_with_early_stopping(vae, baseline_loader, val_loader, optimizer, scheduler, device, max_epochs=200, patience=10)
    
    baseline_latent = extract_latent(baseline_loader, vae, device)
    leadup_latent = extract_latent(leadup_loader, vae, device)
    post_invasion_latent = extract_latent(post_invasion_loader, vae, device)

    print(f"Tuning OPTICS hyperparameters for satellite {sat_id} (baseline period)...")
    best_params = optimize_optics(baseline_latent)
    best_params_per_satellite[sat_id] = best_params
    print(f"Best OPTICS parameters for satellite {sat_id}: {best_params}")

    optics_leadup = OPTICS(**best_params, metric='euclidean')
    optics_post_invasion = OPTICS(**best_params, metric='euclidean')
    
    leadup_labels = optics_leadup.fit_predict(leadup_latent)
    post_invasion_labels = optics_post_invasion.fit_predict(post_invasion_latent)

    leadup_data = assign_optics_labels(leadup_data, leadup_labels, 'optics_cluster')
    post_invasion_data = assign_optics_labels(post_invasion_data, post_invasion_labels, 'optics_cluster')

    all_results.append(pd.concat([leadup_data, post_invasion_data]))

final_results = pd.concat(all_results, ignore_index=True)

# Update anomaly detection logic to exclude noise points (optics_cluster == -1)
final_results['anomaly_ind'] = (final_results['optics_cluster'] != -1) & (final_results['distance_from_baseline'] > 10)


# In[114]:


input_dim = baseline_features.shape[1]
print(f"Input dimension: {input_dim}")


# In[115]:


print(final_results['optics_cluster'].value_counts())
print(final_results['anomaly_ind'].value_counts())


# In[116]:


# Define the lead-up period time range
leadup_start = pd.Timestamp('2021-08-24 00:02:11.010048')
leadup_end = pd.Timestamp('2022-02-24 23:57:58.202784')

# Ensure final_results is a DataFrame and has the 'datetime' column in the correct format
if 'datetime' not in final_results.columns:
    raise KeyError("The 'datetime' column is missing in final_results.")

# Convert 'datetime' column to pandas datetime if not already in datetime format
if not pd.api.types.is_datetime64_any_dtype(final_results['datetime']):
    final_results['datetime'] = pd.to_datetime(final_results['datetime'])

# Filter data for the lead-up period
leadup_data = final_results[
    (final_results['datetime'] >= leadup_start) & (final_results['datetime'] <= leadup_end)
].copy()

# Verify the filtered data
if leadup_data.empty:
    raise ValueError("No data available for the specified lead-up period. Check the date range or final_results data.")

print(f"Lead-up data created with {leadup_data.shape[0]} rows.")


# In[117]:


final_results.to_csv('DL_Final.csv', index=False)


# In[48]:


final_results = pd.read_csv('DL_final.csv')


# In[49]:


final_results.info()


# ### Evaluate the VAE-OPTICS Model

# In[50]:


# Define the columns to check for outliers
features_to_cluster = ['inclination', 'ra_of_asc_node', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 'mean_motion']

# Function to calculate outliers using the specified method
def calculate_outlier(df, column_nm, threshold=2, method='mad'):
    if method == 'iqr':
        # IQR Method
        q1 = df[column_nm].quantile(0.25)
        q3 = df[column_nm].quantile(0.75)
        iqr = q3 - q1
        df[f'outlier_{column_nm}'] = ((df[column_nm] < (q1 - 1.5 * iqr)) | (df[column_nm] > (q3 + 1.5 * iqr))).astype(int)
    
    elif method == 'zscore':
        # Z-Score Method
        z_scores = np.abs((df[column_nm] - df[column_nm].mean()) / df[column_nm].std())
        df[f'outlier_{column_nm}'] = (z_scores > threshold).astype(int)

    elif method == 'mad':
        # MAD Method
        median = df[column_nm].median()
        mad = median_abs_deviation(df[column_nm])
        if mad == 0:  # Avoid division by zero
            df[f'outlier_{column_nm}'] = 0
        else:
            modified_z = 0.6745 * (df[column_nm] - median) / mad
            df[f'outlier_{column_nm}'] = (abs(modified_z) > threshold).astype(int)

    else:
        raise ValueError('Invalid outlier detection method. Choose "iqr", "zscore", or "mad".')

# Add the outlier columns to the DataFrame
def add_outlier_column(df, threshold=2, method='mad'):
    for col in features_to_cluster:
        calculate_outlier(df, col, threshold=threshold, method=method)
    
    # Combine individual column outliers into a general outlier column
    df['outlier'] = (df.filter(like='outlier_') == 1).any(axis=1).astype(int)
    return df

# Example usage
# Assuming `final_results` is the DataFrame
final_results = add_outlier_column(final_results, threshold=2, method='mad')

# Display the first few rows to verify
print(final_results[['outlier'] + [f'outlier_{col}' for col in features_to_cluster]].head())


# In[51]:


final_results['anomaly_ind'].value_counts()


# In[52]:


# Function to evaluate the model
def evaluate_model(df, prediction_column='anomaly_ind', threshold=2, method='mad'):
    # Make a copy of the DataFrame to preserve the original
    evaluation_df = df.copy()

    # Define the columns to check for outliers
    features_to_cluster = ['inclination', 'ra_of_asc_node', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 'mean_motion']

    # Function to calculate outliers using specified method
    def calculate_outlier(df, column_nm, threshold=2, method='mad'):
        if method == 'iqr':
            # IQR Method
            q1 = df[column_nm].quantile(0.25)
            q3 = df[column_nm].quantile(0.75)
            iqr = q3 - q1
            df[f'outlier_{column_nm}'] = ((df[column_nm] < (q1 - 1.5 * iqr)) | (df[column_nm] > (q3 + 1.5 * iqr))).astype(int)
        
        elif method == 'zscore':
            # Z-Score Method
            z_scores = np.abs((df[column_nm] - df[column_nm].mean()) / df[column_nm].std())
            df[f'outlier_{column_nm}'] = (z_scores > threshold).astype(int)

        elif method == 'mad':
            # MAD Method
            median = df[column_nm].median()
            mad = median_abs_deviation(df[column_nm])
            if mad == 0:  # Avoid division by zero
                df[f'outlier_{column_nm}'] = 0
            else:
                modified_z = 0.6745 * (df[column_nm] - median) / mad
                df[f'outlier_{column_nm}'] = (abs(modified_z) > threshold).astype(int)

        else:
            raise ValueError('Invalid outlier detection method. Choose "iqr", "zscore", or "mad".')

    # Add outlier columns to the evaluation DataFrame
    for col in features_to_cluster:
        calculate_outlier(evaluation_df, col, threshold=threshold, method=method)

    # Combine individual column outliers into a general outlier column
    evaluation_df['outlier'] = (evaluation_df.filter(like='outlier_') == 1).any(axis=1).astype(int)

    # Evaluate the model performance
    print("\nEvaluating model performance...")

    def evaluate_binary_classifier(y_true, y_pred, model_name='Model'):
        metrics = {
            'Accuracy': accuracy_score,
            'Precision': precision_score,
            'Recall': recall_score,
            'F1': f1_score
        }

        results = {}
        for metric_name, metric_func in metrics.items():
            if metric_name == 'Accuracy':
                score = metric_func(y_true, y_pred)
            else:
                score = metric_func(y_true, y_pred, average='binary')
            results[metric_name] = score
            print(f"{metric_name} for {model_name}: {score:.3f}")

        return results

    # Ensure required columns exist
    if 'outlier' not in evaluation_df.columns or prediction_column not in evaluation_df.columns:
        raise KeyError(f"'outlier' or '{prediction_column}' column is missing in the evaluation DataFrame.")

    y_true = evaluation_df['outlier']
    y_pred = evaluation_df[prediction_column]

    results = evaluate_binary_classifier(y_true, y_pred, model_name='Anomaly Detection Model')
    return results

evaluation_results = evaluate_model(final_results, prediction_column='anomaly_ind', threshold=2, method='mad')


# In[53]:


if 'final_results' in locals():
    anomaly_counts = final_results[final_results['anomaly_ind'] == 1].groupby('Purpose').size()
    anomaly_counts_df = anomaly_counts.reset_index(name='anomaly_count')
    print(anomaly_counts_df)
else:
    print("final_results dataframe is not defined.")

