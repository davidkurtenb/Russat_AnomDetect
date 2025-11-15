import pandas as pd
import numpy as np
import os
import datetime as dt
import plotly.graph_objs as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)



##################################################################
#                  DATA PREP FOR ANALYSIS
##################################################################

df = pd.read_csv(r"C:\Users\dk412\Desktop\anom_df_ae_anchor_combat_all_sats_0.csv")

nasa_df = pd.read_excel(r'C:\Users\dk412\Desktop\David\Python Projects\RusSat\data\preproc\NASA_SSCDA_satdiscipline.xlsx')

df_collapsed = nasa_df.groupby(['Spacecraft', 'NSSDCA_ID']).agg({
    'discpline': lambda x: ', '.join(x.unique())
}).reset_index()

df = df.merge(df_collapsed[['NSSDCA_ID', 'discpline']], 
              left_on='INTLDES', 
              right_on='NSSDCA_ID', 
              how='left')
df['mission'] = df['discpline'].fillna(df['Purpose'])

df['mission'] = df['mission'].replace('Space Science', 'technology_applications')
df['mission'] = df['mission'].replace('Unidentified', 'unidentified')
df['mission'] = df['mission'].replace('Communications', 'communications')
df['mission'] = df['mission'].replace('Earth Observation', 'earth_science')
df['mission'] = df['mission'].replace('Earth/Space Observation', 'surveillance_and_other_military')

x =df[['Purpose','mission','Discipline']]
filter = x[(x['mission']=='unidentified')]

norad_id_lst= list(df['NORAD_CAT_ID'].unique())

anom_diff_df=pd.DataFrame()

for n in norad_id_lst:
    norad_df = df[df['NORAD_CAT_ID']==n]
    norad_df = norad_df.sort_values('datetime')
    anom_cnt = norad_df['anom_count']

    if anom_cnt.sum() > 2:
        #print(f"Calculating difference for RSO {n} with anomaly count of {anom_cnt.sum()}")
        norad_df['diff_inclination'] = norad_df['inclination'].diff(-1)
        norad_df['diff_ra_of_asc_node'] = norad_df['ra_of_asc_node'].diff(-1)
        norad_df['diff_eccentricity'] = norad_df['eccentricity'].diff(-1)
        norad_df['diff_arg_of_perigee'] = norad_df['arg_of_perigee'].diff(-1)
        norad_df['diff_mean_anomaly'] = norad_df['mean_anomaly'].diff(-1)
        norad_df['diff_mean_motion'] = norad_df['mean_motion'].diff(-1)

        anom_diff_df = pd.concat([anom_diff_df, norad_df], ignore_index=True)

def add_altitude_column(df, earth_radius=6378.137):
    df['r'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    df['altitude'] = df['r'] - earth_radius
    return df

anom_diff_df = add_altitude_column(anom_diff_df)

conditions = [
    (anom_diff_df['mean_motion'] >= 11) & (anom_diff_df['altitude'] <= 2000),
    (anom_diff_df['mean_motion'] >= 1.5) & (anom_diff_df['mean_motion'] < 11) & (anom_diff_df['altitude'] > 2000) & (anom_diff_df['altitude'] < 35000), 
    (anom_diff_df['mean_motion'] >= 0.9) & (anom_diff_df['mean_motion'] < 1.5),  
    (anom_diff_df['mean_motion'] < 0.9)
]

choices = ['LEO', 'MEO', 'GEO', 'HEO']

anom_diff_df['orbital_regime'] = np.select(conditions, choices, default='Unknown')

#anom_diff_df = anom_diff_df.drop('Operator/Owner', axis=1)
anom_diff_df = anom_diff_df.drop('Users', axis=1)

anom_diff_df['Operator/Owner'] = anom_diff_df['Operator/Owner'].astype('string')


def ecef_to_lla(x, y, z):

   a = 6378137.0
   f = 1/298.257223563
   e2 = 2*f - f**2
   
   lng = np.arctan2(y, x)
   p = np.sqrt(x**2 + y**2)
   lat = np.arctan2(z, p * (1 - e2))
   
   for _ in range(5):
       N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
       alt = p / np.cos(lat) - N
       lat = np.arctan2(z, p * (1 - e2 * N / (N + alt)))
   
   return np.degrees(lat), np.degrees(lng), alt

anom_diff_df[['lat','lng','alt']] = anom_diff_df.apply(lambda row: ecef_to_lla((row['x'], row['y'], row['z']), axis=1, result_type='expand'))

save_dir = r'C:\Users\dk412\Desktop\David\Python Projects\RusSat\output'
anom_diff_df.to_parquet(os.path.join(save_dir, 'anomDet_CIS_aeanchor_02242021_to_02232024.parquet'), compression='gzip')

