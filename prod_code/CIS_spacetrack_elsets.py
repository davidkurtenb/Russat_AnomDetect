####################################################################
#                     PROD CODE- Space Track TLE Fetch
####################################################################

# Imports
import requests
from requests.exceptions import ReadTimeout, RequestException
from typing import Any, Dict, Optional
import logging
from datetime import datetime, timedelta
import pandas as pd
import glob
import os
from tqdm import tqdm
import pickle
import time
import spacetrack.operators as op
from spacetrack import SpaceTrackClient

# UDL Data to get teh list of NORAD IDs that have TLEs during Feb-April 2022
parquet_file = "/homes/dkurtenb/projects/russat/output/udl_CIS_data.parquet"
df = pd.read_parquet(parquet_file)

filtered_df = df[(df['year'] == 2022) & ((df['month']==2) | (df['month']==3) | (df['month']==4))]
sat_lst = list(filtered_df['satNo'])
norad_id_datefilter_lst = list(set(sat_lst))
norad_id_datefilter_lst.sort()

#Spacetrack Satcat data to get list of norad id that have object type ROCKET BODY, PAYLOAD, or UNKNOWN 
with open('/homes/dkurtenb/projects/russat/output/CIS_satcat.pkl', 'rb') as f:  # 'rb' means read binary mode
    data = pickle.load(f)

norad_list = [item['NORAD_CAT_ID'] for item in data 
             if item['OBJECT_TYPE'] in ['ROCKET BODY', 'PAYLOAD', 'UNKNOWN']]
norad_id_typefilter_lst = list(set(norad_list))
norad_id_typefilter_lst.sort()
norad_id_typefilter_lst = [int(x) for x in norad_id_typefilter_lst]

# Norad IDs filtered by Date (2022 Feb-April) & Type (ROCKET BODY, PAYLOAD, or UNKNOWN)
nord_id_typeDate_filter = list(set(norad_id_datefilter_lst) & set(norad_id_typefilter_lst))
nord_id_typeDate_filter.sort()

#SpaceTrack Fetch
with open('/homes/dkurtenb/projects/russat/spacetrackcreds.txt', 'r') as f:
    content = f.read()
st_un = content.split(",")[0].strip()
st_pw = content.split(",")[1].strip()
udl_un = content.split(",")[2].strip()
udl_pw = content.split(",")[3].strip()

st = SpaceTrackClient(identity=f'{st_un}', password=f'{st_pw}')

full_lst = []
try:
    norad_ids = ','.join(str(i) for i in nord_id_typeDate_filter)
    query = st.tle(norad_cat_id=norad_ids, orderby='epoch', limit=None, format='tle')
    tles = query.split('\n')
    full_lst.extend([tles[i:i+2] for i in range(0, len(tles), 2)])
    
    time.sleep(5)
    
except Exception as e:
    print(f"Error occurred: {e}")
    time.sleep(900)

# Push full_lst to dataframe with parsed TLE data
def parse_scientific_notation(string):
    try:
        if string.strip() == '+00000-0' or string.strip() == '+00000+0':
            return 0.0
        
        mantissa = float(string[0] + '.' + string[1:6])
        exponent = int(string[6:8])
        return mantissa * (10 ** exponent)
    except:
        return 0.0

def parse_tle_to_df(tle_list):
    data = []
    
    for tle in tle_list:
        # Skip if not a proper TLE pair
        if not isinstance(tle, list) or len(tle) != 2:
            print(f"Skipping invalid TLE pair: {tle}")
            continue
            
        line1, line2 = tle
        line1_data = {
            'line1': line1,
            'line2': line2,
            # Line 1 elements
            'catalog_number': int(line1[2:7]),
            'classification': line1[7],
            'launch_year': line1[9:11],
            'launch_number': line1[11:14],
            'launch_piece': line1[14:17].strip(),
            'epoch_year': int(line1[18:20]),
            'epoch_day': float(line1[20:32]),
            'mean_motion_dot': float(line1[33:43]),
            'mean_motion_ddot': parse_scientific_notation(line1[44:52] + line1[52:54]),
            'bstar': parse_scientific_notation(line1[53:61] + line1[61:63]),
            'ephemeris_type': int(line1[63]) if line1[63].strip() else 0,
            'element_number': int(line1[64:68]) if line1[64:68].strip() else 0,
            # Line 2 elements
            'satellite_number': int(line2[2:7]),
            'inclination': float(line2[8:16]),
            'ra_of_asc_node': float(line2[17:25]),
            'eccentricity': float('0.' + line2[26:33]),
            'arg_of_perigee': float(line2[34:42]),
            'mean_anomaly': float(line2[43:51]),
            'mean_motion': float(line2[52:63]),
            'rev_at_epoch': int(line2[63:68]) if line2[63:68].strip() else 0
        }
        data.append(line1_data)

    return pd.DataFrame(data)

df = parse_tle_to_df(full_lst)

if not df.empty:
    df.to_parquet("/homes/dkurtenb/projects/russat/output/spacetrack_tle_df.parquet",engine='pyarrow', compression = 'gzip', index =True)