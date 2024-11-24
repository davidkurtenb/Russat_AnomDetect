################################################################
#           PROD CODE - UDL
################################################################

import spacetrack.operators as op
from spacetrack import SpaceTrackClient
import pickle
import cudf  # Replace pandas with cudf
import requests, base64
import numpy as np
import os

def udl_tle_to_par(cntry_nm, output_dir, st_un, st_pw, udl_un, udl_pw):
    st = SpaceTrackClient(identity= f"{st_un}", password= f"{st_pw}")
    sat_cat = st.satcat(country=f'{cntry_nm}', current='Y')

    os.makedirs(output_dir, exist_ok=True)

    print("************* Generating Norad ID list ********************")
    with open(os.path.join(output_dir,f'{cntry_nm}_satcat.pkl'), 'wb') as file:
        pickle.dump(sat_cat, file)

    norad_ids = []

    for e in sat_cat:
        norad_num = int(e.get('NORAD_CAT_ID'))
        norad_ids.append(norad_num)

    norad_ids.sort()

    basicAuth = "Basic " + base64.b64encode((f"{udl_un}:{udl_pw}").encode('utf-8')).decode("ascii")

    tle_df = cudf.DataFrame()  # Initialize as cuDF DataFrame

    for e in norad_ids:    
        test_url = f"https://unifieddatalibrary.com/udl/elset/history?epoch=%3E2016-08-21T00:00:00.000000Z&satNo={e}"
        
        result = requests.get(test_url, headers={'Authorization':basicAuth}, verify=False)
        json_data = result.json()
        if json_data:
            if not isinstance(json_data,list):
                json_data= [json_data]
            df = cudf.DataFrame(json_data)  # Create cuDF DataFrame
            df['satNo'] = e
            print(f"Sat Num: {e} with shape {df.shape}")
            tle_df = cudf.concat([tle_df, df], ignore_index=True)  # Use cuDF concat
        else:
            print(f'{e} has nothing')

    tle_df.to_parquet(os.path.join(output_dir,f'{cntry_nm}_tle_data.parquet'), index=False)

#with open(r'C:\Users\dk412\Desktop\spacetrackcreds.txt', 'r') as f:
with open('/homes/dkurtenb/projects/russat/spacetrackcreds.txt','r') as f:
    content = f.read()
st_un = content.split(",")[0].strip()
st_pw = content.split(",")[1].strip()
udl_un = content.split(",")[2].strip()
udl_pw =content.split(",")[3].strip()

output_dir = '/homes/dkurtenb/projects/russat/output'
udl_tle_to_par('CIS', output_dir, st_un, st_pw, udl_un, udl_pw)