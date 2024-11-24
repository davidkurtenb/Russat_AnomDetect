################################################################
#           PROD CODE - UDL
################################################################

import spacetrack.operators as op
from spacetrack import SpaceTrackClient
import pickle
import pandas as pd
import requests, base64
import numpy as np
import os

def udl_tle_to_par(cntry_nm, output_dir, st_un, st_pw, udl_un, udl_pw):
#######################################################################
# Inputs: cntry_nm = 3 letter conuntry code for TLEs you want to pull, example Russia = 'CIS'
#         output_dir = directory to save final pickle and paraquet to
#         st_un, st_pw = SpaceTrack Username (st_un) & password (st_pw)
#         udl_un, udl_pw = UDL Username (udl_un) & password (udl_pw)
# Outputs: {cntry_nm}_satcat.pkl = Satellite Catalog data from Space track
#          {cntry_nm}_tle_data.parquet = Paraquet file containing dataframe of all TLEs
# Notes: Code is set to only query TLEs from February 2012 to current. Date chosen for 10 year look prior to Ukrain invasion
#        To change date modidfy the epoch in url line (see UDL documentation)
#######################################################################

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

    basicAuth = "Basic " + base64.b64encode((f"{udl_un}:{udl_pw}").encode('utf-8')).decode("ascii")

    tle_df = pd.DataFrame()

    for index,value in enumerate(norad_ids):   
        percent_complete= index/len(norad_ids)*100
        print(f"************* Querying UDL {index} out of {len(norad_ids)}: {percent_complete}% complete********************")

        url = f"https://unifieddatalibrary.com/udl/elset/history?epoch=%3E2012-02-01T00:00:00.000000Z&satNo={value}"
        result = requests.get(url, headers={'Authorization':basicAuth}, verify=False)
        df = pd.DataFrame(result.json())
        tle_df = pd.concat([tle_df, df],ignore_index = True)

    tle_df.to_parquet(os.path.join(output_dir,f'{cntry_nm}_tle_data.parquet'), index=False)

with open(r'C:\Users\dk412\Desktop\spacetrackcreds.txt', 'r') as f:
    content = f.read()
st_un = content.split(",")[0].strip()
st_pw = content.split(",")[1].strip()
udl_un = content.split(",")[2].strip()
udl_pw =content.split(",")[3].strip()

output_dir = r'C:\Users\dk412\Desktop\David\Python Projects\RusSat\output'

udl_tle_to_par('CIS', output_dir, st_un, st_pw, udl_un, udl_pw)
