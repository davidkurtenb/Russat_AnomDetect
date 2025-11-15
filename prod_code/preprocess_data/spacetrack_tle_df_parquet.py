################################################################
#           PROD CODE
################################################################
import spacetrack.operators as op
from spacetrack import SpaceTrackClient
import pickle
import pandas as pd
import os


def spacetrack_tle_cntry(cntry_code,st_un, st_pw, output_dir):
    st = SpaceTrackClient(identity=f'{st_un}', password=f'{st_pw}')

    satcat_query = st.satcat(country=cntry_code, current='Y')

    cis_norad_ids = []

    for e in satcat_query:
        norad_num = int(e.get('NORAD_CAT_ID'))
        cis_norad_ids.append(norad_num)

    #lst_pull = [61445,51511,61447]
    
    tle_hist = st.tle(norad_cat_id=cis_norad_ids, orderby='epoch', limit=None, format='tle').split('\n')

    def parse_line1(line1):
        return {
            'satellite_num': line1[2:7].strip(),
            'intl_des': line1[9:17].strip(),
            'epoch_year': line1[18:20].strip(),
            'epoch_day': line1[20:32].strip(),
            'first_derivative_mean_motion': line1[33:43].strip(),
            'second_derivative_mean_motion': line1[44:52].strip(),
            'bstar_drag': line1[53:61].strip(),
            'ephemeris_type': line1[62].strip(),
            'element_set_number': line1[64:68].strip(),
            'checksum_1': line1[68].strip()
        }

    def parse_line2(line2):
        return {
            'satellite_num': line2[2:7].strip(),
            'inclination': line2[8:16].strip(),  
            'right_ascension': line2[17:25].strip(),  
            'eccentricity': line2[26:33].strip(),  
            'argument_of_perigee': line2[34:42].strip(), 
            'mean_anomaly': line2[43:51].strip(),  
            'mean_motion': line2[52:63].strip(),  
            'revolution_number': line2[63:68].strip(),  
            'checksum_2': line2[68].strip()
        }

    tle_pairs = [(tle_hist[i], tle_hist[i+1]) for i in range(0, len(tle_hist)-1, 2)]

    tle_data = []
    for line1, line2 in tle_pairs:
        tle_dict = {}
        tle_dict.update(parse_line1(line1))
        tle_dict.update(parse_line2(line2))
        tle_data.append(tle_dict)

    df = pd.DataFrame(tle_data)

    df.to_parquet(os.path.join(output_dir,'tle_data.parquet'), index=False)

with open(r'C:\Users\dk412\Desktop\spacetrackcreds.txt', 'r') as f:
    content = f.read()
st_un = content.split(",")[0].strip()
st_pw = content.split(",")[1].strip()

output_dir = r'C:\Users\dk412\Desktop\David\Python Projects\RusSat\output'
spacetrack_tle_cntry('CIS', st_un, st_pw, output_dir)