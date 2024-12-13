Applying Deep Learning to Pattern Mining of Sequential Data and Anomaly Detection of Russian Satellite Activity Prior to Military Action 

Project Summary
We apply deep learning techniques for anomaly detection and sequential pattern mining to analyze activity of Russian owned resident space objects (RSO) prior to the Ukraine invasion and assess the results for any findings that can be used as indications or warnings of aggressive military behavior for future conflicts. This research looks to assess the existence of statistically significant changes in Russian RSO pattern of life/pattern of behavior (PoL/PoB) using Keplerian elements that are publicly available. Additional analysis looks at RSO activity during an active combat period by sampling two-line element (TLE) data after the invasion date. To capture the nuance and unique characteristics of each RSO an individual model was trained for each observed space object. The total number of autoencoder models trained and used for inference is 2,544 which is based on a select number of space objects that met the defined criteria to be included in the research. To further drive explainability of the research, space objects were analyzed based on their categorized mission set and purpose. This helps to create understandable results that can be extended to establish a generalized profile of space activity leading up to aggressive military actions.  

Data Description (*/dataout_HPC)
CIS_satcat.pkl = Satellite catalog data from SpaceTrack for all CIS owned objects (Includes Debris ~25K object)
DEV_*.parquet = TLE data & XYZ
MODEL_*.parquet = TLE data, XYZ, SatCat, & UCS
model_test_train.parquet = TLE data, XYZ, SatCat, UCS, & Outlier

notebooks: These are Jupyter notebooks used to develop

output: Directory to push outputs to when testing and building

prod_code: Final python scripts that were sent to Beocat
 



 
 
