# Russian Satellite Activity Analysis & Anomaly Detection

## Project Summary
We apply deep learning techniques for anomaly detection and sequential pattern mining to analyze activity of Russian owned resident space objects (RSO) prior to the Ukraine invasion. This research assesses potential indicators or warnings of aggressive military behavior through statistical analysis of RSO pattern of life/pattern of behavior (PoL/PoB) using publicly available Keplerian elements.

Our analysis includes:
- RSO activity monitoring during active combat periods using two-line element (TLE) data post-invasion
- Individual autoencoder models for 2,544 space objects meeting research criteria
- Categorization of space objects by mission set and purpose for enhanced explainability
- Development of generalized profiles for space activity preceding military actions

## Repository Structure

### Data Files (`/dataout_HPC & https://drive.google.com/drive/folders/107m_W9zvD3yVL7DjbPA4rmlVvP4kWwEM`)
- `CIS_satcat.pkl`: SpaceTrack satellite catalog data for CIS-owned objects (~25K objects including debris)
- `MODEL_*.parquet`: Combined data including:
  - TLE data
  - XYZ coordinates
  - SatCat information
  - UCS data

### Directories
- `/output`: Testing and build output directory
- `/prod_code`: Production Python scripts for Beocat deployment

## Technical Details
- Number of models: 2,544 autoencoders
- Data source: Public Keplerian elements
- Analysis period: Pre-Ukraine invasion through active combat period

## Contributors
- David Kurtenbach, Kansas State University
- Megan Manly, Kansas State University
- Zach Metzinger, Kansas State University
## License
[Add license information here]
