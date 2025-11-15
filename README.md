# Russian Satellite Anomaly Detection

Deep learning-based anchor based anomaly detection system for analyzing Russian-owned satellite activity patterns before and during the Ukraine invasion.
https://amostech.space/year/2025/applying-deep-learning-to-anomaly-detection-of-russian-satellite-activity-for-indications-prior-to-military-activity/

## Overview

We apply deep learning techniques for anomaly detection to analyze activity of Russian-owned resident space objects (RSO) prior to the Ukraine invasion and assess the results for any findings that can be used as indications or warnings of aggressive military behavior for future conflicts. This research looks to assess the existence of statistically significant changes in Russian RSO pattern of life/pattern of behavior (PoL/PoB) using Keplerian elements that are publicly available. This research looks at statistical and deep learning approaches to assess anomalous activity. The deep learning method uses and compares a variational autoencoder (VAE), traditional autoencoder (AE), an anchor loss based autoencoder (Anchor AE), isolation forest (IF), and Kolmogorov Aronold Network (KAN) approach to establish a baseline of on-orbit activity based on a five-year data sample. The primary investigation period focuses on the six months leading up to the invasion date of February 24, 2022. Additional analysis looks at RSO activity during an active combat period by sampling two-line element (TLE) data after the invasion date. The deep learning models identify anomalies based on reconstruction errors that surpass a threshold of two standard deviations. To capture the nuance and unique characteristics of each RSO an individual model was trained for each observed space object. The total number of autoencoder models trained and used for inference is 2,544 which is based on a select number of space objects that met the defined criteria to be included in the research. The research strived to prioritize explainability and interpretability of the model results thus each observation was assessed for anomalous behavior of the individual six orbital elements versus analyzing the input data as a single monolithic observation. The results demonstrate not just statistically significant anomalies of Russian RSO activity but detail anomalous findings to the individual Keplerian element. To drive explainability, objects were analyzed based on their categorized mission. This helps to create understandable results that can be extended to establish a generalized profile of space activity leading up to aggressive military actions. 

### Key Features

- **2,544 individual autoencoder models** trained on unique space objects
- **Multi-source data integration** combining TLE data, orbital coordinates, and satellite catalogs
- **Mission-based categorization** for enhanced interpretability and explainability
- **Pre/post-invasion analysis** comparing normal operations with wartime activity patterns
- **Pattern of Life (PoL) profiling** for different satellite mission types

## Research Questions

1. Can anomalous satellite behavior patterns precede military action?
2. Do different mission types (communications, reconnaissance, navigation) exhibit distinct behavioral signatures during conflict?
3. Can we develop generalized profiles for space activity associated with military operations?

## Repository Structure
```
├── data/
│   └── model_dev/          # Training and validation datasets
├── prod_code/
│   ├── model_dev/          # Model development scripts
│   ├── model_inf/          # Inference pipeline
│   ├── post_build_analysis/# Analysis and visualization tools
│   └── preprocess_data/    # Data cleaning and feature engineering
├── artifacts/              # Trained models and outputs
├── output/                 # Experimental results and logs
└── README.md
```

## Data Sources

### Primary Datasets

**Available in `/data` and [Google Drive](https://drive.google.com/drive/folders/107m_W9zvD3yVL7DjbPA4rmlVvP4kWwEM)**

- **`CIS_satcat.pkl`**: SpaceTrack satellite catalog for CIS-owned objects
  - ~25,000 objects including active satellites and debris
  
- **`MODEL_*.parquet`**: Integrated datasets containing:
  - Two-Line Element (TLE) orbital parameters
  - Cartesian (XYZ) coordinates
  - Satellite catalog metadata
  - Union of Concerned Scientists (UCS) satellite database

### Data Collection

All orbital data sourced from [Space-Track.org](https://www.space-track.org/), providing public Keplerian elements for tracked space objects.

## Methodology

1. **Data Preprocessing**: Extract and clean TLE data for Russian-owned satellites
2. **Feature Engineering**: Convert orbital elements to behavioral features
3. **Model Training**: Train individual LSTM autoencoders per satellite
4. **Anomaly Detection**: Identify deviations from normal operational patterns
5. **Temporal Analysis**: Compare pre-invasion vs. wartime activity
6. **Mission Categorization**: Group findings by satellite purpose

## Technical Stack

- **Python 3.x**
- **Deep Learning**: TensorFlow/Keras autoencoders
- **Data Processing**: pandas, NumPy
- **Deployment**: High-Performance Computing (Beocat cluster)

## Getting Started

### Prerequisites
```bash
python >= 3.8
tensorflow >= 2.x
pandas
numpy
scikit-learn
```

### Installation
```bash
# Clone repository
git clone https://github.com/davidkurtenb/Russat_AnomDetect.git
cd Russat_AnomDetect

# Install dependencies
pip install -r requirements.txt
```

### Usage
```bash
# Preprocess data
python prod_code/preprocess_data/main.py

# Train models
python prod_code/model_dev/train.py

# Run inference
python prod_code/model_inf/detect_anomalies.py
```

## Research Team

**Kansas State University**
- David Kurtenbach
- Megan Manly
- Zach Metzinger

## Citation

If you use this work in your research, please cite:
```bibtex
@misc{kurtenbach2024russat,
  author = {Kurtenbach, David and Manly, Megan and Metzinger, Zach},
  title = {Russian Satellite Anomaly Detection},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/davidkurtenb/Russat_AnomDetect}
}
```

## License

[Specify license - e.g., MIT, Apache 2.0, GPL-3.0]

## Acknowledgments

- Space-Track.org for providing public orbital data
- Kansas State University for computational resources
- Union of Concerned Scientists for satellite database

## Contact

For questions or collaboration inquiries, please open an issue or contact the research team.

---

**Note**: This research uses only publicly available data and is conducted for academic purposes.
