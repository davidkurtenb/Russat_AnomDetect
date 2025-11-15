import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter
from typing import List, Dict, Tuple
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest


#################################################################
#               Ploty Plots Classes
#################################################################

class TLEVisualizer:
    @staticmethod
    def format_timestamps(timestamps, indices=None):
        if indices is not None:
            timestamps = timestamps[indices]
        if not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.DatetimeIndex(timestamps)
        return [ts.strftime('%Y-%m-%d %H:%M') for ts in timestamps]

    @staticmethod
    def plot_training_loss(losses: List[float], norad_id: int, plot_save_dir: str, figsize=(10, 5)):
        plt.figure(figsize=figsize)
        plt.plot(losses, label='Training Loss')
        plt.title(f'Training Autoencoder Training Loss Over Time {norad_id}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(plot_save_dir, f'TRAIN_trainingloss_{norad_id}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_reconstruction_errors(reconstruction_errors: np.ndarray, 
                                feature_names: List[str],
                                norad_id: int,
                                plot_save_dir: str,
                                figsize=(8, 5)):
        plt.figure(figsize=figsize)
        sns.boxplot(data=reconstruction_errors)
        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
        plt.title(f'Training Distribution of Reconstruction Errors by Feature {norad_id}')
        plt.xlabel('Feature')
        plt.ylabel('Reconstruction Error')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_save_dir, f'TRAIN_reconstruction_errors_{norad_id}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    @classmethod
    def plot_anomaly_heatmap(cls, 
                            anomaly_details: Dict, 
                            feature_names: List[str],
                            timestamps,
                            norad_id: int,
                            plot_save_dir: str,
                            max_samples: int = 50,
                            figsize=(15, 8)):
        anomalous_indices = np.where(np.any(anomaly_details['feature_anomalies'], axis=1))[0]
        
        if len(anomalous_indices) == 0:
            print("No anomalies found to visualize")
            return
        
        if len(anomalous_indices) > max_samples:
            anomalous_indices = anomalous_indices[:max_samples]
        
        z_scores = anomaly_details['z_scores'][anomalous_indices]
        time_labels = cls.format_timestamps(timestamps, anomalous_indices)
        
        plt.figure(figsize=figsize)
        sns.heatmap(z_scores, 
                xticklabels=feature_names,
                yticklabels=time_labels,
                cmap='RdYlBu_r',
                center=0)
        plt.title(f'Training Anomaly Z-Scores Heatmap {norad_id}')
        plt.xlabel('Feature')
        plt.ylabel('Timestamp')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_save_dir, f'TRAIN_anom_heatmap_{norad_id}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_feature_timeline(data: np.ndarray,
                            anomalies: np.ndarray,
                            feature_names: List[str],
                            timestamps,
                            norad_id: int,
                            plot_save_dir: str,
                            n_features: int = 6,
                            figsize=(15, 12)):
        if not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.DatetimeIndex(timestamps)
            
        n_features = min(n_features, data.shape[1])
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_features, 1, figure=fig)
        
        date_formatter = DateFormatter('%Y-%m-%d')
        
        for i in range(n_features):
            ax = fig.add_subplot(gs[i, 0])
            
            ax.plot(timestamps[~anomalies], data[~anomalies, i], 'b.', 
                alpha=0.5, label='Normal', markersize=4)
            
            if np.any(anomalies):
                ax.plot(timestamps[anomalies], data[anomalies, i], 'r.', 
                    label='Anomaly', alpha=0.7, markersize=6)
            
            ax.set_ylabel(feature_names[i])
            ax.xaxis.set_major_formatter(date_formatter)
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend()
            if i == n_features - 1:
                ax.set_xlabel('Time')
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle(f'Training Feature Timeline with Anomalies {norad_id}')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_save_dir, f'TRAIN_feat_timeline_{norad_id}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    @classmethod
    def plot_summary(cls, 
                    data: np.ndarray,
                    anomaly_details: Dict,
                    anomalies: np.ndarray, 
                    feature_names: List[str],
                    timestamps,
                    losses: List[float],  
                    norad_id: int,
                    plot_save_dir: str,
                    figsize=(15, 8)):

        if not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.DatetimeIndex(timestamps)
            
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 3, figure=fig)
        
        #training Loss
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(losses)  
        ax1.set_title(f'Training Loss {norad_id}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        #Reconstruction Error
        ax2 = fig.add_subplot(gs[0, 1:])
        sns.boxplot(data=anomaly_details['reconstruction_errors'], ax=ax2)
        ax2.set_xticklabels(feature_names, rotation=45, ha='right')
        ax2.set_title(f'Training Reconstruction Error Distribution {norad_id}')
        ax2.grid(True, alpha=0.3)
        
        #Anomaly Timeline
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.plot(timestamps[~anomalies], np.zeros(np.sum(~anomalies)), 'b.',
                label='Normal', alpha=0.5, markersize=4)
        if np.any(anomalies):
            ax3.plot(timestamps[anomalies], np.ones(np.sum(anomalies)), 'r.',
                    label='Anomaly', alpha=0.7, markersize=6)
        ax3.set_title(f'Training Anomaly Timeline {norad_id}')
        ax3.set_ylabel('Anomaly (1) / Normal (0)')
        ax3.set_xlabel('Time')
        ax3.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        #Feature Distributiona
        ax4 = fig.add_subplot(gs[1, 2])
        feature_anomaly_counts = anomaly_details['feature_anomalies'].sum(axis=0)
        ax4.bar(range(len(feature_names)), feature_anomaly_counts)
        ax4.set_xticks(range(len(feature_names)))
        ax4.set_xticklabels(feature_names, rotation=45, ha='right')
        ax4.set_title(f'Training Anomalies per Feature {norad_id}')
        ax4.set_ylabel('Count')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_save_dir, f'TRAIN_summary_{norad_id}.png'), dpi=300, bbox_inches='tight')
        #plt.show()

#################################################################
#               Model Classes
#################################################################

#################################################################
#               Autoencoder
#################################################################
class TLEEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super(TLEEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, latent_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

class TLEDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super(TLEDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.LayerNorm(16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        return self.decoder(x)

class TLEAnomalyDetector:
    
    def __init__(self, input_dim: int, latent_dim: int = 3):
        print(f"Initializing detector with input_dim={input_dim}, latent_dim={latent_dim}")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.encoder = TLEEncoder(input_dim, latent_dim).to(self.device)
        self.decoder = TLEDecoder(latent_dim, input_dim).to(self.device)
        self.scaler = StandardScaler()
        
    def save_model(self, path: str, NORAD_ID_NUM: int):
        os.makedirs(path, exist_ok=True)
        
        torch.save({
            'encoder_state': self.encoder.state_dict(),
            'decoder_state': self.decoder.state_dict(),
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim
        }, os.path.join(path, f'{NORAD_ID_NUM}_model.pt'))
        
        joblib.dump(self.scaler, os.path.join(path, f'{NORAD_ID_NUM}_scaler.pkl'))
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str, NORAD_ID_NUM: int) -> 'TLEAnomalyDetector':
        checkpoint = torch.load(os.path.join(path, f'{NORAD_ID_NUM}_model.pt'))
        
        detector = cls(input_dim=checkpoint['input_dim'], 
                     latent_dim=checkpoint['latent_dim'])
        
        detector.encoder.load_state_dict(checkpoint['encoder_state'])
        detector.decoder.load_state_dict(checkpoint['decoder_state'])
        
        detector.scaler = joblib.load(os.path.join(path, f'{NORAD_ID_NUM}_scaler.pkl'))
        
        detector.encoder.eval()
        detector.decoder.eval()
        
        print(f"Model loaded from {path}")
        return detector
    
    @staticmethod    
    def visualize_results(data: np.ndarray, 
                         anomalies: np.ndarray,
                         anomaly_details: Dict,
                         feature_names: List[str],
                         timestamps: pd.DatetimeIndex,
                         losses: List[float],
                         NORAD_ID_NUM: int,
                         plot_save_dir: str):
        visualizer = TLEVisualizer()
        
        print("Generating visualization summary")
        visualizer.plot_summary(data, anomaly_details, anomalies, 
                              feature_names, timestamps, losses,
                              NORAD_ID_NUM, plot_save_dir)
        
        print("\nGenerating detailed visualizations")
        visualizer.plot_training_loss(losses, NORAD_ID_NUM, plot_save_dir)
        visualizer.plot_reconstruction_errors(anomaly_details['reconstruction_errors'], 
                                           feature_names, NORAD_ID_NUM, plot_save_dir)
        visualizer.plot_anomaly_heatmap(anomaly_details, feature_names, timestamps,
                                      NORAD_ID_NUM, plot_save_dir)
        visualizer.plot_feature_timeline(data, anomalies, feature_names, timestamps,
                                       NORAD_ID_NUM, plot_save_dir)

    def _validate_input_data(self, data: np.ndarray) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(data)}")
        
        if len(data.shape) != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")
            
        if data.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {data.shape[1]}")
            
        if np.isnan(data).any():
            raise ValueError("Input data contains NaN values")
            
        if np.isinf(data).any():
            raise ValueError("Input data contains infinite values")

    def train(self, data: np.ndarray, epochs: int = 150, batch_size: int = 32) -> List[float]:
        print(f"Starting training with data shape: {data.shape}")
        self._validate_input_data(data)
        
        scaled_data = self.scaler.fit_transform(data)
        train_data = torch.FloatTensor(scaled_data).to(self.device)
        optimizer = optim.Adam(list(self.encoder.parameters()) + 
                             list(self.decoder.parameters()))
        criterion = nn.MSELoss(reduction='none')
        losses = []
        
        n_batches = (len(train_data) + batch_size - 1) // batch_size
        print(f"Training with {n_batches} batches per epoch")
        
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:min(i + batch_size, len(train_data))]
                
                latent = self.encoder(batch)
                reconstructed = self.decoder(latent)
                
                loss = criterion(reconstructed, batch).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / n_batches
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
                
        return losses
    
    def detect_anomalies(self, data: np.ndarray, threshold_sigma: float = 2) -> Tuple[np.ndarray, Dict]:
        print(f"Detecting anomalies in data with shape: {data.shape}")
        self._validate_input_data(data)
        
        scaled_data = self.scaler.transform(data)
        test_data = torch.FloatTensor(scaled_data).to(self.device)
        
        with torch.no_grad():
            latent = self.encoder(test_data)
            reconstructed = self.decoder(latent)
        
        reconstruction_errors = torch.abs(test_data - reconstructed).cpu().numpy()
        error_mean = np.mean(reconstruction_errors, axis=0)
        error_std = np.std(reconstruction_errors, axis=0)
        error_std = np.where(error_std == 0, 1e-10, error_std)
        
        z_scores = (reconstruction_errors - error_mean) / error_std
        anomalies = np.any(np.abs(z_scores) > threshold_sigma, axis=1)
        
        anomaly_details = {
            'z_scores': z_scores,
            'feature_anomalies': np.abs(z_scores) > threshold_sigma,
            'reconstruction_errors': reconstruction_errors
        }
        
        print(f"Found {np.sum(anomalies)} anomalous samples")
        return anomalies, anomaly_details
    
    def explain_anomalies(self, anomaly_details: Dict, feature_names: List[str] = None) -> List[Dict]:
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(anomaly_details['z_scores'].shape[1])]
        
        if len(feature_names) != anomaly_details['z_scores'].shape[1]:
            raise ValueError(f"Expected {anomaly_details['z_scores'].shape[1]} feature names, got {len(feature_names)}")
            
        print(f"Generating explanations for shape {anomaly_details['z_scores'].shape}")
        explanations = []
        
        for i in range(len(anomaly_details['z_scores'])):
            if np.any(anomaly_details['feature_anomalies'][i]):
                anomalous_features = []
                for j, is_anomaly in enumerate(anomaly_details['feature_anomalies'][i]):
                    if is_anomaly:
                        anomalous_features.append({
                            'feature': feature_names[j],
                            'z_score': float(anomaly_details['z_scores'][i, j]),  
                            'reconstruction_error': float(anomaly_details['reconstruction_errors'][i, j])
                        })
                
                if anomalous_features:
                    explanations.append({
                        'sample_index': int(i),  
                        'anomalous_features': anomalous_features
                    })
                    
        print(f"Generated {len(explanations)} anomaly explanations")
        return explanations    
def run_anomaly_detection_pipeline(df: pd.DataFrame, 
                                 feature_names: List[str] = None,
                                 model_path: str = None,
                                 should_train: bool = False,
                                 NORAD_ID_NUM: int = None,
                                 plot_save_dir: str = None,
                                 print_plots: bool = False):

    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        data = df.values
        timestamps = df.index
        input_dim = data.shape[1]
        
        if should_train:
            print("Training new model...")
            detector = TLEAnomalyDetector(input_dim=input_dim, latent_dim = 3)
            losses = detector.train(data, epochs=150)
            
            if model_path:
                detector.save_model(model_path, NORAD_ID_NUM)
        else:
            if not model_path:
                raise ValueError("model_path must be provided when should_train=False")
            print("Loading existing model")
            detector = TLEAnomalyDetector.load_model(model_path, NORAD_ID_NUM)
        
        print("Detecting anomalies")
        anomalies, anomaly_details = detector.detect_anomalies(data, 2)
        
        if feature_names is None:
            feature_names = df.columns.tolist()
            
        print("Generating explanations")
        explanations = detector.explain_anomalies(anomaly_details, feature_names)
        
        if print_plots:
            print("Generating visualizations")
            detector.visualize_results(data, anomalies, anomaly_details, 
                                    feature_names, timestamps, 
                                    losses if should_train else [],
                                    NORAD_ID_NUM, plot_save_dir)
        
        return detector, anomalies, explanations, timestamps, anomaly_details
        
    except Exception as e:
        print(f"Error during anomaly detection: {str(e)}")
        raise


#################################################################
#               Variational Autoencoder
#################################################################
class TLEVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 3):
        super(TLEVAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.LeakyReLU(0.2)
        )
        
        self.mean_layer = nn.Linear(hidden_dim//2, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim//2, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def encode(self, x):
        x = self.encoder(x)
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
        
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

class TLEVariationalAnomalyDetector:
    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 4):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TLEVAE(input_dim, hidden_dim, latent_dim).to(self.device)
        self.scaler = StandardScaler()

    def save_model(self, path: str, NORAD_ID_NUM: int):
        os.makedirs(path, exist_ok=True)
        
        torch.save({
            'model_state': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim
        }, os.path.join(path, f'{NORAD_ID_NUM}_vae_model.pt'))
        joblib.dump(self.scaler, os.path.join(path, f'{NORAD_ID_NUM}_vae_scaler.pkl'))
        
        print(f"VAE model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str, NORAD_ID_NUM: int) -> 'TLEVariationalAnomalyDetector':
        checkpoint = torch.load(os.path.join(path, f'{NORAD_ID_NUM}_vae_model.pt'))
        
        detector = cls(input_dim=checkpoint['input_dim'],
                     hidden_dim=checkpoint['hidden_dim'],
                     latent_dim=checkpoint['latent_dim'])
        
        detector.model.load_state_dict(checkpoint['model_state'])
        detector.scaler = joblib.load(os.path.join(path, f'{NORAD_ID_NUM}_vae_scaler.pkl'))
        detector.model.eval()
        
        print(f"VAE model loaded from {path}")
        return detector
        
    def loss_function(self, x, x_hat, mean, logvar):
        recon_loss = nn.MSELoss(reduction='sum')(x_hat, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss
    
    def train(self, data: np.ndarray, epochs: int = 150, batch_size: int = 32) -> List[float]:
        print(f"Starting VAE training with data shape: {data.shape}")
        scaled_data = self.scaler.fit_transform(data)
        train_data = torch.FloatTensor(scaled_data).to(self.device)
        
        dataset = TensorDataset(train_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters())
        losses = []

        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (batch,) in enumerate(dataloader):
                optimizer.zero_grad()
                
                x_hat, mean, logvar = self.model(batch)
                
                loss = self.loss_function(batch, x_hat, mean, logvar)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader.dataset)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
                
        return losses
    
    def vae_detect_anomalies(self, data: np.ndarray, threshold_sigma: float = 2) -> Tuple[np.ndarray, Dict]:
        print(f"Detecting anomalies using VAE in data with shape: {data.shape}")
        scaled_data = self.scaler.transform(data)
        test_data = torch.FloatTensor(scaled_data).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            x_hat, mean, logvar = self.model(test_data)
            
        reconstruction_errors = torch.abs(test_data - x_hat).cpu().numpy()
        
        error_mean = np.mean(reconstruction_errors, axis=0)
        error_std = np.std(reconstruction_errors, axis=0)
        error_std = np.where(error_std == 0, 1e-10, error_std)
        z_scores = (reconstruction_errors - error_mean) / error_std
        
        anomalies = np.any(np.abs(z_scores) > threshold_sigma, axis=1)
        
        anomaly_details = {
            'z_scores': z_scores,
            'feature_anomalies': np.abs(z_scores) > threshold_sigma,
            'reconstruction_errors': reconstruction_errors,
            'latent_space': mean.cpu().numpy()  
        }
        
        print(f"Found {np.sum(anomalies)} anomalous samples")
        return anomalies, anomaly_details
    
    def explain_anomalies(self, anomaly_details: Dict, feature_names: List[str] = None) -> List[Dict]:
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(anomaly_details['z_scores'].shape[1])]
        
        if len(feature_names) != anomaly_details['z_scores'].shape[1]:
            raise ValueError(f"Expected {anomaly_details['z_scores'].shape[1]} feature names, got {len(feature_names)}")
            
        print(f"Generating explanations for shape {anomaly_details['z_scores'].shape}")
        explanations = []
        
        for i in range(len(anomaly_details['z_scores'])):
            if np.any(anomaly_details['feature_anomalies'][i]):
                anomalous_features = []
                for j, is_anomaly in enumerate(anomaly_details['feature_anomalies'][i]):
                    if is_anomaly:
                        anomalous_features.append({
                            'feature': feature_names[j],
                            'z_score': float(anomaly_details['z_scores'][i, j]),  
                            'reconstruction_error': float(anomaly_details['reconstruction_errors'][i, j])
                        })
                
                if anomalous_features:
                    explanations.append({
                        'sample_index': int(i),  
                        'anomalous_features': anomalous_features
                    })
                    
        print(f"Generated {len(explanations)} anomaly explanations")
        return explanations   

def run_vae_anomaly_detection(df: pd.DataFrame, 
                            feature_names: List[str] = None,
                            hidden_dim: int = 32,
                            latent_dim: int = 4,
                            epochs: int = 150,
                            batch_size: int = 32,
                            model_path: str = None,
                            should_train: bool = True,
                            NORAD_ID_NUM: int = None):
    
    data = df.values
    input_dim = data.shape[1]
    
    if should_train:
        print("Training new VAE model...")
        detector = TLEVariationalAnomalyDetector(input_dim=input_dim, 
                                               hidden_dim=hidden_dim,
                                               latent_dim=latent_dim)
        losses = detector.train(data, epochs=epochs, batch_size=batch_size)
        
        if model_path:
            detector.save_model(model_path, NORAD_ID_NUM)
    else:
        if not model_path:
            raise ValueError("model_path must be provided when should_train=False")
        print("Loading existing VAE model...")
        detector = TLEVariationalAnomalyDetector.load_model(model_path, NORAD_ID_NUM)
        losses = []
    
    anomalies, anomaly_details = detector.vae_detect_anomalies(data)
    
    return detector, anomalies, anomaly_details, losses

#################################################################
#               AE Anchor
#################################################################
class TLEAEAnchorModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 5):
        super(TLEAEAnchorModel, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim//2, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

class TLEAEAnchorDetector:
    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 4):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TLEAEAnchorModel(input_dim, hidden_dim, latent_dim).to(self.device)
        self.scaler = StandardScaler()

    def save_model(self, path: str, NORAD_ID_NUM: int):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim
        }, os.path.join(path, f'{NORAD_ID_NUM}_aeanchor_model.pt'))
        joblib.dump(self.scaler, os.path.join(path, f'{NORAD_ID_NUM}_aeanchor_scaler.pkl'))
        print(f"AE Anchor model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str, NORAD_ID_NUM: int) -> 'TLEAEAnchorDetector':
        checkpoint = torch.load(os.path.join(path, f'{NORAD_ID_NUM}_eanchor_model.pt'))
        detector = cls(input_dim=checkpoint['input_dim'],
                     hidden_dim=checkpoint['hidden_dim'],
                     latent_dim=checkpoint['latent_dim'])
        detector.model.load_state_dict(checkpoint['model_state'])
        detector.scaler = joblib.load(os.path.join(path, f'{NORAD_ID_NUM}_aeanchor_scaler.pkl'))
        detector.model.eval()
        return detector

    def loss_function(self, x, x_hat, z):
        recon_loss = F.mse_loss(x_hat, x, reduction='none')
        
        #Compute pairwise distances in latent space
        z_dist = torch.cdist(z, z)
        #Get k nearest neighbors
        k = min(3, z.size(0)-1)
        _, nn_idx = torch.topk(z_dist, k+1, largest=False)
        nn_idx = nn_idx[:, 1:]  # Exclude self
        
        #Compute anchor loss
        anchor_loss = 0
        for i in range(z.size(0)):
            nn_distances = z_dist[i, nn_idx[i]]
            anchor_loss += torch.mean(nn_distances)
        
        anchor_loss = anchor_loss / z.size(0)
        
        return recon_loss, anchor_loss

    def train(self, data: np.ndarray, epochs: int = 150, batch_size: int = 32) -> List[float]:
        print(f"Starting AE Anchor training with data shape: {data.shape}")
        scaled_data = self.scaler.fit_transform(data)
        train_data = torch.FloatTensor(scaled_data).to(self.device)
        
        dataset = TensorDataset(train_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters())
        losses = []
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (batch,) in enumerate(dataloader):
                optimizer.zero_grad()
                
                x_hat, z = self.model(batch)
                recon_loss, anchor_loss = self.loss_function(batch, x_hat, z)
                
                loss = torch.mean(recon_loss) + 0.1 * anchor_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
                
        return losses

    def aeanchor_detect_anomalies(self, data: np.ndarray, threshold_sigma: float = 2) -> Tuple[np.ndarray, Dict]:
        print(f"Detecting anomalies using AE Anchor in data with shape: {data.shape}")
        scaled_data = self.scaler.transform(data)
        test_data = torch.FloatTensor(scaled_data).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            x_hat, z = self.model(test_data)
            
        reconstruction_errors = torch.abs(test_data - x_hat).cpu().numpy()
        
        error_mean = np.mean(reconstruction_errors, axis=0)
        error_std = np.std(reconstruction_errors, axis=0)
        error_std = np.where(error_std == 0, 1e-10, error_std)
        z_scores = (reconstruction_errors - error_mean) / error_std
        
        anomalies = np.any(np.abs(z_scores) > threshold_sigma, axis=1)
        
        anomaly_details = {
            'z_scores': z_scores,
            'feature_anomalies': np.abs(z_scores) > threshold_sigma,
            'reconstruction_errors': reconstruction_errors,
            'latent_space': z.cpu().numpy()
        }
        
        print(f"Found {np.sum(anomalies)} anomalous samples")
        return anomalies, anomaly_details

    def explain_anomalies(self, anomaly_details: Dict, feature_names: List[str] = None) -> List[Dict]:
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(anomaly_details['z_scores'].shape[1])]
        
        if len(feature_names) != anomaly_details['z_scores'].shape[1]:
            raise ValueError(f"Expected {anomaly_details['z_scores'].shape[1]} feature names, got {len(feature_names)}")
            
        print(f"Generating explanations for shape {anomaly_details['z_scores'].shape}")
        explanations = []
        
        for i in range(len(anomaly_details['z_scores'])):
            if np.any(anomaly_details['feature_anomalies'][i]):
                anomalous_features = []
                for j, is_anomaly in enumerate(anomaly_details['feature_anomalies'][i]):
                    if is_anomaly:
                        anomalous_features.append({
                            'feature': feature_names[j],
                            'z_score': float(anomaly_details['z_scores'][i, j]),  
                            'reconstruction_error': float(anomaly_details['reconstruction_errors'][i, j])
                        })
                
                if anomalous_features:
                    explanations.append({
                        'sample_index': int(i),  
                        'anomalous_features': anomalous_features
                    })
                    
        print(f"Generated {len(explanations)} anomaly explanations")
        return explanations    

def run_aeanchor_anomaly_detection(df: pd.DataFrame, 
                            feature_names: List[str] = None,
                            hidden_dim: int = 32,
                            latent_dim: int = 4,
                            epochs: int = 150,
                            batch_size: int = 64,
                            model_path: str = None,
                            should_train: bool = True,
                            NORAD_ID_NUM: int = None,
                            plot_save_dir: str = None,
                            print_plots: bool = False):
    
    data = df.values
    timestamps = df.index
    input_dim = data.shape[1]
    
    if should_train:
        print("Training new AE Anchor model...")
        detector = TLEAEAnchorDetector(input_dim=input_dim, 
                                hidden_dim=hidden_dim,
                                latent_dim=latent_dim)
        losses = detector.train(data, epochs=epochs, batch_size=batch_size)
        
        if model_path:
            detector.save_model(model_path, NORAD_ID_NUM)
    else:
        if not model_path:
            raise ValueError("model_path must be provided when should_train=False")
        print("Loading existing AE Anchor model...")
        detector = TLEAEAnchorDetector.load_model(model_path, NORAD_ID_NUM)
        losses = []
    
    anomalies, anomaly_details = detector.aeanchor_detect_anomalies(data)

    if feature_names is None:
        feature_names = df.columns.tolist()

    print("Generating explanations...")
    explanations = detector.explain_anomalies(anomaly_details, feature_names)
    
    if print_plots:
        print(f"Generating visualizations to {plot_save_dir}")
        TLEAnomalyDetector.visualize_results(data, 
                                             anomalies, 
                                             anomaly_details, 
                                             feature_names, 
                                             timestamps, 
                                             losses if should_train else [],
                                             NORAD_ID_NUM, 
                                             plot_save_dir)
    
    return detector, anomalies, explanations, timestamps, anomaly_details, losses

#################################################################
#               Isolation Forest
#################################################################

class TLEIsolationForest:
    def __init__(self, contamination='auto', n_estimators=100, max_samples='auto', random_state=42):
        self.model = IsolationForest(
            contamination=contamination, 
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        
    def save_model(self, path: str, NORAD_ID_NUM: int):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model, os.path.join(path, f'{NORAD_ID_NUM}_iforest.pkl'))
        joblib.dump(self.scaler, os.path.join(path, f'{NORAD_ID_NUM}_iforest_scaler.pkl'))
        
    @classmethod
    def load_model(cls, path: str, NORAD_ID_NUM: int) -> 'TLEIsolationForest':
        detector = cls()
        detector.model = joblib.load(os.path.join(path, f'{NORAD_ID_NUM}_iforest.pkl'))
        detector.scaler = joblib.load(os.path.join(path, f'{NORAD_ID_NUM}_iforest_scaler.pkl'))
        return detector

    def train(self, data: np.ndarray):
        scaled_data = self.scaler.fit_transform(data)
        self.model.fit(scaled_data)
        return []  
        
    def detect_anomalies(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        scaled_data = self.scaler.transform(data)
        scores = self.model.score_samples(scaled_data)
        predictions = self.model.predict(scaled_data)
        
        anomalies = predictions == -1
        z_scores = -scores  
        feature_importance = np.abs(np.array([tree.feature_importances_ 
                                            for tree in self.model.estimators_])).mean(axis=0)
        
        anomaly_details = {
            'z_scores': np.tile(z_scores[:, np.newaxis], (1, data.shape[1])),
            'feature_anomalies': np.tile(anomalies[:, np.newaxis], (1, data.shape[1])),
            'reconstruction_errors': np.abs(z_scores[:, np.newaxis] * feature_importance),
            'feature_importance': feature_importance
        }
        
        return anomalies, anomaly_details

def run_iforest_detection(df: pd.DataFrame,
                         contamination='auto',
                         n_estimators=100,
                         max_samples='auto',
                         model_path: str = None,
                         should_train: bool = True,
                         NORAD_ID_NUM: int = None):
    data = df.values

    if should_train:
        detector = TLEIsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples
        )

        losses = detector.train(data)
        
        if model_path:
            detector.save_model(model_path, NORAD_ID_NUM)
    else:
        if not model_path:
            raise ValueError("model_path must be provided when should_train=False")
        detector = TLEIsolationForest.load_model(model_path, NORAD_ID_NUM)
        losses = []
    
    anomalies, anomaly_details = detector.detect_anomalies(data)
    
    return detector, anomalies, anomaly_details, losses