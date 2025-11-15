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
import random
import torch.nn.functional as F
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


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
        
        # Training Loss
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(losses)  # Now losses is available
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
#                             KAN - AD
#################################################################

class TimeSeriesAnomalyDataset(torch.utils.data.Dataset):
    def __init__(self, time_series, labels, window_size=20, step_size=10, transform=None):
        self.time_series = time_series
        self.labels = labels
        self.window_size = window_size
        self.step_size = step_size
        self.transform = transform
        self.sample_indices = list(range(0, len(time_series) - window_size + 1, step_size))

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        if idx >= len(self.sample_indices) or idx < 0:
            raise IndexError(f"Index {idx} out of range for sample_indices of length {len(self.sample_indices)}")
        i = self.sample_indices[idx]
        window = self.time_series[i : i + self.window_size]
        window_labels = self.labels[i : i + self.window_size]
        x = torch.tensor(window, dtype=torch.float).unsqueeze(-1)  # Shape: [window_size, 1]
        y = torch.tensor(1.0 if window_labels.any() else 0.0, dtype=torch.float)
        return x, y

def stratified_split(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    labels = [y.item() for _, y in dataset]
    train_val_indices, test_indices = train_test_split(np.arange(len(labels)), test_size=test_ratio, stratify=labels, random_state=seed)
    val_relative_ratio = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=val_relative_ratio, stratify=[labels[i] for i in train_val_indices], random_state=seed)
    return train_indices, val_indices, test_indices

class ResampledDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = [torch.tensor(x, dtype=torch.float).view(-1, 1) for x in X]
        self.y = [torch.tensor(label, dtype=torch.float) for label in y]
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def data_loader_build(dataset, args, df, use_smote, x):
    train_indices, val_indices, test_indices = stratified_split(dataset, seed=args.seed)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    X_train = [x.numpy().flatten() for x, _ in train_dataset]
    y_train = [int(y.item()) for _, y in train_dataset]
    
    # Count total nulls in X_train
    X_train_null_count = sum(np.isnan(x).sum() for x in X_train)
    print(f"\nTotal nulls in X_train: {X_train_null_count}")    # Count total nulls in X_train
    y_train_null_count = sum(np.isnan(y).sum() for y in y_train)
    print(f"Total nulls in y_train: {y_train_null_count}")

    if use_smote == True:
        smote = SMOTE(random_state=args.seed)

        if df[f'outlier_iqr_{x}'].nunique() > 1:
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        else:
            X_resampled = X_train
            y_resampled = y_train
    else:
        X_resampled = X_train
        y_resampled = y_train           

    balanced_train_dataset = ResampledDataset(X_resampled, y_resampled)

    train_loader = torch.utils.data.DataLoader(balanced_train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, balanced_train_dataset, val_dataset, train_dataset, test_dataset, test_indices
"""
CHANGE UPATED LOSS FUNCTION
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * ((1 - pt) ** self.gamma) * BCE_loss
        return F_loss.mean()
"""    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)

class NaiveFourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize=50, addbias=True):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        self.fouriercoeffs = nn.Parameter(torch.randn(2 * gridsize, inputdim, outdim) / (np.sqrt(inputdim) * np.sqrt(gridsize)))
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(outdim))

    def forward(self, x):
        batch_size, window_size, inputdim = x.size()
        k = torch.arange(1, self.gridsize + 1, device=x.device).float()
        k = k.view(1, 1, 1, self.gridsize)
        x_expanded = x.unsqueeze(-1)
        angles = x_expanded * k * np.pi
        sin_features = torch.sin(angles)
        cos_features = torch.cos(angles)
        features = torch.cat([sin_features, cos_features], dim=-1)
        features = features.view(batch_size * window_size, inputdim, -1)
        coeffs = self.fouriercoeffs
        y = torch.einsum('bik,kio->bo', features, coeffs)
        y = y.view(batch_size, window_size, self.outdim)
        if self.addbias:
            y += self.bias
        return y

class KAN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, grid_feat, num_layers, use_bias=True, dropout=0.3):
        super(KAN, self).__init__()
        self.num_layers = num_layers
        self.positional_encoding = PositionalEncoding(hidden_feat)
        self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        self.bn_in = nn.BatchNorm1d(hidden_feat)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))
            self.bns.append(nn.BatchNorm1d(hidden_feat))
        self.lin_out = nn.Linear(hidden_feat, out_feat, bias=use_bias)

    def forward(self, x):
        batch_size, window_size, _ = x.size()
        x = self.lin_in(x)
        x = self.bn_in(x.view(-1, x.size(-1))).view(batch_size, window_size, -1)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer, bn in zip(self.layers, self.bns):
            x = layer(x)
            x = bn(x.view(-1, x.size(-1))).view(batch_size, window_size, -1)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = self.dropout(x)
        x = x.mean(dim=1)
        x = self.lin_out(x).squeeze()
        return x

def train_kan_model(args,                       
                    model, 
                    train_loader, 
                    best_val_f1,
                    optimizer, 
                    criterion,
                    scheduler,
                    balanced_train_dataset,
                    val_loader,
                    val_dataset,
                    patience,
                    sat_save_dir,
                    i,
                    x):
    
    ################## CHANGE ADD
    #patience_counter = 0
    #optimal_threshold = 0.5
    ##################

    for epoch in range(args.epochs):
        # Training Phase
        model.train()
        total_loss = 0
        total_acc = 0
        #total_preds_pos = 0  # Monitor number of positive predictions
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)
            
            optimizer.zero_grad()
            out = model(x_batch)  # Output shape: [batch_size]
            loss = criterion(out, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * x_batch.size(0)
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).float()
            acc = (preds == y_batch).float().mean().item()
            total_acc += acc * x_batch.size(0)
            #total_preds_pos += preds.sum().item()
        avg_loss = total_loss / len(balanced_train_dataset)
        avg_acc = total_acc / len(balanced_train_dataset)

        #print(f"Epoch {epoch+1}, Training Positive Predictions: {total_preds_pos}")

        # Validation Phase
        model.eval()
        val_loss = 0
        val_acc = 0
        all_true = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(args.device)
                y_batch = y_batch.to(args.device)
                out = model(x_batch)
                loss = criterion(out, y_batch)
                val_loss += loss.item() * x_batch.size(0)
                probs = torch.sigmoid(out)
                preds = (probs > 0.5).float()
                acc = (preds == y_batch).float().mean().item()
                val_acc += acc * x_batch.size(0)
                all_true.extend(y_batch.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        avg_val_loss = val_loss / len(val_dataset)
        avg_val_acc = val_acc / len(val_dataset)
        precision, recall, f1, roc_auc_val = evaluate_metrics(all_true, all_preds, all_probs)
        
        
        #CHANGE DROPPED
        # Find Optimal Threshold
        current_threshold, current_f1 = find_optimal_threshold(all_probs, all_true)

        print(
            f"Epoch: {epoch+1:04d}, "
            f"Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
            f"F1: {f1:.4f}, ROC AUC: {roc_auc_val:.4f}, "
            f"Optimal Threshold: {current_threshold:.4f}, Val F1: {current_f1:.4f}"
        )
        """
        ############### CHAGNE ADD
        if len(set(all_true)) > 1:  # Only if we have both classes
            optimal_threshold, current_f1 = find_optimal_threshold(all_probs, all_true)
            all_preds = (np.array(all_probs) > optimal_threshold).astype(int)
            precision = precision_score(all_true, all_preds, zero_division=0)
            recall = recall_score(all_true, all_preds, zero_division=0)
            f1 = f1_score(all_true, all_preds, zero_division=0)
            
            try:
                roc_auc_val = roc_auc_score(all_true, all_probs)
            except:
                roc_auc_val = 0.0
        else:
            optimal_threshold = 0.5
            current_f1 = f1 = precision = recall = roc_auc_val = 0.0

        print(f"Epoch: {epoch+1:04d}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
              f"Threshold: {optimal_threshold:.4f}")
        ###############
        """
        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Early Stopping
        if f1 > best_val_f1:
            best_val_f1 = f1
            patience_counter = 0
            optimal_threshold = current_threshold  # Update optimal threshold CHANGE Dropped
            # Save the best model
            torch.save(model.state_dict(), os.path.join(sat_save_dir, f"{i}_kan_model_{x}.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    return model, preds

# Define evaluation metrics
def evaluate_metrics(true_labels, pred_labels, pred_probs):
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    if len(set(true_labels))> 1:
        roc_auc_val = roc_auc_score(true_labels, pred_probs)
    else:
        roc_auc_val = float('nan')
    return precision, recall, f1, roc_auc_val

# Function to determine optimal threshold based on validation set
def find_optimal_threshold(probs, labels):
    precision_vals, recall_vals, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)

    ############ CHANGE ADD
    valid_indices = ~np.isnan(f1_scores)
    if not np.any(valid_indices):
        return 0.5, 0.0
    
    f1_scores = f1_scores[valid_indices]
    thresholds = thresholds[valid_indices] if len(thresholds) > len(f1_scores) else np.append(thresholds, 0.5)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    optimal_f1 = f1_scores[optimal_idx]
    ####################

    return optimal_threshold, optimal_f1
    """
    CHANGE DROPPED THIS
    optimal_idx = np.argmax(f1_scores)
    if optimal_idx < len(thresholds):
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = 0.5  # Default threshold
    optimal_f1 = f1_scores[optimal_idx]
    return optimal_threshold, optimal_f1
    """

def model_eval(model, 
               test_loader,
               args,
               criterion,
               optimal_threshold,
               test_dataset):
    
    # Test the model using the optimal threshold
    model.eval()
    test_loss = 0
    test_acc = 0
    all_true_test = []
    all_preds_test = []
    all_probs_test = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)
            out = model(x_batch)
            loss = criterion(out, y_batch)
            test_loss += loss.item() * x_batch.size(0)
            probs = torch.sigmoid(out)
            #preds = (probs > optimal_threshold).float()
            preds = (probs > 0.1).float() #CHANGE OF PREDS 
            acc = (preds == y_batch).float().mean().item()
            test_acc += acc * x_batch.size(0)
            all_true_test.extend(y_batch.cpu().numpy())
            all_preds_test.extend(preds.cpu().numpy())
            all_probs_test.extend(probs.cpu().numpy())
    avg_test_loss = test_loss / len(test_dataset)
    avg_test_acc = test_acc / len(test_dataset)
    precision, recall, f1, roc_auc_val = evaluate_metrics(
        all_true_test, all_preds_test, all_probs_test)
    
    print(
    f"\nTest Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}, "
    f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
    f"F1: {f1:.4f}, ROC AUC: {roc_auc_val:.4f}")

    return all_true_test, all_preds_test, all_probs_test

# Visualization of anomalies
def plot_anomalies(time_series, labels, preds, i, e, base_save_dir, start=0, end=1000 ):
    plt.figure(figsize=(15, 5))
    plt.plot(time_series[start:end], label="Time Series")
    plt.scatter(
        np.arange(start, end)[labels[start:end] == 1],
        time_series[start:end][labels[start:end] == 1],
        color="red",
        label="True Anomalies",
    )
    plt.scatter(
        np.arange(start, end)[preds[start:end] == 1],
        time_series[start:end][preds[start:end] == 1],
        color="orange",
        marker="x",
        label="Predicted Anomalies",
    )
    plt.legend()
    plt.title(f"Anomaly Detection for NORAD ID {i} - {e}")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Value")
    plt.savefig(os.path.join(base_save_dir, f'plots/anom_plot_{i}_{e}.png'))
    plt.close()

# Aggregate predictions on the test set
def aggregate_predictions(indices, preds, window_size, total_length):
    aggregated = np.zeros(total_length, dtype=float)
    counts = np.zeros(total_length, dtype=float)
    for idx, pred in zip(indices, preds):
        start = idx
        end = idx + window_size
        if end > total_length:
            end = total_length
        aggregated[start:end] += pred
        counts[start:end] += 1
    counts[counts == 0] = 1
    averaged = aggregated / counts
    return (averaged > 0.5).astype(int)

# Additional Visualization: ROC and Precision-Recall Curves
def plot_metrics(true_labels, pred_probs, i, e, base_save_dir):
    # ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc_val = auc(fpr, tpr)

    # Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(true_labels, pred_probs)
    pr_auc_val = auc(recall_vals, precision_vals)

    plt.figure(figsize=(12, 5))

    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_val:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic (ROC) Curve for NORAD ID {i} - {e}")
    plt.legend()

    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall_vals, precision_vals, label=f"PR Curve (AUC = {pr_auc_val:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall (PR) Curve for NORAD ID {i} - {e}")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(base_save_dir, f'plots/roc_prc_plot_{i}_{e}.png'))
    plt.close()
