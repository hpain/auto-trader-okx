import logging
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Optional, Tuple

# Try importing torch, handle if not installed yet (for dry-run)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    
from features.tensor_loader import create_lazy_loader

class TimeSeriesTransformer(nn.Module if HAS_TORCH else object):
    """
    Simple Transformer for Time Series Forecasting.
    Input: (Batch, Seq_Len, Features)
    Output: (Batch, 1) -> Binary Classification (Up/Down) or Regression
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        # Project input features to d_model size
        # This ensures divisible by nhead and allows any input_dim
        self.embedding = nn.Linear(input_dim, d_model)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # Flatten and project to output
        # Here we use a simple linear head on the last time step
        self.decoder = nn.Linear(d_model, 1) 
        # self.activation = nn.Sigmoid() # Removed for BCEWithLogitsLoss stability

    def forward(self, src):
        # src: [Batch, Seq_Len, Features]
        
        # Embed inputs
        x = self.embedding(src) # [Batch, Seq_Len, d_model]
        
        # Transformer output: [Batch, Seq_Len, d_model]
        output = self.transformer_encoder(x)
        
        # Take the last time step as the summary of the sequence
        last_step = output[:, -1, :]
        
        # Project to target
        # Project to target
        prediction = self.decoder(last_step)
        return prediction # Return raw logits for BCEWithLogitsLoss stability

class TransformerStrategy:
    """
    Transformer-based Trading Strategy.
    Hardware Agnostic: Runs on CPU or CUDA.
    Memory Optimized: Uses Lazy Loading for low RAM environments.
    """
    def __init__(self, strategy_name="Transformer_v1", window_size=60, features=None):
        self.logger = logging.getLogger(__name__)
        self.strategy_name = strategy_name
        self.window_size = window_size
        self.features = features or ['close', 'volume', 'high', 'low'] # Default features
        
        self.device = 'cpu'
        self.model = None
        
        if HAS_TORCH:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.logger.info(f"TransformerStrategy initialized on DEVICE: {self.device.upper()}")
        else:
            self.logger.warning("PyTorch not installed. Strategy disabled.")

    def build_model(self, input_dim):
        if not HAS_TORCH: return
        self.model = TimeSeriesTransformer(input_dim=input_dim).to(self.device).float() # Ensure float32
        # Use AdamW for better regularization
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0005, weight_decay=1e-4)
        # Scheduler to reduce LR when loss plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
        self.criterion = nn.BCEWithLogitsLoss() # More stable than BCELoss
        self.logger.info(f"Model built with Input Dim: {input_dim}")

    def train_model(self, df: pd.DataFrame, target_col='target_up', epochs=5, batch_size=32):
        """
        Train the model on provided DataFrame using Lazy Loader.
        """
        if not HAS_TORCH or self.model is None:
            self.logger.warning("Cannot train: Torch missing or model not built.")
            return
            
        # Prepare Data
        # Filter features
        data = df[self.features]
        target = df[target_col] # Expecting binary 0/1 target
        
        loader = create_lazy_loader(data, target, self.window_size, batch_size)
        
        if loader is None or len(loader) == 0:
            self.logger.warning("Not enough data to train.")
            return
            
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            for batch_X, batch_y in loader:
                # Lazy loader returns tensors directly
                batch_X = batch_X.to(self.device).float()
                batch_y = batch_y.to(self.device).float().unsqueeze(1)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            if batch_count > 0:
                avg_loss = total_loss / batch_count
                # Step the scheduler
                if hasattr(self, 'scheduler'):
                    self.scheduler.step(avg_loss)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")
            
    def load_scaler(self, scaler_path: str):
        """Load the feature scaler from a pickle file."""
        if not os.path.exists(scaler_path):
            self.logger.warning(f"Scaler file not found: {scaler_path}")
            return
        
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.logger.info(f"Scaler loaded from {scaler_path}")
            
            # Auto-detect features from scaler if available
            if hasattr(self.scaler, 'feature_names_in_'):
                self.features = self.scaler.feature_names_in_.tolist()
                self.logger.info(f"Auto-configured {len(self.features)} features from scaler.")
            elif hasattr(self.scaler, 'n_features_in_'):
                # Fallback if names aren't saved (older sklearn), checking dimension later
                self.logger.info(f"Scaler expects {self.scaler.n_features_in_} features.")
        except Exception as e:
            self.logger.error(f"Failed to load scaler: {e}")

    def generate_signal(self, df: pd.DataFrame) -> int:
        """
        Generate Buy(1)/Sell(-1)/Hold(0) signal based on latest window.
        """
        if not HAS_TORCH or self.model is None:
            return 0
            
        if len(df) < self.window_size:
            return 0

        # Dynamic Feature & Scaling Logic
        if hasattr(self, 'scaler') and self.scaler:
            # 1. Update features list if possible
            if hasattr(self.scaler, 'feature_names_in_'):
                # Strict alignment: use exactly the features the scaler expects
                required_features = self.scaler.feature_names_in_.tolist()
                # Check if all required features exist
                missing = [f for f in required_features if f not in df.columns]
                if missing:
                    self.logger.warning(f"DataFrame missing required features: {missing[:3]}...")
                    return 0
                self.features = required_features
            
            # 2. Select and Scale Data
            try:
                # Select the exact features
                feature_data = df[self.features].iloc[-self.window_size:].values
                
                # Check dimension
                if feature_data.shape[1] != self.scaler.n_features_in_:
                     # If previous step relied on self.features list which was wrong
                     self.logger.error(f"Feature count mismatch. Strategy: {feature_data.shape[1]}, Scaler: {self.scaler.n_features_in_}")
                     return 0

                # SCALE THE DATA (Crucial!)
                scaled_window = self.scaler.transform(feature_data)
                
            except Exception as e:
                self.logger.error(f"Error during feature scaling: {e}")
                return 0
        else:
            # Fallback for un-normalized raw data (Not recommended for Transformer)
            self.logger.warning("No scaler loaded! Models trained on scaled data will fail with raw input.")
            # Previous dynamic logic as last resort
            if hasattr(self.model, 'embedding'):
                expected_dim = self.model.embedding.in_features
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                # Only use if counts match exactly
                if len(self.features) != expected_dim and len(numeric_cols) == expected_dim:
                    self.features = numeric_cols
            
            try:
                scaled_window = df[self.features].iloc[-self.window_size:].values
            except KeyError:
                return 0

        # Predict
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits = self.model(input_tensor)
            prob = torch.sigmoid(logits).item() 
            
        self.logger.info(f"Transformer Prediction Prob: {prob:.4f}")
        
        if prob > 0.6:
            return 1 # Buy
        elif prob < 0.4:
            return -1 # Sell
        else:
            return 0

    def save_model(self, path: str):
        """Save model state dict to path."""
        if not HAS_TORCH or self.model is None: return
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str, input_dim: int):
        """Load model state dict from path."""
        if not HAS_TORCH: return
        
        # Ensure model is built first
        if self.model is None:
            self.build_model(input_dim)
            
        if os.path.exists(path):
            # Map location is crucial for GPU -> CPU transfer
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            self.logger.info(f"Model loaded from {path}")
        else:
            self.logger.warning(f"Model file not found: {path}")
