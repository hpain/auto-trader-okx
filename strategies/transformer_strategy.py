import logging
import numpy as np
import pandas as pd
import json
import os
import pickle
import math
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

class PositionalEncoding(nn.Module if HAS_TORCH else object):
    """
    Injects some information about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings, so that the two can be summed.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [Batch, Seq_Len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module if HAS_TORCH else object):
    """
    Lightweight Transformer for Time Series Forecasting.
    
    IMPORTANT: This is a simplified version designed to prevent overfitting
    when training with Triple Barrier labels on ~48k samples.
    
    Original (overfitting): d_model=128, num_layers=3, dropout=0.2
    New (regularized): d_model=32, num_layers=1, dropout=0.5
    
    Input: (Batch, Seq_Len, Features)
    Output: (Batch, 1) -> Binary Classification via Logits
    """
    def __init__(self, input_dim, d_model=32, nhead=2, num_layers=1, dropout=0.5):
        super(TimeSeriesTransformer, self).__init__()
        
        # 1. Input Projection with Dropout
        self.input_dropout = nn.Dropout(dropout)
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 3. Single Transformer Encoder Layer (minimal complexity)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2,  # Smaller FF layer
            batch_first=True, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # 4. Simple Output Head (direct projection, no hidden layer)
        self.output_dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(d_model, 1)
        
        self.input_dim = input_dim
        self.d_model = d_model

    def forward(self, src):
        # src: [Batch, Seq_Len, Features]
        
        # Input dropout for regularization
        x = self.input_dropout(src)
        
        # Embed inputs
        x = self.embedding(x)  # [Batch, Seq_Len, d_model]
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # Global Average Pooling
        x = torch.mean(x, dim=1)
        
        # Output with dropout
        x = self.output_dropout(x)
        prediction = self.decoder(x)
        return prediction

class TransformerStrategy:
    """
    Transformer-based Trading Strategy (Phase 2 Upgrade).
    Hardware Agnostic: Runs on CPU or CUDA.
    Memory Optimized: Uses Lazy Loading for low RAM environments.
    """
    def __init__(self, strategy_name="Transformer_v2", window_size=60, features=None, buy_threshold=0.60, sell_threshold=0.40):
        self.logger = logging.getLogger(__name__)
        self.strategy_name = strategy_name
        self.window_size = window_size
        self.features = features or ['close', 'volume'] # Should be set dynamically
        
        # Dynamic Thresholds
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.logger.info(f"Thresholds -> Buy: {self.buy_threshold}, Sell: {self.sell_threshold}")
        
        self.device = 'cpu'
        self.model = None
        self.scaler = None
        
        if HAS_TORCH:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.logger.info(f"TransformerStrategy initialized on DEVICE: {self.device.upper()}")
        else:
            self.logger.warning("PyTorch not installed. Strategy disabled.")

    def build_model(self, input_dim):
        if not HAS_TORCH: return
        self.model = TimeSeriesTransformer(input_dim=input_dim).to(self.device).float()
        
        # Use AdamW for better regularization
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0003, weight_decay=1e-3)
        
        # Scheduler to reduce LR when loss plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
        # Binary Cross Entropy with Logits (Combined Sigmoid + BCELoss for stability)
        # Using pos_weight to handle class imbalance if needed (future upgrade)
        self.criterion = nn.BCEWithLogitsLoss() 
        
        self.logger.info(f"Model built with Input Dim: {input_dim}")

    def train_model(self, df: pd.DataFrame, target_col='target_up', epochs=10, batch_size=32):
        """
        Train the model on provided DataFrame using Lazy Loader.
        """
        if not HAS_TORCH or self.model is None:
            self.logger.warning("Cannot train: Torch missing or model not built.")
            return
            
        # Prepare Data
        # Filter features
        try:
            data = df[self.features]
        except KeyError as e:
            self.logger.error(f"Training features missing in DF: {e}")
            return

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
                
                # Gradient Clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
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

    def generate_signal(self, df: pd.DataFrame, symbol: str = "") -> int:
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
                    self.logger.warning(f"Feature mismatch: Filling {len(missing)} missing features with training mean (neutral).")
                    # impute missing features with mean from scaler to get 0 after normalization
                    if hasattr(self.scaler, 'mean_'):
                        feature_map = dict(zip(required_features, self.scaler.mean_))
                        for m in missing:
                             # Fill with mean so transform -> 0
                            df[m] = feature_map.get(m, 0.0) 
                    else:
                         # Fallback if no mean_ (unlikely for StandardScaler)
                         for m in missing:
                            df[m] = 0.0
                
                self.features = required_features
            
            # 2. Select and Scale Data
            try:
                # Select the exact features
                feature_data = df[self.features].iloc[-self.window_size:]
                
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
            try:
                # Ensure we only pick numeric columns
                numeric_df = df.select_dtypes(include=[np.number])
                # Limit to input dim
                if self.model.input_dim <= len(numeric_df.columns):
                     target_cols = numeric_df.columns.tolist()[:self.model.input_dim]
                else:
                     target_cols = numeric_df.columns.tolist()
                
                scaled_window = df[target_cols].iloc[-self.window_size:].values
            except KeyError:
                return 0

        # Predict
        self.model.eval()

        # NAN SANITIZATION
        if np.isnan(scaled_window).any():
            self.logger.warning("Input data contains NaNs after scaling. Replacing with 0.0.")
            scaled_window = np.nan_to_num(scaled_window, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            with torch.no_grad():
                input_tensor = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(self.device)
                logits = self.model(input_tensor)
                prob = torch.sigmoid(logits).item() 
            
            if np.isnan(prob):
                 self.logger.error("Model predicted NaN! Defaulting to 0.5")
                 prob = 0.5

            symbol_tag = f" ({symbol})" if symbol else ""
            self.logger.info(f"Transformer Prediction Prob{symbol_tag}: {prob:.4f}")
        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            prob = 0.5
        
        if prob > self.buy_threshold:
            return 1 # Buy
        elif prob < self.sell_threshold:
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
