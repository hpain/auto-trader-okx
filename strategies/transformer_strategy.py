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

class TimeSeriesGRU(nn.Module if HAS_TORCH else object):
    """
    Switching to GRU (Gated Recurrent Unit) which is more robust for smaller datasets (<100k samples).
    Transformer was overfitting/underfitting (Val Loss > Baseline).
    """
    def __init__(self, input_dim, d_model=64, num_layers=2, dropout=0.2):
        super(TimeSeriesGRU, self).__init__()
        
        # 1. Input Projection (Optional, but helps to map features to hidden dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. GRU Layer
        # batch_first=True: Input is (Batch, Seq, Feature)
        self.rnn = nn.GRU(
            input_size=d_model, 
            hidden_size=d_model, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 3. Output Head
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )


    def forward(self, src):
        # src: [Batch, Seq_Len, Features]
        
        # Project Input
        x = self.input_proj(src)
        
        # RNN Forward
        # out: (Batch, Seq, Hidden), hn: (Layers, Batch, Hidden)
        out, _ = self.rnn(x)
        
        # Take the last time step
        last_step = out[:, -1, :]
        
        # Project to target
        prediction = self.decoder(last_step)
        return prediction # Return raw logits

class TransformerStrategy:
    """
    Deep Learning Strategy (Currently using GRU for stability on small datasets).
    Hardware Agnostic: Runs on CPU or CUDA.
    Memory Optimized: Uses Lazy Loading for low RAM environments.
    """
    def __init__(self, strategy_name="Transformer_v2", window_size=60, features=None, buy_threshold=0.60, sell_threshold=0.40, dropout=0.2):
        self.logger = logging.getLogger(__name__)
        self.strategy_name = strategy_name
        self.window_size = window_size
        self.features = features or ['close', 'volume'] # Should be set dynamically
        
        # Dynamic Thresholds
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.logger.info(f"Thresholds -> Buy: {self.buy_threshold}, Sell: {self.sell_threshold}")
        self.dropout = dropout
        
        self.device = 'cpu'
        self.model = None
        self.scaler = None
        
        if HAS_TORCH:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.logger.info(f"TransformerStrategy initialized on DEVICE: {self.device.upper()}")
        else:
            self.logger.warning("PyTorch not installed. Strategy disabled.")

    def build_model(self, input_dim, pos_weight=None):
        if not HAS_TORCH: return
        # Use the new GRU model (renamed class or swapped implementation)
        self.model = TimeSeriesGRU(input_dim=input_dim, dropout=self.dropout).to(self.device).float()
        
        # Increased LR to 0.001 to help model escape baseline
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-3)
        
        # Scheduler to reduce LR when loss plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
        # Binary Cross Entropy with Logits
        # Use pos_weight to handle class imbalance (makes model focus more on minority class)
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight]).to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            self.logger.info(f"Using pos_weight={pos_weight:.2f} for class imbalance")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        self.logger.info(f"Model built with Input Dim: {input_dim}")

    def train_model(self, df: pd.DataFrame, target_col='target_up', epochs=10, batch_size=32, val_df=None):
        """
        Train the model on provided DataFrame using Lazy Loader.
        Supports optional validation set for early stopping.
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
        
        # Prepare validation loader if provided
        val_loader = None
        if val_df is not None:
            try:
                val_data = val_df[self.features]
                val_target = val_df[target_col]
                val_loader = create_lazy_loader(val_data, val_target, self.window_size, batch_size)
            except Exception as e:
                self.logger.warning(f"Could not create validation loader: {e}")
        
        # Early stopping setup
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_state = None
            
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
                
                # Validation evaluation
                val_loss_str = ""
                avg_val_loss = None
                if val_loader is not None and len(val_loader) > 0:
                    self.model.eval()
                    val_loss = 0
                    val_count = 0
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            batch_X = batch_X.to(self.device).float()
                            batch_y = batch_y.to(self.device).float().unsqueeze(1)
                            outputs = self.model(batch_X)
                            val_loss += self.criterion(outputs, batch_y).item()
                            val_count += 1
                    
                    if val_count > 0:
                        avg_val_loss = val_loss / val_count
                        val_loss_str = f" - Val Loss: {avg_val_loss:.4f}"
                        
                        # Early stopping check
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            patience_counter = 0
                            best_state = self.model.state_dict().copy()
                        else:
                            patience_counter += 1
                            if patience_counter >= patience:
                                self.logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                                if best_state is not None:
                                    self.model.load_state_dict(best_state)
                                break
                    
                    self.model.train()
                
                # Step the scheduler
                if hasattr(self, 'scheduler'):
                    if avg_val_loss is not None:
                        self.scheduler.step(avg_val_loss)
                    else:
                        self.scheduler.step(avg_loss)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}{val_loss_str} - LR: {current_lr:.6f}")
            
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
                # Ensure we only pick numeric columns (simple heuristic)
                numeric_df = df.select_dtypes(include=[np.number]) 
                # Try to match input dim blindly if features not set
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
