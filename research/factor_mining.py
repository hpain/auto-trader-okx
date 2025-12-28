import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicTransformer
from gplearn.functions import make_function
import joblib
import json
import os
import logging
from typing import List, Dict, Union

# --- PATCH: Fix gplearn compatibility with newer sklearn (missing _validate_data) ---
def _validate_data_patch(self, X, y=None, reset=True, validate_separately=False, **check_params):
    from sklearn.utils.validation import check_X_y, check_array
    if y is not None:
        # gplearn often passes X and y together
        return check_X_y(X, y, **check_params)
    return check_array(X, **check_params)

if not hasattr(SymbolicTransformer, '_validate_data'):
    SymbolicTransformer._validate_data = _validate_data_patch
# -----------------------------------------------------------------------------------

# --- 1. Custom Financial Operators (from gpquant/Qlib) ---

def _ts_rank(data, window=10):
    """Time-series Rank: Rank of the current value in the past window."""
    # Handle numpy array input from gplearn
    if isinstance(data, np.ndarray):
        s = pd.Series(data.flatten())
        return s.rolling(window=window).rank(pct=True).fillna(0.5).values
    return data

def _ts_corr(data1, data2, window=10):
    """Time-series Correlation."""
    if isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
        s1 = pd.Series(data1.flatten())
        s2 = pd.Series(data2.flatten())
        return s1.rolling(window=window).corr(s2).fillna(0).values
    return np.zeros_like(data1)

def _ts_decay_linear(data, window=10):
    """Linear Decay Weighted Average."""
    if isinstance(data, np.ndarray):
        s = pd.Series(data.flatten())
        weights = np.arange(1, window + 1)
        w_sum = weights.sum()
        # Use apply for custom weights (can be slow, but accurate)
        # Faster vectorized approx: EWMA
        return s.rolling(window=window).apply(lambda x: np.dot(x, weights) / w_sum, raw=True).fillna(method='bfill').values
    return data

def _ts_std_dev(data, window=10):
    """Rolling Standard Deviation."""
    if isinstance(data, np.ndarray):
        s = pd.Series(data.flatten())
        return s.rolling(window=window).std().fillna(0).values
    return data

def _ts_min(data, window=10):
    """Rolling Min."""
    if isinstance(data, np.ndarray):
        s = pd.Series(data.flatten())
        return s.rolling(window=window).min().fillna(method='bfill').values
    return data

def _ts_max(data, window=10):
    """Rolling Max."""
    if isinstance(data, np.ndarray):
        s = pd.Series(data.flatten())
        return s.rolling(window=window).max().fillna(method='bfill').values
    return data

# Register functions with gplearn
ts_rank_10 = make_function(function=lambda x: _ts_rank(x, 10), name='ts_rank_10', arity=1)
ts_corr_10 = make_function(function=lambda x, y: _ts_corr(x, y, 10), name='ts_corr_10', arity=2)
ts_decay_10 = make_function(function=lambda x: _ts_decay_linear(x, 10), name='ts_decay_10', arity=1)
ts_std_10 = make_function(function=lambda x: _ts_std_dev(x, 10), name='ts_std_10', arity=1)
ts_min_10 = make_function(function=lambda x: _ts_min(x, 10), name='ts_min_10', arity=1)
ts_max_10 = make_function(function=lambda x: _ts_max(x, 10), name='ts_max_10', arity=1)

# --- 2. Factor Miner Class ---

class FactorMiner:
    def __init__(self, generations=20, population_size=1000, n_components=10, random_state=42):
        self.generations = generations
        self.population_size = population_size
        self.n_components = n_components
        self.random_state = random_state
        
        # Function set including our custom operators
        self.function_set = [
            'add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv',
            ts_rank_10, ts_corr_10, ts_decay_10, ts_std_10, ts_min_10, ts_max_10
        ]
        
        self.gp = SymbolicTransformer(
            generations=generations,
            population_size=population_size,
            hall_of_fame=100,
            n_components=n_components,
            function_set=self.function_set,
            parsimony_coefficient=0.0005,
            max_samples=0.9,
            verbose=1,
            random_state=random_state,
            n_jobs=1  # Parallel might crash on Windows/some backends, start with 1
        )
        self.logger = logging.getLogger(__name__)

    def fit(self, X_train, y_train, feature_names):
        """
        Fit the GP model to discover factors.
        X_train: pandas DataFrame of features
        y_train: target (e.g., Returns)
        feature_names: list of feature names acting as terminals
        """
        self.logger.info(f"Starting Factor Mining (Pop: {self.population_size}, Gens: {self.generations})...")
        
        # gplearn expects numpy array for X. 
        # Column order must match feature_names.
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train[feature_names].values
        
        # y should be next return or target
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        # Ensure no NaNs
        mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
        X_clean = X_train[mask]
        y_clean = y_train[mask]

        self.gp.fit(X_clean, y_clean)
        self.logger.info("Factor Mining Complete.")
        
        return self

    def extract_best_factors(self, feature_names: List[str], output_path="config/mined_factors.json"):
        """Save the top factors to JSON."""
        factors = []
        for i, program in enumerate(self.gp._best_programs):
            if program is None:
                continue
            
            # str(program) returns the LISP-like expression: mul(ts_rank_10(X0), X1)
            # We save the formula and the list of feature names it allows X0, X1 to map to
            factor_def = {
                "name": f"mined_factor_{i}_{int(program.fitness_ * 1000)}",
                "formula": str(program),
                "fitness": float(program.fitness_),
                "base_features": feature_names # Save the mapping context
            }
            factors.append(factor_def)
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(factors, f, indent=4)
        
        self.logger.info(f"Saved {len(factors)} mined factors to {output_path}")
        return factors

    def transform(self, X, feature_names):
        """Transform data using mined factors."""
        if isinstance(X, pd.DataFrame):
            X_values = X[feature_names].values
        else:
            X_values = X
            
        return self.gp.transform(X_values)

# --- 3. Main execution block (can be run standalone) ---
if __name__ == "__main__":
    # Setup logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Needs a real DataFrame to run. Here we just print setup.
    print("FactorMiner initialized. Import and use 'fit' with real data.")
