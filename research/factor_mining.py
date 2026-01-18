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
# All operators must handle zero vectors and short arrays for gplearn compatibility

def _safe_series(data, window):
    """Convert to series and check if data is valid for rolling window."""
    if isinstance(data, np.ndarray):
        arr = data.flatten()
        if len(arr) < window:
            return None, arr  # Too short, return original
        return pd.Series(arr), arr
    return None, data if isinstance(data, np.ndarray) else np.array([data])

def _ts_rank(data, window=10):
    """Time-series Rank: Rank of the current value in the past window."""
    if isinstance(data, np.ndarray):
        if len(data.flatten()) < window:
            return data.flatten() if len(data.shape) > 0 else np.array([data])
        s = pd.Series(data.flatten())
        result = s.rolling(window=window).rank(pct=True)
        # Fill NaN with 0.5 (neutral rank)
        return np.nan_to_num(result.values, nan=0.5)
    return data

def _ts_corr(data1, data2, window=10):
    """Time-series Correlation."""
    if isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
        d1, d2 = data1.flatten(), data2.flatten()
        n = min(len(d1), len(d2))
        if n < window:
            return np.zeros(n)
        s1 = pd.Series(d1[:n])
        s2 = pd.Series(d2[:n])
        result = s1.rolling(window=window).corr(s2)
        return np.nan_to_num(result.values, nan=0.0)
    return np.zeros_like(data1) if isinstance(data1, np.ndarray) else np.array([0])

def _ts_decay_linear(data, window=10):
    """Linear Decay Weighted Average."""
    if isinstance(data, np.ndarray):
        arr = data.flatten()
        if len(arr) < window:
            return arr
        s = pd.Series(arr)
        weights = np.arange(1, window + 1, dtype=float)
        w_sum = weights.sum()
        result = s.rolling(window=window).apply(lambda x: np.dot(x, weights) / w_sum, raw=True)
        # Fill NaN with the original values
        result = result.fillna(s)
        return result.values
    return data

def _ts_std_dev(data, window=10):
    """Rolling Standard Deviation."""
    if isinstance(data, np.ndarray):
        arr = data.flatten()
        if len(arr) < window:
            return np.zeros_like(arr)
        s = pd.Series(arr)
        result = s.rolling(window=window).std()
        return np.nan_to_num(result.values, nan=0.0)
    return data

def _ts_min(data, window=10):
    """Rolling Min."""
    if isinstance(data, np.ndarray):
        arr = data.flatten()
        if len(arr) < window:
            return arr
        s = pd.Series(arr)
        result = s.rolling(window=window).min()
        # Fill NaN with original values
        return np.where(np.isnan(result.values), arr, result.values)
    return data

def _ts_max(data, window=10):
    """Rolling Max."""
    if isinstance(data, np.ndarray):
        arr = data.flatten()
        if len(arr) < window:
            return arr
        s = pd.Series(arr)
        result = s.rolling(window=window).max()
        # Fill NaN with original values
        return np.where(np.isnan(result.values), arr, result.values)
    return data

# --- Crypto-Specific Operators ---

def _zscore(data, window=24):
    """Z-Score: Standardized deviation from rolling mean. Detects extremes."""
    if isinstance(data, np.ndarray):
        arr = data.flatten()
        if len(arr) < window:
            return np.zeros_like(arr)
        s = pd.Series(arr)
        roll_mean = s.rolling(window=window).mean()
        roll_std = s.rolling(window=window).std()
        # Avoid division by zero
        roll_std = roll_std.replace(0, 1e-9)
        result = (s - roll_mean) / roll_std
        return np.nan_to_num(result.values, nan=0.0)
    return data

def _momentum(data, window=8):
    """Momentum: Rate of change over window. Captures trend strength."""
    if isinstance(data, np.ndarray):
        arr = data.flatten()
        if len(arr) < window + 1:
            return np.zeros_like(arr)
        s = pd.Series(arr)
        mom = s.pct_change(periods=window)
        # Clip extreme values and handle NaN/Inf
        result = np.clip(np.nan_to_num(mom.values, nan=0.0, posinf=1.0, neginf=-1.0), -1, 1)
        return result
    return data

def _delta(data, window=1):
    """Delta: Simple difference. Captures acceleration."""
    if isinstance(data, np.ndarray):
        arr = data.flatten()
        if len(arr) < window + 1:
            return np.zeros_like(arr)
        s = pd.Series(arr)
        result = s.diff(periods=window)
        return np.nan_to_num(result.values, nan=0.0)
    return data

def _sign(data):
    """Sign: Returns -1, 0, or 1. Useful for direction signals."""
    if isinstance(data, np.ndarray):
        return np.sign(data)
    return np.sign(np.array([data]))

def _clip(data, threshold=3.0):
    """Clip: Limit extreme values to threshold std devs."""
    if isinstance(data, np.ndarray):
        arr = data.flatten()
        std = np.std(arr)
        mean = np.mean(arr)
        if std == 0:
            return arr
        return np.clip(arr, mean - threshold * std, mean + threshold * std)
    return data

def _cross_above(data1, data2):
    """Cross Above: 1 when data1 crosses above data2, 0 otherwise."""
    if isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
        d1, d2 = data1.flatten(), data2.flatten()
        n = min(len(d1), len(d2))
        if n < 2:
            return np.zeros(max(len(d1), len(d2)))
        prev_below = np.roll(d1[:n], 1) <= np.roll(d2[:n], 1)
        curr_above = d1[:n] > d2[:n]
        result = (prev_below & curr_above).astype(float)
        result[0] = 0  # First element undefined
        # Pad if needed
        if len(d1) > n:
            result = np.concatenate([result, np.zeros(len(d1) - n)])
        return result
    return np.zeros_like(data1) if isinstance(data1, np.ndarray) else np.array([0])

# Register functions with gplearn - Multiple windows for different time scales
# Note: wrap=False disables gplearn's zero-closure testing which doesn't work with time-series operators
# For 1H crypto data: 10h (intraday), 24h (daily), 72h (3-day)
ts_rank_10 = make_function(function=lambda x: _ts_rank(x, 10), name='ts_rank_10', arity=1, wrap=False)
ts_rank_24 = make_function(function=lambda x: _ts_rank(x, 24), name='ts_rank_24', arity=1, wrap=False)
ts_rank_72 = make_function(function=lambda x: _ts_rank(x, 72), name='ts_rank_72', arity=1, wrap=False)

ts_corr_10 = make_function(function=lambda x, y: _ts_corr(x, y, 10), name='ts_corr_10', arity=2, wrap=False)
ts_corr_24 = make_function(function=lambda x, y: _ts_corr(x, y, 24), name='ts_corr_24', arity=2, wrap=False)

ts_decay_10 = make_function(function=lambda x: _ts_decay_linear(x, 10), name='ts_decay_10', arity=1, wrap=False)
ts_decay_24 = make_function(function=lambda x: _ts_decay_linear(x, 24), name='ts_decay_24', arity=1, wrap=False)

ts_std_10 = make_function(function=lambda x: _ts_std_dev(x, 10), name='ts_std_10', arity=1, wrap=False)
ts_std_24 = make_function(function=lambda x: _ts_std_dev(x, 24), name='ts_std_24', arity=1, wrap=False)

ts_min_10 = make_function(function=lambda x: _ts_min(x, 10), name='ts_min_10', arity=1, wrap=False)
ts_min_24 = make_function(function=lambda x: _ts_min(x, 24), name='ts_min_24', arity=1, wrap=False)

ts_max_10 = make_function(function=lambda x: _ts_max(x, 10), name='ts_max_10', arity=1, wrap=False)
ts_max_24 = make_function(function=lambda x: _ts_max(x, 24), name='ts_max_24', arity=1, wrap=False)

# Crypto-specific operators
zscore_24 = make_function(function=lambda x: _zscore(x, 24), name='zscore_24', arity=1, wrap=False)
zscore_72 = make_function(function=lambda x: _zscore(x, 72), name='zscore_72', arity=1, wrap=False)
momentum_8 = make_function(function=lambda x: _momentum(x, 8), name='momentum_8', arity=1, wrap=False)
momentum_24 = make_function(function=lambda x: _momentum(x, 24), name='momentum_24', arity=1, wrap=False)
delta_1 = make_function(function=lambda x: _delta(x, 1), name='delta_1', arity=1, wrap=False)
delta_8 = make_function(function=lambda x: _delta(x, 8), name='delta_8', arity=1, wrap=False)
sign_op = make_function(function=_sign, name='sign', arity=1, wrap=False)
clip_op = make_function(function=lambda x: _clip(x), name='clip', arity=1, wrap=False)
cross_above = make_function(function=_cross_above, name='cross_above', arity=2, wrap=False)

# --- 2. Factor Miner Class ---

class FactorMiner:
    def __init__(self, generations=20, population_size=1000, n_components=10, random_state=42):
        self.generations = generations
        self.population_size = population_size
        self.n_components = n_components
        self.random_state = random_state
        
        # Function set including our custom operators with multiple windows
        self.function_set = [
            'add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv',
            # Multi-window time series operators
            ts_rank_10, ts_rank_24, ts_rank_72,
            ts_corr_10, ts_corr_24,
            ts_decay_10, ts_decay_24,
            ts_std_10, ts_std_24,
            ts_min_10, ts_min_24,
            ts_max_10, ts_max_24,
            # Crypto-specific operators
            zscore_24, zscore_72,
            momentum_8, momentum_24,
            delta_1, delta_8,
            sign_op, clip_op, cross_above
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
        
        # Save for later evaluation
        self.feature_names = feature_names
        
        # gplearn expects numpy array for X. 
        # Column order must match feature_names.
        if isinstance(X_train, pd.DataFrame):
            self.X_train_df = X_train[feature_names].copy()
            X_train = X_train[feature_names].values
        else:
            self.X_train_df = pd.DataFrame(X_train, columns=feature_names)
        
        # y should be next return or target
        if isinstance(y_train, pd.Series):
            self.y_train = y_train.copy()
            y_train = y_train.values
        else:
            self.y_train = pd.Series(y_train)

        # Ensure no NaNs
        mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
        X_clean = X_train[mask]
        y_clean = y_train[mask]
        
        # Store clean data for evaluation
        self._X_clean = X_clean
        self._y_clean = y_clean
        self._clean_mask = mask

        self.gp.fit(X_clean, y_clean)
        self.logger.info("Factor Mining Complete.")
        
        return self

    def evaluate_factor_quality(self, factor_values: np.ndarray, forward_returns: np.ndarray, 
                                holding_period: int = 12) -> Dict:
        """
        Evaluate a factor using standard quant metrics.
        
        Returns:
            dict with: ic, ic_ir, turnover, monotonicity, composite_score
        """
        from scipy.stats import spearmanr
        
        # Ensure alignment
        n = min(len(factor_values), len(forward_returns))
        factor_values = factor_values[:n]
        forward_returns = forward_returns[:n]
        
        # Handle NaN/Inf
        valid_mask = np.isfinite(factor_values) & np.isfinite(forward_returns)
        if valid_mask.sum() < 100:
            return {'ic': 0, 'ic_ir': 0, 'turnover': 1, 'monotonicity': 0, 'composite_score': 0}
        
        fv = factor_values[valid_mask]
        fr = forward_returns[valid_mask]
        
        # 1. IC (Information Coefficient) - Spearman correlation with future returns
        try:
            ic, _ = spearmanr(fv, fr)
            if np.isnan(ic):
                ic = 0
        except:
            ic = 0
        
        # 2. Rolling IC for IC_IR calculation
        window = min(168, len(fv) // 4)  # 1 week or 1/4 of data
        if window < 24:
            ic_ir = abs(ic)  # Fallback
        else:
            rolling_ics = []
            for i in range(window, len(fv), window // 2):
                try:
                    r_ic, _ = spearmanr(fv[i-window:i], fr[i-window:i])
                    if np.isfinite(r_ic):
                        rolling_ics.append(r_ic)
                except:
                    pass
            if len(rolling_ics) >= 3:
                ic_mean = np.mean(rolling_ics)
                ic_std = np.std(rolling_ics)
                ic_ir = ic_mean / ic_std if ic_std > 0.001 else 0
            else:
                ic_ir = abs(ic)
        
        # 3. Turnover - Average absolute change in factor values (normalized)
        factor_changes = np.abs(np.diff(fv))
        factor_range = np.percentile(fv, 95) - np.percentile(fv, 5)
        turnover = np.mean(factor_changes) / factor_range if factor_range > 0 else 1
        turnover = min(turnover, 1)  # Cap at 1
        
        # 4. Monotonicity - Are quantile returns monotonically ordered?
        try:
            n_quantiles = 5
            quantile_labels = pd.qcut(fv, n_quantiles, labels=False, duplicates='drop')
            quantile_returns = pd.DataFrame({'q': quantile_labels, 'ret': fr}).groupby('q')['ret'].mean()
            
            if len(quantile_returns) >= 3:
                monotonicity, _ = spearmanr(range(len(quantile_returns)), quantile_returns.values)
                if np.isnan(monotonicity):
                    monotonicity = 0
            else:
                monotonicity = 0
        except:
            monotonicity = 0
        
        # 5. Composite Score - Weighted combination
        # High IC, High IC_IR, Low Turnover, High Monotonicity = Good Factor
        composite_score = (
            abs(ic) * 0.3 +           # IC contribution
            abs(ic_ir) * 0.3 +         # Stability contribution  
            (1 - turnover) * 0.2 +     # Low turnover is good
            abs(monotonicity) * 0.2    # Monotonicity contribution
        )
        
        return {
            'ic': round(ic, 4),
            'ic_ir': round(ic_ir, 4),
            'turnover': round(turnover, 4),
            'monotonicity': round(monotonicity, 4),
            'composite_score': round(composite_score, 4)
        }

    def extract_best_factors(self, feature_names: List[str], output_path="config/mined_factors.json",
                            min_composite_score: float = 0.15, max_factors: int = 10):
        """
        Save the top factors to JSON after quality evaluation.
        
        Args:
            feature_names: List of feature names for formula mapping
            output_path: Path to save JSON
            min_composite_score: Minimum quality score to keep a factor
            max_factors: Maximum number of factors to save
        """
        self.logger.info("Evaluating mined factors with IC/IR metrics...")
        
        factors = []
        all_factor_values = []  # For orthogonalization
        
        for i, program in enumerate(self.gp._best_programs):
            if program is None:
                continue
            
            # Compute factor values on training data
            try:
                factor_values = program.execute(self._X_clean)
            except Exception as e:
                self.logger.warning(f"Failed to execute factor {i}: {e}")
                continue
            
            # Evaluate quality
            quality = self.evaluate_factor_quality(factor_values, self._y_clean)
            
            factor_def = {
                "name": f"mined_factor_{i}_{int(quality['composite_score'] * 1000)}",
                "formula": str(program),
                "fitness": float(program.fitness_),
                "quality_metrics": quality,
                "base_features": feature_names
            }
            
            # Filter by quality
            if quality['composite_score'] >= min_composite_score:
                factors.append(factor_def)
                all_factor_values.append(factor_values)
                self.logger.info(f"  Factor {i}: IC={quality['ic']:.3f}, IR={quality['ic_ir']:.3f}, "
                               f"Turnover={quality['turnover']:.3f}, Score={quality['composite_score']:.3f} ✓")
            else:
                self.logger.debug(f"  Factor {i}: Score={quality['composite_score']:.3f} < {min_composite_score} (rejected)")
        
        # --- Orthogonalization: Remove highly correlated factors ---
        if len(factors) > 1 and len(all_factor_values) > 1:
            factors, all_factor_values = self._orthogonalize_factors(factors, all_factor_values, max_factors)
        
        # Sort by composite score and limit
        factors = sorted(factors, key=lambda x: x['quality_metrics']['composite_score'], reverse=True)[:max_factors]
        
        # Save to file
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(factors, f, indent=4)
        
        self.logger.info(f"Saved {len(factors)} quality-filtered factors to {output_path}")
        return factors

    def _orthogonalize_factors(self, factors: List[Dict], factor_values: List[np.ndarray], 
                               max_factors: int = 10) -> tuple:
        """
        Remove highly correlated factors to ensure diversity.
        Uses greedy selection: keep the highest-scoring factor, remove correlated ones.
        
        Returns:
            (filtered_factors, filtered_values)
        """
        self.logger.info("Orthogonalizing factors (removing correlated duplicates)...")
        
        n = len(factors)
        if n <= 1:
            return factors, factor_values
        
        # Build correlation matrix
        factor_matrix = np.column_stack([fv for fv in factor_values])
        valid_rows = np.isfinite(factor_matrix).all(axis=1)
        factor_matrix = factor_matrix[valid_rows]
        
        if len(factor_matrix) < 100:
            return factors, factor_values
        
        corr_matrix = np.corrcoef(factor_matrix.T)
        
        # Greedy selection
        selected_indices = []
        remaining = list(range(n))
        correlation_threshold = 0.7  # Factors with |corr| > 0.7 are considered duplicates
        
        # Sort by composite score
        scores = [f['quality_metrics']['composite_score'] for f in factors]
        sorted_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)
        
        for idx in sorted_indices:
            if idx not in remaining:
                continue
            
            # Add this factor
            selected_indices.append(idx)
            remaining.remove(idx)
            
            if len(selected_indices) >= max_factors:
                break
            
            # Remove highly correlated factors
            to_remove = []
            for other_idx in remaining:
                if abs(corr_matrix[idx, other_idx]) > correlation_threshold:
                    to_remove.append(other_idx)
                    self.logger.debug(f"  Removing factor {other_idx} (corr={corr_matrix[idx, other_idx]:.2f} with factor {idx})")
            
            for r in to_remove:
                remaining.remove(r)
        
        # Filter
        filtered_factors = [factors[i] for i in selected_indices]
        filtered_values = [factor_values[i] for i in selected_indices]
        
        self.logger.info(f"Orthogonalization: {n} -> {len(filtered_factors)} factors (removed {n - len(filtered_factors)} correlated)")
        
        return filtered_factors, filtered_values

    def transform(self, X, feature_names):
        """Transform data using mined factors."""
        if isinstance(X, pd.DataFrame):
            X_values = X[feature_names].values
        else:
            X_values = X
            
        return self.gp.transform(X_values)


# --- 3. Standalone Factor Evaluation Utility ---

def evaluate_existing_factors(factor_df: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    """
    Evaluate quality of existing factors in a DataFrame.
    
    Args:
        factor_df: DataFrame where each column is a factor
        returns: Forward returns series aligned with factor_df
        
    Returns:
        DataFrame with quality metrics for each factor
    """
    from scipy.stats import spearmanr
    
    results = []
    for col in factor_df.columns:
        fv = factor_df[col].values
        fr = returns.values
        
        valid = np.isfinite(fv) & np.isfinite(fr)
        if valid.sum() < 100:
            continue
        
        fv, fr = fv[valid], fr[valid]
        
        try:
            ic, _ = spearmanr(fv, fr)
        except:
            ic = 0
        
        turnover = np.mean(np.abs(np.diff(fv))) / (np.std(fv) + 1e-9)
        
        results.append({
            'factor': col,
            'ic': ic,
            'turnover': min(turnover, 1),
            'score': abs(ic) * (1 - min(turnover, 1))
        })
    
    return pd.DataFrame(results).sort_values('score', ascending=False)


# --- 4. Factor Decay Monitor ---

class FactorDecayMonitor:
    """
    Monitors factor performance over time to detect decay.
    
    Factors discovered through GP tend to have a "half-life" - their predictive
    power decays as the market adapts. This class tracks rolling IC performance
    and alerts when a factor has decayed significantly.
    
    Usage:
        monitor = FactorDecayMonitor(lookback_windows=[168, 336, 720])
        result = monitor.check_decay(factor_values, returns, factor_name='my_factor')
        if result['is_decayed']:
            print(f"Factor has decayed: {result['decay_ratio']:.2%}")
    """
    
    def __init__(self, lookback_windows: List[int] = None, decay_threshold: float = 0.5):
        """
        Args:
            lookback_windows: Windows (in bars) to compute rolling IC. Default: [168, 336, 720] for 1H data (1w, 2w, 1m)
            decay_threshold: If current IC / peak IC < threshold, factor is considered decayed
        """
        self.lookback_windows = lookback_windows or [168, 336, 720]
        self.decay_threshold = decay_threshold
        self.history = {}  # factor_name -> list of (timestamp, ic) tuples
        self.logger = logging.getLogger(__name__)
    
    def compute_rolling_ic(self, factor_values: np.ndarray, returns: np.ndarray, 
                           window: int = 168) -> pd.Series:
        """Compute rolling IC (Spearman correlation) with forward returns."""
        from scipy.stats import spearmanr
        
        n = min(len(factor_values), len(returns))
        factor_values = factor_values[:n]
        returns = returns[:n]
        
        rolling_ics = []
        for i in range(window, n):
            fv = factor_values[i-window:i]
            fr = returns[i-window:i]
            
            valid = np.isfinite(fv) & np.isfinite(fr)
            if valid.sum() >= window // 2:
                try:
                    ic, _ = spearmanr(fv[valid], fr[valid])
                    rolling_ics.append(ic if np.isfinite(ic) else 0)
                except:
                    rolling_ics.append(0)
            else:
                rolling_ics.append(np.nan)
        
        # Pad the beginning with NaN
        result = [np.nan] * window + rolling_ics
        return pd.Series(result)
    
    def check_decay(self, factor_values: np.ndarray, returns: np.ndarray, 
                    factor_name: str = "unnamed") -> Dict:
        """
        Check if a factor has decayed from its historical performance.
        
        Returns:
            dict with: is_decayed, current_ic, peak_ic, decay_ratio, recommendation
        """
        # Use the medium window for primary analysis
        window = self.lookback_windows[len(self.lookback_windows) // 2]
        rolling_ic = self.compute_rolling_ic(factor_values, returns, window)
        
        # Remove NaN values
        valid_ic = rolling_ic.dropna()
        
        if len(valid_ic) < 10:
            return {
                'is_decayed': False,
                'current_ic': 0,
                'peak_ic': 0,
                'decay_ratio': 1.0,
                'recommendation': 'Insufficient data for decay analysis'
            }
        
        # Calculate metrics
        peak_ic = valid_ic.abs().max()
        recent_ic = valid_ic.tail(window // 4).abs().mean()  # Average of recent quarter
        current_ic = valid_ic.iloc[-1] if len(valid_ic) > 0 else 0
        
        decay_ratio = recent_ic / peak_ic if peak_ic > 0.01 else 1.0
        is_decayed = decay_ratio < self.decay_threshold
        
        # Trend analysis
        if len(valid_ic) >= window:
            first_half_ic = valid_ic.iloc[:len(valid_ic)//2].abs().mean()
            second_half_ic = valid_ic.iloc[len(valid_ic)//2:].abs().mean()
            trend = "declining" if second_half_ic < first_half_ic * 0.8 else "stable"
        else:
            trend = "unknown"
        
        # Generate recommendation
        if is_decayed:
            recommendation = f"Factor '{factor_name}' has decayed ({decay_ratio:.1%} of peak). Consider re-mining or removing."
        elif trend == "declining":
            recommendation = f"Factor '{factor_name}' showing decline. Monitor closely."
        else:
            recommendation = f"Factor '{factor_name}' is healthy."
        
        result = {
            'factor_name': factor_name,
            'is_decayed': is_decayed,
            'current_ic': round(float(current_ic), 4),
            'peak_ic': round(float(peak_ic), 4),
            'recent_avg_ic': round(float(recent_ic), 4),
            'decay_ratio': round(float(decay_ratio), 4),
            'trend': trend,
            'recommendation': recommendation
        }
        
        # Log if decayed
        if is_decayed:
            self.logger.warning(f"FACTOR DECAY DETECTED: {recommendation}")
        
        return result
    
    def batch_check(self, factor_df: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        """
        Check decay for all factors in a DataFrame.
        
        Returns:
            DataFrame with decay analysis for each factor
        """
        results = []
        for col in factor_df.columns:
            result = self.check_decay(factor_df[col].values, returns.values, factor_name=col)
            results.append(result)
        
        return pd.DataFrame(results).sort_values('decay_ratio', ascending=True)


# --- 5. Main execution block (can be run standalone) ---
if __name__ == "__main__":
    # Setup logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("="*60)
    print("FactorMiner - Genetic Programming Factor Discovery")
    print("="*60)
    print()
    print("Available Classes:")
    print("  - FactorMiner: GP-based factor discovery with quality evaluation")
    print("  - FactorDecayMonitor: Track factor performance over time")
    print()
    print("Key Features:")
    print("  - evaluate_factor_quality(): IC, IC_IR, Turnover, Monotonicity")
    print("  - _orthogonalize_factors(): Remove correlated factors")
    print("  - Crypto-specific operators: zscore, momentum, delta, sign, cross_above")
    print()
    print("Usage Example:")
    print("  from research.factor_mining import FactorMiner, FactorDecayMonitor")
    print("  miner = FactorMiner(generations=20, population_size=1000)")
    print("  miner.fit(X_train, y_train, feature_names)")
    print("  miner.extract_best_factors(feature_names, min_composite_score=0.15)")


