import numpy as np
import pandas as pd

class BayesianRegimeDetector:
    """
    Detects market regimes (BULL, BEAR, RANGING) using a Bayesian approach.
    """
    def __init__(self, priors=None):
        """
        Initializes the detector with prior probabilities for each regime.
        
        :param priors: A dictionary with keys 'BULL', 'BEAR', 'RANGING' and their prior probabilities.
        """
        self.states = ['BULL', 'BEAR', 'RANGING']
        if priors is None:
            # Start with an uninformative prior
            self.priors = {'BULL': 1/3, 'BEAR': 1/3, 'RANGING': 1/3}
        else:
            self.priors = priors
            
        self.posteriors = self.priors.copy()

        # --- Likelihoods P(Evidence | State) ---
        # These are assumptions and can be fine-tuned based on historical data analysis.
        
        # Evidence 1: Moving Average Slope (e.g., 20-period MA slope)
        # Categories: 'POSITIVE', 'NEGATIVE', 'FLAT'
        self.ma_slope_likelihoods = {
            'BULL':    {'POSITIVE': 0.8, 'NEGATIVE': 0.1, 'FLAT': 0.1},
            'BEAR':    {'POSITIVE': 0.1, 'NEGATIVE': 0.8, 'FLAT': 0.1},
            'RANGING': {'POSITIVE': 0.2, 'NEGATIVE': 0.2, 'FLAT': 0.6}
        }
        
        # Evidence 2: Volatility (e.g., ATR normalized by price)
        # Categories: 'HIGH', 'LOW'
        self.volatility_likelihoods = {
            'BULL':    {'HIGH': 0.7, 'LOW': 0.3},
            'BEAR':    {'HIGH': 0.3, 'LOW': 0.7},
            'RANGING': {'HIGH': 0.2, 'LOW': 0.8}
        }

    def _get_ma_slope_category(self, slope, threshold=10.0):
        """Categorizes the MA slope."""
        if slope > threshold:
            return 'POSITIVE'
        elif slope < -threshold:
            return 'NEGATIVE'
        else:
            return 'FLAT'

    def _get_volatility_category(self, atr_normalized, threshold=0.009):
        """Categorizes volatility."""
        if atr_normalized > threshold:
            return 'HIGH'
        else:
            return 'LOW'

    def update(self, ma_slope, atr_normalized):
        """
        Updates the posterior probabilities based on new evidence.
        
        :param ma_slope: The slope of the moving average.
        :param atr_normalized: The Average True Range normalized by price.
        :return: A dictionary with the updated posterior probabilities for each state.
        """
        # Handle potential NaN values from indicators
        if pd.isna(ma_slope) or pd.isna(atr_normalized):
            return self.posteriors # Return last known posteriors if evidence is missing

        ma_slope_cat = self._get_ma_slope_category(ma_slope)
        volatility_cat = self._get_volatility_category(atr_normalized)

        # Bayes' Theorem: P(State | E1, E2) = P(E1 | State) * P(E2 | State) * P(State) / P(E1, E2)
        # We assume conditional independence of evidence for simplicity (Naive Bayes)
        
        # Calculate the numerator: P(E1|State) * P(E2|State) * P(State)
        numerators = {}
        for state in self.states:
            likelihood_ma = self.ma_slope_likelihoods[state][ma_slope_cat]
            likelihood_vol = self.volatility_likelihoods[state][volatility_cat]
            prior = self.posteriors[state] # Use last posterior as the new prior
            
            numerators[state] = likelihood_ma * likelihood_vol * prior

        # Calculate the denominator (marginal likelihood or evidence): sum of all numerators
        marginal_likelihood = sum(numerators.values())
        
        # Avoid division by zero
        if marginal_likelihood == 0:
            return self.posteriors

        # Calculate the new posteriors
        for state in self.states:
            self.posteriors[state] = numerators[state] / marginal_likelihood

        # --- Smoothing: Unconditionally prevent posteriors from becoming zero ---
        min_prob = 1e-9
        # Add a small floor value to all posteriors and re-normalize
        # This ensures no probability ever becomes exactly zero, preventing stuck states.
        for state in self.states:
            self.posteriors[state] += min_prob
        
        total_prob = sum(self.posteriors.values())
        if total_prob > 0: # Should always be true, but good practice
            for state in self.states:
                self.posteriors[state] /= total_prob
        # --- End of Smoothing ---
        
        total_prob = sum(self.posteriors.values())
        if total_prob > 0: # Should always be true, but good practice
            for state in self.states:
                self.posteriors[state] /= total_prob
        # --- End of Smoothing ---
            
        return self.posteriors.copy()
