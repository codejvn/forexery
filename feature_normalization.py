"""
Feature normalization wrapper for the trading environment
Normalizes observations to improve learning stability
"""

import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

class NormalizeObservation(ObservationWrapper):
    """
    Normalizes observations using running mean and standard deviation
    """
    def __init__(self, env, epsilon=1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self.obs_mean = None
        self.obs_std = None
        self.count = 0

    def observation(self, obs):
        """Normalize the observation"""
        if self.obs_mean is None:
            # Initialize with first observation
            self.obs_mean = np.zeros_like(obs)
            self.obs_std = np.ones_like(obs)

        # Update running statistics
        self.count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.count
        delta2 = obs - self.obs_mean
        self.obs_std = np.sqrt(
            ((self.count - 1) * self.obs_std**2 + delta * delta2) / self.count
        )

        # Normalize
        normalized_obs = (obs - self.obs_mean) / (self.obs_std + self.epsilon)
        return np.clip(normalized_obs, -10, 10)  # Clip extreme values
