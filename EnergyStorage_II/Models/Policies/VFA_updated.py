
from collections import defaultdict
from typing import Literal
from tqdm import tqdm
import numpy as np, joblib
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from Models.EnergyStoragePolicy import EnergyStoragePolicy
from Models.EnergyStorageModel import EnergyStorageModel

class BADP(EnergyStoragePolicy):
    def __init__(self, model, sample_size, price_samples, test_size=0.2, discount_factor=0.99):
        super().__init__(model, "BADP")
        self.sample_size = sample_size
        self.price_samples = price_samples  # Shape: (num_samples, T, 24)
        self.test_size = test_size
        self.discount_factor = discount_factor
        self.value_functions = {}  # Speichert ein Modell pro t

    def train_vfa(self):
        """Trainiert eine Value Function Approximation (VFA) für jedes t separat."""
        T = self.price_samples.shape[1]
        
        for t in tqdm(range(T), desc="Training VFAs for each t"):
            X, y = [], []
            for sample_idx in range(self.price_samples.shape[0]):
                price_path = self.price_samples[sample_idx, t, :]  # Shape: (24,)
                state = self.model.get_state_at_t(t)  # Funktion zum Abrufen des Zustands
                rewards = self.compute_rewards(state, price_path)

                for step in range(24):  # 24 Entscheidungen pro t
                    X.append([state, price_path[step]])
                    y.append(rewards[step])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
            model = RidgeCV(alphas=np.logspace(-3, 3, 10))
            model.fit(X_train, y_train)
            
            self.value_functions[t] = model

            # Teste das Modell (Debugging)
            preds = model.predict(X_test)
            print(f"t={t}, MSE: {mean_squared_error(y_test, preds):.4f}")

    def compute_rewards(self, state, price_path):
        """Berechnet Belohnungen für eine gegebene Preissequenz."""
        rewards = np.zeros(24)
        for step in range(24):
            action = self.model.get_optimal_action(state, price_path[step])
            rewards[step] = self.model.reward_function(state, action, price_path[step])
        return rewards

    def get_decisions(self, state, t):
        """Gibt 24 Entscheidungen für ein gegebenes t zurück."""
        model = self.value_functions.get(t)
        if not model:
            raise ValueError(f"No trained VFA model for t={t}")

        price_path = self.price_samples[:, t, :]  # Shape: (num_samples, 24)
        decisions = []
        for i in range(price_path.shape[0]):  # Mehrere Sample Paths
            decisions.append([self.model.get_optimal_action(state, p) for p in price_path[i]])

        return np.array(decisions)  # Shape: (num_samples, 24)
