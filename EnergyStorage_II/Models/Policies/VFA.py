from collections import defaultdict
from typing import Literal
from tqdm import tqdm
import numpy as np, plotly.graph_objects as go, joblib, sys
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from math import ceil

sys.path.append("../")
from Models.EnergyStoragePolicy import EnergyStoragePolicy
from Models.EnergyStorageModel import EnergyStorageModel



class BADP(EnergyStoragePolicy):
    """
    Initializes a backward dynamic programming policy from EnergyStoragePolicy.

    Args:
        model (EnergyStorageModel): The energy storage model.
        policy_name (str): The name of the policy.
        sample_size (int): The number of states to sample.
        price_samples (np.ndarray): The price samples.
        test_size (float): The test size for the regression model.
        discount_factor (float): The discount factor.
        energy_bonus_factor (float): Factor for energy bonus.
        low_energy_penalty (float): Penalty for low energy.
        low_energy_discount_boost (float): Boost for low energy discount.
        verbose (bool): Whether to print the hourly decisions or not.
    """
    def __init__(
            self, 
            model: EnergyStorageModel, 
            policy_name: str="", 
            sample_size: int=1, 
            price_samples: np.ndarray=None, 
            aggregation_method: Literal["mean", "combine"]="mean",
            discount_factor: float=1, 
            energy_bonus_factor: float=0,
            low_energy_penalty: float=0,
            low_energy_discount_boost: float=0,
            test_size: float=0.2, 
            model_type: Literal["linear", "lasso", "ridge", "elasticnet"]="ridge",
            verbose: bool=False
        ) -> None:
        super().__init__(model, policy_name)

        # Check input types and values
        assert isinstance(price_samples, np.ndarray) and (price_samples.ndim == 2 or price_samples.ndim == 3), "Price samples must be a 2D or 3D numpy array"
        assert 0 < discount_factor <= 1, "Discount factor must be between 0 and 1"
        assert energy_bonus_factor >= 0, "Energy bonus factor must be non-negative"
        assert low_energy_penalty >= 0, "Low energy penalty must be non-negative"
        assert low_energy_discount_boost >= 0, "Low energy discount boost must be non-negative"

        # Model params
        self.Rmax = self.model.init_args["Rmax"]    
        self.eta = self.model.init_args["eta"]
        self.max_load_per_hour = self.model.init_args["max_load_per_hour"] * self.eta
        self.R0 = self.model.state.energy_amount

        # Sample params
        self.sample_size = sample_size
        self.price_samples = price_samples

        # BDP params
        self.aggregation_method = aggregation_method
        self.discount_factor = discount_factor
        self.energy_bonus_factor = energy_bonus_factor
        self.low_energy_penalty = low_energy_penalty    
        self.low_energy_discount_boost = low_energy_discount_boost

        # Regression params
        self.scaler = StandardScaler()
        self.model_type = model_type
        models = {
            "linear": LinearRegression(),
            "ridge": RidgeCV(),
            "lasso": LassoCV(),
            "elasticnet": ElasticNetCV()
        }
        self.linear_model = models[model_type]
        self.test_size = test_size

        # Verbose
        self.verbose = verbose

    def create_discrete_energy_levels(self) -> list:
        """
        Create discrete energy levels for the energy storage.

        Returns:
            list: Sorted list of discrete energy levels.
        """
        energy_levels = set()

        # From init_energy_amount to Rmax
        current_energy = self.R0
        while current_energy < self.Rmax:
            energy_levels.add(current_energy)
            current_energy += self.max_load_per_hour
        if current_energy != self.Rmax:
            energy_levels.add(self.Rmax)

        # From init_energy_amount to 0
        current_energy = self.R0
        while current_energy > 0:
            energy_levels.add(current_energy)
            current_energy -= self.max_load_per_hour
        if current_energy != 0:
            energy_levels.add(0)

        # From 0 to Rmax
        current_energy = 0
        while current_energy < self.Rmax:
            energy_levels.add(current_energy)
            current_energy += self.max_load_per_hour
        if current_energy != self.Rmax:
            energy_levels.add(self.Rmax)

        # From Rmax to 0
        current_energy = self.Rmax
        while current_energy > 0:
            energy_levels.add(current_energy)
            current_energy -= self.max_load_per_hour
        if current_energy != 0:
            energy_levels.add(0)

        # Sort and store the result
        self.energy_levels = sorted(list(energy_levels))
        return self.energy_levels

    def create_sample_states(self) -> list:
        """
        Sample states for the energy storage policy.
        Returns:
            list: List of sampled states.
        """
        price_samples = self.price_samples
        energy_levels = self.create_discrete_energy_levels()

        # Ensure price_samples has 3 dimensions (add a scenario dimension if necessary)
        if price_samples.ndim == 2:
            price_samples = price_samples[np.newaxis, :, :]

        sampled_states = [
            [
                {
                    "day": day,
                    "hour": hour,
                    "price": price_samples[scenario, day, hour].item(),
                    "energy_amount": energy,
                    # "is_terminal": (day == self.sample_size - 1) and (hour == 23),
                    "is_terminal": (day == 0) and hour == 0,
                    "scenario": scenario
                }
                for day in range(self.sample_size)
                for hour in range(24)
                for energy in energy_levels
            ]
            for scenario in range(price_samples.shape[0])
        ]

        self.sampled_states = sampled_states
        return sampled_states

    def bdp(self) -> None:
        """
        Perform the backward dynamic programming (BDP) algorithm.

        Returns:
            None: Results are stored in `self.values_dict`.
        """
        self.sampled_states = self.create_sample_states()
        sampled_states = self.sampled_states
        num_scenarios = len(sampled_states)
        possible_decisions = self.model.possible_decisions
        scenario_values = []  # Store values for all scenarios

        if self.verbose == True:
            print(f"Running BDP with {num_scenarios} scenarios. Sample size: {self.sample_size}, Energy levels: {len(self.energy_levels)}.")

        for scenario in range(num_scenarios):
            values = defaultdict(dict)
            current_sampled_states = sampled_states[scenario]

            # Start with the defined initial state
            # start_state = next(
            #     (state for state in current_sampled_states if state["day"] == 0 and state["hour"] == 0 and state["energy_amount"] == self.R0),
            #     None
            # )

            start_state = next(
                (state for state in current_sampled_states if state["day"] == self.sample_size - 1 and state["hour"] == 23 and state["energy_amount"] == self.R0),
                None
            )

            if not start_state:
                print(f"Error: Start state not found for scenario {scenario}!")
                continue

            queue = [start_state]
            visited_states = set()

            with tqdm(total=len(current_sampled_states), desc=f"BDP - Scenario {scenario}", disable=not self.verbose) as pbar:
                while queue:
                    current_state = queue.pop(0)

                    day = current_state["day"]
                    hour = current_state["hour"]
                    price = current_state["price"]
                    energy_amount = current_state["energy_amount"]
                    is_terminal = current_state["is_terminal"]

                    current_state_key = (day, hour, energy_amount, price)

                    if current_state_key in visited_states:
                        continue

                    visited_states.add(current_state_key)
                    pbar.update(1)

                    if is_terminal:
                        values[day][current_state_key] = 0
                        continue

                    v_list = []
                    for decision in possible_decisions:
                        if energy_amount == self.Rmax and decision["buy"] > 0:
                            v_list.append(-1e2)
                            continue
                        if energy_amount == 0 and decision["sell"] > 0:
                            v_list.append(-1e2)
                            continue

                        sell_amount = min(energy_amount, self.max_load_per_hour)
                        buy_amount = min(self.Rmax - energy_amount, self.max_load_per_hour)
                        contribution = price * (decision["sell"] * sell_amount - decision["buy"] * buy_amount) * 1e-3

                        next_hour = hour -1 if hour > 0 else 23
                        next_day = day - 1 if next_hour == 23 else day
                        next_energy_amount = energy_amount + decision["buy"] * buy_amount - decision["sell"] * sell_amount

                        next_state = next(
                            (state for state in current_sampled_states if state["day"] == next_day and state["hour"] == next_hour and state["energy_amount"] == next_energy_amount),
                            None
                        )
                        if next_state and (next_day, next_hour, next_energy_amount, next_state["price"]) not in visited_states:
                            queue.append(next_state)

                        next_state_key = (next_day, next_hour, next_energy_amount, next_state["price"]) if next_state else None
                        future_value = values.get(next_day, {}).get(next_state_key, 0)

                        # if energy_amount == 0 and decision["buy"] > 0:
                        #     contribution += 5  

                        energy_reward = self.energy_bonus_factor * next_energy_amount
                        # energy_penalty = -self.low_energy_penalty if energy_amount == 0 else 0
                        
                        # adjusted_discount_factor = self.discount_factor * (self.low_energy_discount_boost if energy_amount == 0 else 1)

                        # Calculation of value
                        v = contribution + energy_reward + self.discount_factor * future_value 

                        v_list.append(v)

                    max_v = max(v_list) if v_list else 0
                    values[day][current_state_key] = max_v

            scenario_values.append(values)

        # Aggregate values based on the chosen method
        # if self.aggregation_method == "mean":
        #     # Calculate average values across all scenarios
        #     aggregated_values = defaultdict(dict)
        #     for day in range(self.sample_size):
        #         for scenario in scenario_values:
        #             for key, value in scenario.get(day, {}).items():
        #                 if key not in aggregated_values[day]:
        #                     aggregated_values[day][key] = []
        #                 aggregated_values[day][key].append(value)
        #         # aggregated_values[day] = {key: np.mean(vals) for key, vals in aggregated_values[day].items()}
            
        #     final_values = defaultdict(dict)
        #     for day in aggregated_values:
        #         for key, vals in aggregated_values[day].items():
        #             final_values[day][key] = np.mean(vals).item()
        #     self.values_dict = final_values
        if self.aggregation_method == "mean":
            # Zwischenspeicher für die Werte
            aggregated_values = defaultdict(dict)

            # Schleife über alle Tage und Szenarien
            for day in range(self.sample_size):
                for scenario in scenario_values:
                    for (key, value) in scenario.get(day, {}).items():
                        # Extrahiere day, hour, energy_amount und price
                        day, hour, energy_amount, price = key  # Der Preis bleibt weiterhin Teil des Schlüssels

                        # Erstelle den neuen Key ohne Preis, nur mit day, hour und energy_amount
                        new_key = (day, hour, energy_amount)  

                        # Falls der Key noch nicht existiert, erstell eine leere Liste
                        if new_key not in aggregated_values[day]:
                            aggregated_values[day][new_key] = {"values": [], "prices": []}

                        # Füge den Wert und den Preis zur entsprechenden Liste hinzu
                        aggregated_values[day][new_key]["values"].append(value)
                        aggregated_values[day][new_key]["prices"].append(price)

                # Berechne den Mittelwert der Werte und des Preises
                final_values = defaultdict(dict)
                for day in aggregated_values:
                    for new_key, data in aggregated_values[day].items():
                        # Berechne den Mittelwert der Werte
                        avg_value = np.mean(data["values"]).item()
                        # Berechne den Mittelwert des Preises
                        avg_price = np.mean(data["prices"]).item()

                        # Kombiniere den Mittelwert der Werte und den Mittelwert des Preises im neuen Key
                        final_values[day][(new_key[0], new_key[1], new_key[2], avg_price)] = avg_value

            # Speichere die aggregierten Werte
            self.values_dict = final_values

        elif self.aggregation_method == "combine":
            # Combine all scenarios into one dataset
            combined_values = defaultdict(dict)
            for scenario in scenario_values:
                for day in scenario.keys():
                    combined_values[day].update(scenario[day])
            self.values_dict = combined_values
        else:
            raise ValueError(f"Unknown aggregate method: {self.aggregation_method}")

    def plot_values_heatmap(self) -> None:
        """
        Plots a heatmap showing the relationship between energy levels, prices, and values.
        """
        assert self.values_dict is not None, "Values dictionary is empty. Please run the BDP method first."

        # Extract unique energy levels and prices from the values_dict
        energy_levels = sorted({key[2] for _, states in self.values_dict.items() for key in states.keys()})
        prices = sorted({key[3] for _, states in self.values_dict.items() for key in states.keys()})

        # Create a matrix (energy_levels x prices)
        heatmap_data = np.full((len(energy_levels), len(prices)), np.nan)

        # Fill the matrix with values
        for _, states in self.values_dict.items():
            for (day, hour, energy, price), value in states.items():
                energy_idx = energy_levels.index(energy)
                price_idx = prices.index(price)
                heatmap_data[energy_idx, price_idx] = value

        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=prices,
            y=energy_levels,
            colorscale="Viridis",
            colorbar_title="Value (€)",
            hoverongaps=False
        ))

        # Customize the layout
        fig.update_layout(
            title="Heatmap of Values by Energy Levels and Prices",
            xaxis_title="Price (€/MWh)",
            yaxis_title="Energy Level (kWh)"
        )

        fig.show()

    def plot_values_3d(self) -> None:
        """
        Creates a 3D plot to visualize the relationship between energy levels, prices, and values.
        """
        assert self.values_dict is not None, "Values dictionary is empty. Please run the BDP method first."

        # Create lists for 3D plotting
        x_prices = []
        y_energy_levels = []
        z_values = []

        for _, states in self.values_dict.items():
            for (day, hour, energy, price), value in states.items():
                x_prices.append(price)
                y_energy_levels.append(energy)
                z_values.append(value)

        # Create 3D Scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x_prices,
            y=y_energy_levels,
            z=z_values,
            mode='markers',
            marker=dict(
                size=5,
                color=z_values,  # Use values for color
                colorscale='Viridis',
                colorbar_title="Value (€)"
            )
        )])

        # Customize layout
        fig.update_layout(
            title="3D Visualization of Values by Energy Levels and Prices",
            scene=dict(
                xaxis_title="Price (€/MWh)",
                yaxis_title="Energy Level (kWh)",
                zaxis_title="Value (€)"
            ),
            template="plotly_white",
            width=900,
            height=700
        )

        fig.show()

    def extract_features(self) -> None:
        """
        Extract features from the values dictionary for regression.

        Returns:
            None: Results are stored in `self.train_dataset`.
        """
        values = self.values_dict
        train_dataset = []

        for time, states in values.items():
            for (day, hour, energy_amount, price), value in states.items():
                train_dataset.append(
                    np.array([
                        value,
                        hour,
                        energy_amount,
                        price,
                        hour * price,
                        energy_amount * price,
                        energy_amount ** 2,
                        price ** 2,
                    ])
                )
        self.train_dataset = np.array(train_dataset)

    def fit_regression_model(self) -> None:
        """
        Fit a regression model to the extracted features.

        Returns:
            None: Results are stored in `self.linear_model`.
        """
        train_dataset = self.train_dataset
        test_size = self.test_size
        scaler = self.scaler

        X = train_dataset[:, 1:]
        y = train_dataset[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.linear_model.fit(X_train_scaled, y_train)

        y_train_pred = self.linear_model.predict(X_train_scaled)
        y_pred = self.linear_model.predict(X_test_scaled)

        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_pred)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_pred)

        if self.verbose == True:
            print(f"Train MSE: {train_mse}, Test MSE: {test_mse}")
            print(f"Train MAE: {train_mae}, Test MAE: {test_mae}")  

        coefs = self.linear_model.coef_
        feature_names = [
            "hour",
            "energy_amount",
            "price",
            "hour * price",
            "energy_amount * price",
            "energy_amount^2",
            "price^2",
        ]

        if self.verbose == True:
            print("Feature Coefficients:")
            for name, importance in zip(feature_names, coefs):
                print(f"  {name}: {importance}")

        self.pred = y_pred
        self.train_pred = y_train_pred
        self.y_train = y_train
        self.y_test = y_test

    def save_model(self, model_path, scaler_path) -> None:
        joblib.dump(self.linear_model, model_path)
        joblib.dump(self.scaler, scaler_path)  

    def load_model(self, model_path, scaler_path) -> None:
        self.linear_model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def get_value(self, state) -> float:
        """
        Get the predicted value for a given state.

        Args:
            state (dict): The state for which to predict the value.

        Returns:
            float: The predicted value.
        """
        X = np.array([
            state["hour"],
            state["energy_amount"],
            state["price"],
            state["hour"] * state["price"],
            state["energy_amount"] * state["price"],
            state["energy_amount"] ** 2,
            state["price"] ** 2,
        ]).reshape(1, -1)

        X_scaled = self.scaler.transform(X)
        value_prediction = self.linear_model.predict(X_scaled)
        return value_prediction.item()

    def train(self) -> None:
        """
        Train the policy using backward dynamic programming and regression.

        Returns:
            None: Results are stored in `self.values_dict` and `self.linear_model`.
        """
        if self.verbose == True:
            print(f"Training policy: {self.policy_name}") 
        
        self.bdp()
        self.extract_features()
        self.fit_regression_model()
        
        if self.verbose == True:
            print("Training completed.")

    def get_decision(self, state, t, T) -> list:
        """
        Generate a decision vector for a given state.
        """
        possible_decisions = self.model.possible_decisions
        energy_amount = state.energy_amount
        current_energy_amount = energy_amount

        forecast = self.model.get_price_forecast(24)
        forecast_hour0 = forecast[0].item()

        max_Dec_vector = []

        for hour in range(24):

            price = state.price[hour].item()
            max_Value = -np.inf
            max_Dec = None

            for decision in possible_decisions:

                if t == T - 1:
                    needed_hours = ceil(current_energy_amount / self.max_load_per_hour)
                    remaining_hours = 24 - hour
                    if remaining_hours <= needed_hours and current_energy_amount > 0:
                        decision = {"buy": 0, "sell": 1, "hold": 0}

                if current_energy_amount == 0 and decision["sell"] > 0:
                    continue

                if current_energy_amount == self.Rmax and decision["buy"] > 0:
                    continue

                sell_amount = min(current_energy_amount, self.max_load_per_hour)
                buy_amount = min(self.Rmax - current_energy_amount, self.max_load_per_hour)

                contribution = price * (decision["sell"] * sell_amount - decision["buy"] * buy_amount) * 1e-3

                next_hour = hour + 1 if hour < 23 else 0
                next_energy_amount = current_energy_amount + decision["buy"] * buy_amount - decision["sell"] * sell_amount
                next_price = state.price[next_hour].item() if next_hour < 23 else forecast_hour0

                next_state = {
                    "hour": next_hour,
                    "energy_amount": next_energy_amount,
                    "price": next_price
                }   

                future_value = self.get_value(next_state)
                v = contribution + self.discount_factor * future_value

                # Logging
                if self.verbose == True:
                    print(f"Hour: {hour}, Decision: {decision}, Energy_amount: {current_energy_amount}, Contribution: {contribution}, Future value: {future_value}, Value: {v}")
                    if t == T - 1:
                        print(f"Remaining hours: {remaining_hours}, Needed hours: {needed_hours}")

                if v > max_Value:
                    max_Value = v
                    max_Dec = decision
        
            if max_Dec is None:
                max_Dec = possible_decisions[1] # Fallback to hold if decision is None
                # Logging
                if self.verbose == True:
                    print("Decision is None")

            current_energy_amount += (
                max_Dec["buy"] * min(self.Rmax - current_energy_amount, self.max_load_per_hour) -
                max_Dec["sell"] * min(current_energy_amount, self.max_load_per_hour)
            )

            # Logging
            if self.verbose == True:
                print(f"|---- t: {t}, h: {hour}, Decision: {max_Dec}, Energy: {current_energy_amount} ----|")

            max_Dec_vector.append(max_Dec)        
        
        return max_Dec_vector
    












class Dynamic_BADP(BADP):

    def __init__(
            self,
            model: EnergyStorageModel,
            policy_name: str="Dynamic_BADP",
            sample_size: int=1,
            price_samples: np.ndarray=None,
            aggregation_method: Literal["mean", "combine"]="mean",
            discount_factor: float=1,
            energy_bonus_factor: float=0
    ):
        super().__init__(model, policy_name, sample_size, price_samples, aggregation_method, discount_factor, energy_bonus_factor)
    

    def get_decision(self, state, t, T) -> list:
        prices = state.price
        energy_amount = state.energy_amount
        discr_energy_levels = super.create_discrete_energy_levels(state.energy_amount, state.Rmax, state.max_load_per_hour)




