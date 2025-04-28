import pandas as pd
import numpy as np
import plotly.graph_objects as go
from copy import copy
from typing import Union
import sys
sys.path.append("../")
from BaseClasses.SDPPolicy import SDPPolicy


class EnergyStoragePolicy(SDPPolicy):
    """
    Parent class for energy storage policies.
    All child classes have to implement the get_decision method. 

    Params:
    -------
    :param model: (EnergyStorageModel) - The energy storage model
    :param policy_name: str - The name of the policy
    """
    def __init__(self, model, policy_name=""):
        
        super().__init__(model, policy_name)
        self.results = pd.DataFrame()
        self.performance = pd.NA

    def get_decision(self, state, t, T):
        """
        Every policy has to implement this method. It must return a list of decisions for each hour of the given day.
        """
        pass

    def run_policy(self, n_iterations: int = 1):
        """
        Runs the policy over the time horizon [0,T] for a specified number of iterations and return the mean performance.

        Params:
        -------
        :param n_iterations: int - The number of iterations to run the policy. Default is 1.

        Returns:
        --------
        :return C_t_sum (mean): float - The mean performance of the policy over the iterations.
        """
        result_list = []
        # Note: the random number generator is not reset when calling copy().
        # When calling deepcopy(), it is reset (then all iterations are exactly the same).
        for i in range(n_iterations):
            model_copy = copy(self.model)
            model_copy.episode_counter = i
            model_copy.reset(reset_prng=False)
            state_t_plus_1 = None
            while model_copy.is_finished() is False:
                state_t = model_copy.state
                decisions_t = model_copy.build_decision(self.get_decision(state_t, model_copy.t, model_copy.T))
                # Logging
                results_dict = {"N": i, "t": model_copy.t, "C_t sum": model_copy.objective}
                results_dict.update(state_t._asdict())
                results_dict.update({"decisions": decisions_t})
                result_list.append(results_dict)

                # # enable this to get original version from base class:
                # for decision_t in decisions_t:
                #    state_t_plus_1 = model_copy.step(decision_t)

                state_t_plus_1 = model_copy.step(decisions_t)

            results_dict = {"N": i, "t": model_copy.t, "C_t sum": model_copy.objective}
            if state_t_plus_1 is not None:
                results_dict.update(state_t_plus_1._asdict())
            result_list.append(results_dict)

        # Logging
        self.results = pd.DataFrame.from_dict(result_list)
        # t_end per iteration
        self.results["t_end"] = self.results.groupby("N")["t"].transform("max")

        # performance of one iteration is the cumulative objective at t_end
        self.performance = self.results.loc[self.results["t"] == self.results["t_end"], ["N", "C_t sum"]]
        self.performance = self.performance.set_index("N")

        # For reporting, convert cumulative objective to contribution per time
        self.results["C_t"] = self.results.groupby("N")["C_t sum"].diff().shift(-1)

        if self.results["C_t sum"].isna().sum() > 0:
            print(f"Warning! For {self.results['C_t sum'].isna().sum()} iterations the performance was NaN.")
    
        return self.performance.mean().iloc[0].item()
    

    def plot_decisions(self):
        """
        Plots the decisions over time for each iteration.
        """
        df = self.results
        policy_name = self.policy_name
        last_C_t_sum_list = []
        fig = go.Figure()

        for run in df['N'].unique():
            run_df = df[df['N'] == run]
            all_decisions = []
            all_prices = []

            for i in range(0, len(run_df) - 1):
                all_prices.extend(run_df["price"].iloc[i])
                for decision in run_df["decisions"].iloc[i]:
                    if decision[0] > 0:
                        all_decisions.append(-1)
                    elif decision[2] > 0:
                        all_decisions.append(1)
                    else:
                        all_decisions.append(0)

            last_C_t_sum = run_df["C_t sum"].iloc[-1] if run_df["decisions"].isna().sum() == 0 else run_df[run_df["decisions"].isna()]["C_t sum"].values[0]
            last_C_t_sum_list.append(last_C_t_sum)

            colors = ["green" if decision == 1 else "red" if decision == -1 else "orange" for decision in all_decisions]

            fig.add_trace(go.Scatter(
                x=list(range(len(all_prices))),
                y=all_prices,
                mode="lines",
                line=dict(color="black"),
                name=f'Run {run} Prices',
                visible=(run == 0)  # Only the first run is visible initially
            ))

            legend_shown = {"buy": False, "sell": False, "hold": False}
            for i, price in enumerate(all_prices):
                color = colors[i]
                if color == "green":
                    legendgroup = "sell"
                    name = "Sell"
                elif color == "red":
                    legendgroup = "buy"
                    name = "Buy"
                else:
                    legendgroup = "hold"
                    name = "Hold"

                showlegend = not legend_shown[legendgroup]
                legend_shown[legendgroup] = True

                fig.add_trace(go.Scatter(
                    x=[i],
                    y=[price],
                    mode="markers+lines",
                    marker=dict(color=color),
                    name=name,
                    legendgroup=legendgroup,
                    showlegend=showlegend,
                    visible=(run == 0) 
                ))

        # Update layout to add buttons for toggling visibility
        fig.update_layout(
            xaxis=dict(tickmode="array", tickvals=list(range(0, len(all_prices), 24)), ticktext=[f"t={i}" for i in range(0, len(all_prices) // 24 + 1)]),
            title=dict(text=f"<b>{policy_name}</b> - Decisions over time", font=dict(size=16)),
            xaxis_title="States",
            yaxis_title="Price",
            legend_title=f"<b>Legend</b>",
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [{"visible": [run == i for run in df['N'].unique() for _ in range(len(all_prices) + 1)]}],
                            "label": f"<b>Run {i}:</b> {last_C_t_sum_list[i]:.3f}",
                            "method": "update"
                        } for i in df['N'].unique()
                    ],
                    "direction": "down",
                    "showactive": True,
                    "x": 1.15,
                    "xanchor": "right",
                    "y": 1.15,
                    "yanchor": "top"
                }
            ]
        )
        fig.show()


class BuyLowSellHigh(EnergyStoragePolicy):
    """
    Initializes high low policy from EnergyStoragePolicy.

    Params
    -------
    :param theta_low: float - lower limit for the policy
    :param theta_high: float - upper limit for the policy
    :param verbose: bool - whether to print the hourly decisions or not (default is False)
    """
    def __init__(self, model, policy_name: str, theta_low: Union[int, float], theta_high: Union[int, float], verbose: bool=False):
        super().__init__(model, policy_name)
        assert isinstance(theta_low, int/float) and isinstance(theta_high, int/float), "Theta_low and theta_high must be integers or floats"
        assert theta_low < theta_high, "Theta_low must be smaller than theta_high"
        self.theta_low = theta_low
        self.theta_high = theta_high
        self.verbose = verbose

    def get_decision(self, state, t, T):
        lower_limit = self.theta_low
        upper_limit = self.theta_high
        Rmax = self.model.init_args["Rmax"]
        buy = self.model.possible_decisions[0]
        hold = self.model.possible_decisions[1] 
        sell = self.model.possible_decisions[2]

        decisions = []
        for i, price in enumerate(state.price):
            if (price >= upper_limit and state.energy_amount > 0) or (state.energy_amount > 0):
                decision = sell 
            elif price <= lower_limit:
                if state.energy_amount == Rmax:
                    decision = hold 
                else:
                    decision = buy  
            else:
                decision = hold # Fallback
            decisions.append(decision)

            # Logging
            if self.verbose:
                print(f"Price: {price}, Decision: {decision}")

        return decisions
    

class BuyLowSellHigh_Forecast(EnergyStoragePolicy):
    """
    Initializes high low policy from EnergyStoragePolicy

    Params
    -------
    :param theta_low: float - lower limit for the policy
    :param theta_high: float - upper limit for the policy
    :verbose: bool - whether to print the hourly decisions or not (default is False)    
    """
    def __init__(self, model, policy_name, theta_low, theta_high, forecast_length, verbose=False):
        super().__init__(model, policy_name)
        self.theta_low = theta_low
        self.theta_high = theta_high
        self.forecast_length = forecast_length
        self.verbose = verbose

    def get_decision(self, state, t, T):
        lower_limit = self.theta_low
        upper_limit = self.theta_high
        Rmax = self.model.init_args["Rmax"]
        buy = self.model.possible_decisions[0]
        hold = self.model.possible_decisions[1] 
        sell = self.model.possible_decisions[2]
   
        forecast = self.model.get_price_forecast(self.forecast_length)
        decisions = []
        for i, price in enumerate(state.price):
            if (price >= upper_limit and state.energy_amount > 0) or (state.energy_amount > 0):    
                if forecast[i] < price:
                    decision = sell
                else:
                    decision = hold

            elif price <= lower_limit and state.energy_amount < Rmax:
                if forecast[i] > price:
                    decision = buy
                else:
                    decision = hold

            else:
                decision = hold
            decisions.append(decision) # Fallback
            
            # Logging
            if self.verbose:
                print(f"Price: {price}, Decision: {decision}")

        return decisions


class TrackPolicy(EnergyStoragePolicy):

    def __init__(self, model, policy_name, theta):
        super().__init__(model, policy_name)
        self.theta = theta

    def get_decision(self, state, t, T):
        pass        


import pulp
from pulp import LpMaximize, LpProblem, LpVariable, lpSum

class DeterministicLookahead(EnergyStoragePolicy):
    """
    Initialized a deterministic lookahead policy from EnergyStoragePolicy
    
    Params
    -------
    :param lookahead: int - the number of states (days) to look ahead
    :param verbose: bool - whether to print the hourly decisions or not (default is False)
    """
    def __init__(self, model, policy_name: str="", horizon: int=1, verbose: bool=False):
        super().__init__(model, policy_name)
        assert horizon > 0 and isinstance(horizon, int), "Lookahead horizon must be an int from 0 to T"
        self.horizon = horizon
        self.verbose = verbose

    def get_decision(self, state, t, T):
        horizon_days = min(self.horizon, T - t)  # horizon in days
        Rmax = self.model.init_args["Rmax"]
        eta = self.model.init_args["eta"]   
        energy_amount = state.energy_amount
        max_load_per_hour = self.model.init_args["max_load_per_hour"] * eta
        current_prices = state.price  # Prices for the current day (1x24 vector)
        forecast_prices = self.model.get_price_forecast(horizon_days * 24)
        all_prices = np.concatenate([current_prices, forecast_prices])

        model = LpProblem(name=f"DL_horizon_{t}", sense=LpMaximize)

        # Decision variables for the entire horizon
        num_hours = 24 * horizon_days
        buy_vars = [LpVariable(f"buy_hour{h}", lowBound=0, upBound=1, cat="Binary") for h in range(num_hours)]
        sell_vars = [LpVariable(f"sell_hour{h}", lowBound=0, upBound=1, cat="Binary") for h in range(num_hours)]
        hold_vars = [LpVariable(f"hold_hour{h}", lowBound=0, upBound=1, cat="Binary") for h in range(num_hours)]
        buy_amounts = [LpVariable(f"buy_amount_hour{h}", 0, min(max_load_per_hour, Rmax - energy_amount)) for h in range(num_hours)]
        sell_amounts = [LpVariable(f"sell_amount_hour{h}", 0, min(max_load_per_hour, energy_amount)) for h in range(num_hours)]

        # Constraints: Only one action (buy, hold, or sell) per hour
        for h in range(num_hours):
            model += buy_vars[h] + sell_vars[h] + hold_vars[h] == 1, f"action_constraint_hour{h}"
            model += buy_amounts[h] <= max_load_per_hour * buy_vars[h], f"buy_constraint_hour{h}"
            model += sell_amounts[h] <= max_load_per_hour * sell_vars[h], f"sell_constraint_hour{h}"

        # Energy balance constraints across the horizon
        energy_balance = energy_amount
        for h in range(num_hours):
            energy_balance += (buy_amounts[h] - sell_amounts[h])
            model += energy_balance >= 0, f"energy_balance_min_hour{h}"
            model += energy_balance <= Rmax, f"energy_balance_max_hour{h}"

        # Enforce selling remaining energy in the last hour if at the last timestep -> perhabs not needed or delete 
        # if t == T - 1:
        #     last_hour = num_hours - 1
        #     model += sell_vars[last_hour] == 1, f"force_sell_last_hour"
        #     model += sell_amounts[last_hour] == energy_balance, f"sell_last_hour"

        # Objective: maximize total reward over the horizon
        immediate_reward = lpSum(
            [-buy_amounts[h] * all_prices[h] + sell_amounts[h] * all_prices[h] for h in range(num_hours)]
        )

        # Estimate the value of the remaining energy at the end of the horizon
        future_reward = energy_balance * all_prices[-1] 
        immediate_reward += future_reward
        model += immediate_reward, "total_reward"

        # Solve the model
        solver = pulp.PULP_CBC_CMD(msg=False)
        model.solve(solver)

        # Logging
        if self.verbose:
            print(f"|---- DL-Model for state: {t}, Status: {pulp.LpStatus[model.status]}, SolutionTime: {model.solutionTime}, Objective: {pulp.value(model.objective)} ----|")

        # Extract decisions for the current day only (first 24 hours)
        daily_decision_vector = []
        for h in range(24):
            decision = {
                "buy": buy_vars[h].varValue,
                "sell": sell_vars[h].varValue,
                "hold": hold_vars[h].varValue,
            }
            daily_decision_vector.append(decision)

            # Logging
            if self.verbose:
                print(f"Day: {t}, Hour: {h}, Decision: {decision}")

        return daily_decision_vector


from collections import defaultdict
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class BADP(EnergyStoragePolicy):
    """
    Initializes a backward dynamic programming policy from EnergyStoragePolicy.

    Params
    -------
    :param sample_size: int - The number of states to sample
    :param discount_factor: float - The discount factor
    :param price_samples: np.ndarray - The price samples
    :param max_iterations: int - The maximum number of iterations
    :param tolerance: float - The tolerance for convergence
    :param test_size: float - The test size for the regression model
    :param verbose: bool - Whether to print the hourly decisions or not (default is False)
    """
    def __init__(self, model, policy_name: str="", sample_size: int=1, discount_factor: float=0.95, price_samples: np.ndarray=None, test_size: float=0.2, max_iterations: int=400, tolerance: float=1e-3, verbose: bool=False):
        super().__init__(model, policy_name)
        assert isinstance(price_samples, np.ndarray) and price_samples.ndim == 2 or price_samples.ndim == 3, "Price samples must be a 2D or 3D numpy array"
        assert isinstance(discount_factor, float) and 0 < discount_factor < 1, "Discount factor must be a float between 0 and 1"
        assert isinstance(sample_size, int) and sample_size > 0, "Sample size represents the amount of states to sample and therefore must be an integer > 0"    
        assert isinstance(max_iterations, int) and max_iterations > 0, "Max iterations must be an integer > 0"

        # Model params
        self.Rmax = self.model.init_args["Rmax"]    
        self.eta = self.model.init_args["eta"]
        self.max_load_per_hour = self.model.init_args["max_load_per_hour"] * self.eta
        self.R0 = self.model.state.energy_amount
    
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.discount_factor = discount_factor
        self.sample_size = sample_size
        self.price_samples = price_samples
        self.test_size = test_size
        self.verbose = verbose

        self.sampled_states = self.sample_states()
        self.scaler = StandardScaler()
        self.linear_model = Ridge()


    def create_discrete_energy_levels(self):
        energy_levels = set()

        # Von init_energy_amount bis Rmax
        current_energy = self.R0
        while current_energy < self.Rmax:
            energy_levels.add(current_energy)
            current_energy += self.max_load_per_hour
        if current_energy != self.Rmax:
            energy_levels.add(self.Rmax)

        # Von init_energy_amount bis 0
        current_energy = self.R0
        while current_energy > 0:
            energy_levels.add(current_energy)
            current_energy -= self.max_load_per_hour
        if current_energy != 0:
            energy_levels.add(0)

        # Von 0 bis Rmax
        current_energy = 0
        while current_energy < self.Rmax:
            energy_levels.add(current_energy)
            current_energy += self.max_load_per_hour
        if current_energy != self.Rmax:
            energy_levels.add(self.Rmax)

        # Von Rmax bis 0
        current_energy = self.Rmax
        while current_energy > 0:
            energy_levels.add(current_energy)
            current_energy -= self.max_load_per_hour
        if current_energy != 0:
            energy_levels.add(0)

        # Ergebnis sortieren und speichern
        self.energy_levels = sorted(list(energy_levels))
        return self.energy_levels
    
    # TODO ORG
    # def sample_states(self):
    #     num_samples = self.sample_size
    #     price_samples = self.price_samples
    #     energy_levels = self.create_discrete_energy_levels()
        
    #     sampled_states = []

    #     if price_samples.ndim == 2:
    #         num_scenarios = 1
    #         price_samples = price_samples[np.newaxis, :, :]
    #     elif price_samples.ndim == 3:
    #         num_scenarios = price_samples.shape[0]
    #     else:
    #         raise ValueError("Price samples must be a 2D or 3D numpy array")

    #     for day in range(num_samples):
    #         for scenario in range(num_scenarios):
    #             state_prices = price_samples[scenario, day]
    #             for hour in range(24):
    #                 for energy in energy_levels:
    #                     is_terminal = (day == num_samples - 1) and (hour == 23)
    #                     state = {
    #                         "day": day,
    #                         "hour": hour,
    #                         "price": state_prices[hour].item(),   
    #                         "energy_amount": energy,
    #                         "is_terminal": is_terminal
    #                     }
    #                     sampled_states.append(state)

    #     self.sampled_states = sampled_states
    #     return sampled_states

    # TODO ORG
    # def bdp(self, num_states=None):
    #     sampled_states = self.sampled_states
    #     print(f"Running BDP: Possible states (sample_size x 24 x 39): {len(sampled_states)}, Sample size: {self.sample_size}, Energy levels: {len(self.energy_levels)}, Price samples: {self.price_samples.shape}")

    #     max_iterations = self.max_iterations 
    #     tolerance = self.tolerance
    #     possible_decisions = self.model.possible_decisions
    #     time_horizon = self.sample_size if num_states is None else num_states

    #     values = defaultdict(dict)
    #     # values = defaultdict(lambda: 0)

    #     # for state in sampled_states:
    #     #     values[(state["day"], state["hour"], state["energy_amount"], state["price"])] = 0   


    #     for iteration in tqdm(range(max_iterations), desc="Value Iterations"):
    #         delta = 0
    #         for time in tqdm(reversed(range(time_horizon*24)), desc="Time steps", leave=False):
    #             for state in sampled_states:

    #                 day = state["day"]
    #                 hour = state["hour"]
    #                 price = state["price"]
    #                 energy_amount = state["energy_amount"]
    #                 is_terminal = state["is_terminal"]

    #                 if day != time:
    #                     continue
                    
    #                 if is_terminal:
    #                     values[time][(day, hour, energy_amount, price)] = 0# (energy_amount * price)
    #                     continue

    #                 v_list = []
    #                 for decision in possible_decisions:
    #                     if energy_amount == self.Rmax and decision["buy"] > 0:
    #                         v_list.append(-5000)
    #                         continue
    #                     if energy_amount == 0 and decision["sell"] > 0:
    #                         v_list.append(-5000)
    #                         continue

    #                     sell_amount = min(energy_amount, self.max_load_per_hour) 
    #                     buy_amount = min(self.Rmax - energy_amount, self.max_load_per_hour)
                        
    #                     contribution = price * (decision["sell"] * sell_amount - decision["buy"] * buy_amount) * 1e-3

    #                     next_hour = hour + 1 if hour < 23 else 0
    #                     next_day = day + 1 if next_hour == 0 else day   
    #                     next_energy_amount = energy_amount + decision["buy"] * buy_amount - decision["sell"] * sell_amount

    #                     next_state = next(
    #                         (state for state in sampled_states if state["day"] == next_day and state["hour"] == next_hour and state["energy_amount"] == next_energy_amount),
    #                         0
    #                     )
    #                     next_state_key = (next_day, next_hour, next_energy_amount, next_state["price"]) if next_state else None

    #                     # if time == (time_horizon*24) - 1:
    #                     #     future_value = 1e4# energy_amount * price   # terminal state sollte eigentlich schon vorher rausgefiltert werden
    #                     # else:
    #                     #     future_value = values.get(next_day, {}).get(next_state_key, 0)
    #                     future_value = values.get(next_day, {}).get(next_state_key, 0)
                        
    #                     if future_value == 0:
    #                         # print("ahahhh")

    #                     v = contribution + self.discount_factor * future_value
    #                     v_list.append(v)
                    
    #                 max_v = max(v_list) if v_list else 0
    #                 current_state_key = (day, hour, energy_amount, price)
    #                 old_value = values[time].get(current_state_key, 0)
    #                 values[time][current_state_key] = max_v

    #                 delta = max(delta, abs(max_v - old_value))
                
    #         if delta < tolerance:
    #             print(f"Converged after {iteration} iterations.")
    #             break
    #     else: 
    #         print("Maximum number of iterations reached.")    

    #     self.values_dict = values



    def create_full_grid(sample_size, energy_levels, price_samples):
        from itertools import product
        days = range(sample_size)
        hours = range(24)
        prices = price_samples.flatten()

        return list(product(days, hours, energy_levels, prices))

    def sample_states(self):
        num_samples = self.sample_size
        price_samples = self.price_samples
        energy_levels = self.create_discrete_energy_levels()

        sampled_states = []

        # Unterscheide zwischen 2D und 3D price_samples
        if price_samples.ndim == 2:
            price_samples = price_samples[np.newaxis, :, :]  # Hinzufügen einer Szenario-Dimension

        num_scenarios = price_samples.shape[0]

        for scenario in range(num_scenarios):
            scenario_states = []
            for day in range(num_samples):
                state_prices = price_samples[scenario, day]
                for hour in range(24):
                    for energy in energy_levels:
                        is_terminal = (day == num_samples - 1) and (hour == 23)
                        state = {
                            "day": day,
                            "hour": hour,
                            "price": state_prices[hour].item(),
                            "energy_amount": energy,
                            "is_terminal": is_terminal,
                            "scenario": scenario  # Szenario-Index hinzufügen
                        }
                        scenario_states.append(state)
            sampled_states.append(scenario_states)

        self.sampled_states = sampled_states
        return sampled_states

    def bdp(self, aggregate_method="mean"):
        """
        Führt die Backward Dynamic Programming (BDP)-Methode durch.

        Args:
            aggregate_method (str, optional): Methode zur Aggregation der Szenarien. 
                                            "mean" für Mittelwertbildung,
                                            "combine" zum Zusammenfügen aller Szenarien.
                                            Default ist "mean".

        Returns:
            None: Ergebnisse werden in `self.values_dict` gespeichert.
        """
        sampled_states = self.sampled_states
        num_scenarios = len(sampled_states)

        print(f"Running BDP with {num_scenarios} scenarios. Sample size: {self.sample_size}, Energy levels: {len(self.energy_levels)}.")


        possible_decisions = self.model.possible_decisions
        scenario_values = []  # Werte für alle Szenarien speichern

        for scenario in range(num_scenarios):
            values = defaultdict(dict)
            current_sampled_states = sampled_states[scenario]

            # Sicherstellen, dass jeder Zustand mindestens einmal betrachtet wird
            all_state_keys = {
                (state["day"], state["hour"], state["energy_amount"], state["price"])
                for state in current_sampled_states
            }

            # Beginne mit dem definierten Startzustand
            start_state = next(
                (state for state in current_sampled_states if state["day"] == 0 and state["hour"] == 0 and state["energy_amount"] == self.R0),
                None
            )
            if not start_state:
                print(f"Error: Start state not found for scenario {scenario}!")
                continue

            queue = [start_state]
            visited_states = set()
            
            with tqdm(total=len(current_sampled_states), desc=f"BDP - Scenario {scenario}") as pbar:
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

                        next_hour = hour + 1 if hour < 23 else 0
                        next_day = day + 1 if next_hour == 0 else day
                        next_energy_amount = energy_amount + decision["buy"] * buy_amount - decision["sell"] * sell_amount

                        next_state = next(
                            (state for state in current_sampled_states if state["day"] == next_day and state["hour"] == next_hour and state["energy_amount"] == next_energy_amount),
                            None
                        )
                        if next_state and (next_day, next_hour, next_energy_amount, next_state["price"]) not in visited_states:
                            queue.append(next_state)

                        next_state_key = (next_day, next_hour, next_energy_amount, next_state["price"]) if next_state else None
                        future_value = values.get(next_day, {}).get(next_state_key, 0)

                        if energy_amount == 0 and decision["buy"] > 0:
                            contribution += 5  # Beispiel für buy_preference_bonus = 3

                        # if is_terminal and energy_amount == 0:
                        #     values[day][current_state_key] = 1e2  # Beispiel für terminal_low_energy_penalty = 15

                        # Berechnung der Belohnung mit zusätzlichen Komponenten
                        self.energy_bonus_factor = 0.08
                        self.low_energy_penalty = 5
                        self.low_energy_discount_boost = 1.2

                        energy_reward = self.energy_bonus_factor * energy_amount
                        energy_penalty = -self.low_energy_penalty if energy_amount == 0 else 0
                        adjusted_discount_factor = self.discount_factor * (self.low_energy_discount_boost if energy_amount == 0 else 1)

                        # Berechnung des Wertes
                        v = contribution + energy_reward + energy_penalty + adjusted_discount_factor * future_value

                        # v = contribution + self.discount_factor * future_value
                        v_list.append(v)

                    max_v = max(v_list) if v_list else 0
                    values[day][current_state_key] = max_v

            scenario_values.append(values)

        # Aggregiere Werte basierend auf der gewählten Methode
        if aggregate_method == "mean":
            # Durchschnittliche Werte über alle Szenarien berechnen
            aggregated_values = defaultdict(dict)
            for day in range(self.sample_size):
                for scenario in scenario_values:
                    for key, value in scenario.get(day, {}).items():
                        if key not in aggregated_values[day]:
                            aggregated_values[day][key] = []
                        aggregated_values[day][key].append(value)
                aggregated_values[day] = {key: np.mean(vals) for key, vals in aggregated_values[day].items()}

            self.values_dict = aggregated_values
        elif aggregate_method == "combine":
            # Alle Szenarien zu einem Datensatz kombinieren
            combined_values = defaultdict(dict)
            for scenario in scenario_values:
                for day in scenario.keys():
                    combined_values[day].update(scenario[day])
            self.values_dict = combined_values
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate_method}")

    def plot_values_heatmap(self):
        """
        Plots a heatmap showing the relationship between energy levels, prices, and values.
        
        :param values_dict: defaultdict(dict) - Dictionary containing state tuples as keys and their corresponding values.
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


    def plot_values_3d(self):
        """
        Creates a 3D plot to visualize the relationship between energy levels, prices, and values.
        
        :param values_dict: defaultdict(dict) - Dictionary containing state tuples as keys and their corresponding values.
        """
        assert self.values_dict is not None, "Values dictionary is empty. Please run the BDP method first."

        # Extract unique energy levels and prices from the values_dict
        energy_levels = sorted({key[2] for _, states in self.values_dict.items() for key in states.keys()})
        prices = sorted({key[3] for _, states in self.values_dict.items() for key in states.keys()})

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

    def extract_features(self):
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

    def fit_regression_model(self):

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
        print("Feature Coefficients:")
        for name, importance in zip(feature_names, coefs):
            print(f"  {name}: {importance}")

        self.pred = y_pred
        self.train_pred = y_train_pred
        self.y_train = y_train
        self.y_test = y_test

    def get_value(self, state):
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

    def save_regr_model(self, model_path, scaler_path):
        import joblib
        joblib.dump(self.linear_model, model_path)
        joblib.dump(self.scaler, scaler_path)  

    def load_regr_model(self, model_path, scaler_path):
        import joblib
        self.linear_model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def train(self):
        print(f"Training policy: {self.policy_name}")
        self.bdp()
        self.extract_features()
        self.fit_regression_model()
        print("Training completed.")

    def get_decision(self, state, t, T):
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
                    print(f"Hour: {hour}, Decision: {decision}, Contribution: {contribution}, Future value: {future_value}, Value: {v}")

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
                print(f"|---- Day: {t}, Hour: {hour}, Decision: {max_Dec}, Energy: {current_energy_amount} ----|")

            max_Dec_vector.append(max_Dec)        
        
        return max_Dec_vector






































































# from collections import defaultdict
# class BDP(EnergyStoragePolicy):

#     def __init__(self, model, policy_name, num_bins, discount_factor, price_sample):
#         super().__init__(model, policy_name)
#         self.discount_factor = discount_factor
#         self.num_bins = num_bins

#         self.price_sample = self.model.exog_params["hist_price"] if price_sample is None else price_sample

#         self.Rmax = self.model.init_args["Rmax"]
#         self.R0 = self.model.state.energy_amount
#         self.eta = self.model.init_args["eta"]
#         self.max_load_per_hour = self.model.init_args["max_load_per_hour"] * self.eta

#         self.discrete_prices = self.create_discrete_price_distribution(self.price_sample, num_bins=self.num_bins)
#         self.f_p = [dist[2] for dist in self.discrete_prices]

#         self.discrete_energy_levels = self.create_discrete_energy_levels(self.Rmax, self.max_load_per_hour)
#         self.possible_states = []
#         self.values_dict = {}


#     def create_discrete_energy_levels(self, Rmax, max_load_per_hour):
#         energy_levels = set()

#         current_energy = 0
#         while current_energy <= Rmax:
#             energy_levels.add(current_energy)
#             current_energy += max_load_per_hour
#         if current_energy - max_load_per_hour != Rmax:
#             energy_levels.add(Rmax)
        
#         current_energy = self.R0
#         while current_energy > 0:
#             energy_levels.add(current_energy)
#             current_energy -= max_load_per_hour
#         energy_levels.add(0)

#         current_energy = self.R0
#         while current_energy <= Rmax:
#             energy_levels.add(current_energy)
#             current_energy += max_load_per_hour

#         return sorted(list(energy_levels))

    
#     def create_discrete_price_distribution(self, price_sample, num_bins):
#         assert isinstance(price_sample, np.ndarray) and price_sample.shape[1] == 24, "Price sample must be a 2D np array with 24 columns"
#         assert isinstance(num_bins, int) and num_bins > 0, "Number of bins must be an integer greater than 0"

#         num_samples, num_hours = price_sample.shape
#         hourly_distribution = []

#         for hour in range(num_hours):
#             hourly_prices = price_sample[:, hour]   
#             sorted_prices = np.sort(hourly_prices)

#             min_price = sorted_prices[0]
#             max_price = sorted_prices[-1]
#             price_bins = np.linspace(min_price, max_price, num_bins+1)

#             probabilities, _ = np.histogram(hourly_prices, bins=price_bins, density=True)
#             probabilities /= probabilities.sum()

#             bin_midpoints = (price_bins[:-1] + price_bins[1:]) / 2

#             cdf = np.cumsum(probabilities)
#             hourly_distribution.append((bin_midpoints, probabilities, cdf))

#         return hourly_distribution


#     def create_possible_states(self):
#         self.possible_states = []

#         for hour in range(24):
#             for energy in self.discrete_energy_levels:
#                 for price in self.discrete_prices[hour][0]:     # 0 ist falsch das sollte für jeden preis in self.discrete_prices[hour] sein
#                     state = {
#                         "hour": hour,
#                         "energy_amount": energy,
#                         "price": price.item()
#                     }
#                     self.possible_states.append(state)


#     def state_transition_fn(self, state, decision, exog_info):

#         next_price = exog_info
#         next_hour = state["hour"] + 1 if state["hour"] < 23 else 0
#         next_energy_amount = state["energy_amount"] + decision["buy"] * min(self.Rmax - state["energy_amount"], self.max_load_per_hour) - decision["sell"] * min(state["energy_amount"], self.max_load_per_hour)

#         next_state = {
#             "hour": next_hour,
#             "energy_amount": next_energy_amount,
#             "price": next_price.item()
#         }
#         return next_state

#     def bellman(self):


#         self.create_possible_states()
#         possible_states = self.possible_states
#         possible_decisions = self.model.possible_decisions
#         time_horizon = self.model.T        
#         values = defaultdict(dict)
        
#         max_iterations = 400
#         tolerance = 1e-2

#         for iteration in range(max_iterations):
#             delta = 0
#             for time in reversed(range(time_horizon)):
#                 for state in possible_states:
#                     # Extrahiere den Preis, die Energie und die Stunde
#                     hour = state["hour"]
#                     price = state["price"]  
#                     energy_amount = state["energy_amount"]
#                     # num_consecutive_holds = 0

#                     v_list = []
#                     for decision in possible_decisions:
#                         if energy_amount == self.Rmax and decision["buy"] > 0:
#                             penalty = -1e6
#                             v_list.append(penalty)
#                             continue
#                         elif energy_amount == 0 and decision["sell"] > 0:
#                             penalty = -1e6
#                             v_list.append(penalty)
#                             continue
#                         # elif energy_amount == 0 and decision["hold"]:
#                         #     penalty = -1e6
#                         #     v_list.append(penalty)
#                         #     continue
#                         hold_penalty = 0
#                         if decision["hold"]:
#                             hold_penalty = -1e4
#                             if energy_amount < self.Rmax * 0.2:
#                                 hold_penalty -= -1e4

#                         # Zusätzliche Gewichtung basierend auf dem Preis
#                         # buy_bonus = 1e5 if decision["buy"] > 0 else 0
#                         # sell_bonus = 1e5 if decision["sell"] > 0 else 0

#                         # buy_bonus = 2.0 * price if decision["buy"] > 0 else 0
#                         # sell_bonus = 1.0 * price if decision["sell"] > 0 else 0

#                         # Sell und Hold penalty
#                         # hold_penalty = -1e6 * num_consecutive_holds if decision["hold"] else 0
#                         # hold_penalty = -1e5 if energy_amount == 0 and decision["hold"] else 0
#                         # sell_penalty = -1e5 if energy_amount == 0 and decision["sell"] else 0

#                         # Berechne den Beitrag des aktuellen Entscheidungsvektors
#                         contribution = price * (
#                             decision["sell"] * min(energy_amount, self.max_load_per_hour) - 
#                             decision["buy"] * min(self.Rmax - energy_amount, self.max_load_per_hour)
#                         ) + hold_penalty #  + buy_bonus + sell_bonus + hold_penalty # + sell_penalty

#                         expected_future_value = 0
#                         next_hour = hour + 1 if hour < 23 else 0 
#                         price_bins, probabilities, cdf = self.discrete_prices[next_hour]

#                         # Berechne den erwarteten zukünftigen Wert
#                         for w_index, w in enumerate(price_bins):
#                             prob = probabilities[w_index]

#                             # Berechne den nächsten Zustand basierend auf der Übergangsfunktion
#                             next_state = self.state_transition_fn(state, decision, w)
#                             next_hour = next_state["hour"]
#                             next_energy_amount = next_state["energy_amount"]

#                             next_time = time + 1 if next_hour == 0 else time

#                             future_value = values.get(next_time, {}).get(
#                                 (next_hour, next_energy_amount, w), 0
#                             )

#                             # Penalty für states mit energy_amount = 0
#                             if next_energy_amount == 0 and decision["sell"] > 0:
#                                 future_value -= 1e4
#                             if next_energy_amount == self.Rmax and decision["buy"] > 0:
#                                 future_value -= 1e4

#                             # if decision["hold"]:
#                             #     future_value *= -10

#                             expected_future_value += prob * future_value

#                         future_value_with_penalty = expected_future_value # + hold_penalty #+ sell_penalty
#                         # Berechne den Wert für diese Entscheidung
#                         v = contribution + self.discount_factor * future_value_with_penalty
#                         v_list.append(v)

#                         # if decision["hold"]:
#                         #     num_consecutive_holds += 1
#                         # else:
#                         #     num_consecutive_holds = 0

#                     # Speichere den maximalen Wert für diese Kombination von hour, energy_amount, price
#                     max_v = max(v_list) if v_list else 0
#                     old_value = values[time].get((hour, energy_amount, price), 0)
#                     values[time][(hour, energy_amount, price)] = max_v.item()

#                     delta = max(delta, abs(max_v - old_value))

#             if delta < tolerance:
#                 print(f"Converged after {iteration} iterations.")
#                 break
#         else: 
#             print("Maximum number of iterations reached.")

#         # Speichere das finale `values_dict`
#         self.values_dict = values


#     def get_decision(self, state, t, T):
#         energy_amount = state.energy_amount

#         possible_decisions = [
#             self.model.possible_decisions[0], 
#             self.model.possible_decisions[1],
#             self.model.possible_decisions[2]
#             ]

#         current_energy_amount = energy_amount

#         max_Dec_vector = []

#         for hour in range(24):
#             price = state.price[hour].item()
#             max_Value = -np.inf
#             max_Dec = None
#             for decision in possible_decisions:

#                 if current_energy_amount == 0 and decision["sell"] > 0:
#                     continue

#                 elif current_energy_amount == self.Rmax and decision["buy"] > 0:
#                     continue

#                 contribution = price * (
#                     decision["sell"] * min(current_energy_amount, self.max_load_per_hour) - 
#                     decision["buy"] * min(self.Rmax - current_energy_amount, self.max_load_per_hour)
#                 )
#                 sum_w = 0
#                 w_index = 0

#                 next_hour = hour + 1 if hour < 23 else 0
#                 price_bins, probabilities, cdf = self.discrete_prices[next_hour]

#                 for w in price_bins:
#                     prob = probabilities[w_index]

#                     next_state = self.state_transition_fn(
#                         {"energy_amount": current_energy_amount, "price": price, "hour": hour}, decision, w
#                     )
#                     next_energy_amount = next_state["energy_amount"]    
#                     next_time = t + 1 if next_hour == 0 else t
#                     next_v = (
#                         self.values_dict.get(next_time, {}).get((next_hour, next_energy_amount, w), 0)
#                         if t < T
#                         else 0
#                     )
#                     sum_w += prob * next_v
#                     w_index += 1

#                     # Logging
#                     print(next_state, prob, next_v, sum_w)
                

#                 v = contribution + self.discount_factor * sum_w

#                 # Logging
#                 print(f"Hour: {hour}, Decision: {decision}, Contribution: {contribution}, Sum_w: {sum_w}, Value: {v}")

#                 if v > max_Value:
#                     max_Value = v
#                     max_Dec = decision

#             if max_Dec is None:
#                 max_Dec = self.model.possible_decisions[1]  # hold if None (buy/sell decisions resulted in lower value than hold)

#             current_energy_amount += (
#                 max_Dec["buy"] * min(self.Rmax - current_energy_amount, self.max_load_per_hour)
#                 - max_Dec["sell"] * min(current_energy_amount, self.max_load_per_hour)
#             )

#             # Logging
#             print(max_Dec)
#             max_Dec_vector.append(max_Dec)
        
#         # Logging
#         print(t, max_Dec_vector)
#         return max_Dec_vector












# from sklearn.linear_model import LinearRegression
# import numpy as np

# class BADP(EnergyStoragePolicy):
#     def __init__(self, model, policy_name, discount_factor, sample_rate, price_samples=None):
#         super().__init__(model, policy_name)
#         self.discount_factor = discount_factor
#         self.sample_rate = sample_rate
#         self.price_samples = price_samples
#         self.features = []  # Feature-Vektoren gesampelter Zustände
#         self.values = []  # Ziele (approximierte Werte gesampelter Zustände)
#         self.theta = None  # Koeffizienten für die Regression
#         self.sampled_s_v_dataset = []
#         self.terminal_contribution = 0
#         self.discretized_prices = None
#         self.train_dataset = None
#         self.model_regression = LinearRegression()

#     # TODO: esm durchlauf auf trainingsdaten machen -> liste and states generieren -> von dieser liste sample_rate viele sample ziehen und abspeichern   
#     #       -> hier könnte man noch die preisszenarien integrieren -> dh. man hätte dann für jedes preisszenario unterschiedliche states und könnte dann aus einer größeren menge an states ziehen 

#     def sample_price_sequence(self, num_samples):
#         all_prices = self.model.exog_params["hist_price"]
#         num_of_states = all_prices.shape[0]
#         assert num_samples < num_of_states, "Number of samples must be less than the number of states"
        
#         start_idx = np.random.randint(0, num_of_states - num_samples)
#         sampled_prices = all_prices[start_idx:start_idx + num_samples, :]
#         return sampled_prices

#     def sample_discretized_states(self, num_samples):
#         Rmax = self.model.init_args["Rmax"]
#         eta = self.model.init_args["eta"]
#         max_load_per_hour = self.model.init_args["max_load_per_hour"] * eta
#         init_energy_amount = self.model.state.energy_amount
        
#         self.energy_levels = []
#         current_energy = init_energy_amount
#         while current_energy < Rmax:
#             self.energy_levels.append(current_energy)
#             if current_energy + max_load_per_hour > Rmax:
#                 current_energy = Rmax
#             else:
#                 current_energy += max_load_per_hour
#         self.energy_levels.append(Rmax)  

#         current_energy = init_energy_amount
#         while current_energy > 0:
#             self.energy_levels.append(current_energy)
#             if current_energy - max_load_per_hour < 0:
#                 current_energy = 0
#             else:
#                 current_energy -= max_load_per_hour
#         self.energy_levels.append(0)  

#         discretized_energy_amounts = np.array(self.energy_levels)
#         discretized_prices = self.sample_price_sequence(num_samples) if self.price_samples is None else self.price_samples
#         self.discretized_prices = discretized_prices

#         self.possible_states = []
#         for state in discretized_prices:
#             for hour, price in enumerate(state):
#                 for energy_amount in discretized_energy_amounts:
#                     state = {
#                         "hour": hour,
#                         "energy_amount": energy_amount.item(),
#                         "price": price.item()
#                     }
#                     self.possible_states.append(state)
        
#         return self.possible_states

#     def state_transition(self, state, decision, exog_info):
#         Rmax = self.model.init_args["Rmax"]
#         eta = self.model.init_args["eta"]
#         max_load_per_hour = self.model.init_args["max_load_per_hour"] * eta

#         new_price = exog_info
#         energy_amount = state["energy_amount"]
#         hour = state["hour"]
#         if decision["buy"] > 0:
#             new_energy = min(energy_amount + max_load_per_hour, Rmax)
#         elif decision["sell"] > 0:
#             new_energy = max(energy_amount - max_load_per_hour, 0)
#         else:
#             new_energy = energy_amount

#         new_state = {
#             "hour": hour + 1 if hour < 23 else 0,
#             "energy_amount": new_energy,
#             "price": new_price
#         }
#         return new_state

#     def get_next_price(self, time, state):
#         if time + 1 < len(self.price_samples):
#             return self.price_samples[time + 1][state["hour"]]
#         return state["price"]

#     def get_next_value(self, current_state, decision, values, time, possible_states):
#         Rmax = self.model.init_args["Rmax"]
#         eta = self.model.init_args["eta"]
#         max_load_per_hour = self.model.init_args["max_load_per_hour"] * eta

#         next_hour = current_state["hour"] + 1 if current_state["hour"] < 23 else 0
#         next_energy_amount = current_state["energy_amount"] + decision["buy"] * min(Rmax - current_energy_amount, max_load_per_hour) - decision["sell"] * min(current_energy_amount, max_load_per_hour)

#         for next_state in possible_states:
#             if (next_state["hour"] == next_hour and abs(next_state["energy_amount"] - next_energy_amount) < 1e-3):
#                 next_value = values.get(time + 1, {}).get(tuple(next_state.values()), 0)
#                 print(f"Next value: {next_value} for next state: {next_state}")
#                 return next_value
#         print(f"No next value found for state: {current_state}")            
#         return 0

#     def calc_value_function(self, num_samples):
#         Rmax = self.model.init_args["Rmax"]
#         eta = self.model.init_args["eta"]
#         max_load_per_hour = self.model.init_args["max_load_per_hour"] * eta
#         possible_decisions = self.model.possible_decisions
#         discount_factor = self.discount_factor
#         possible_states = self.sample_discretized_states(num_samples)

#         values = {t: {} for t in range(num_samples + 1)}
#         time_horizon = num_samples

#         for state in possible_states:
#             values[time_horizon][tuple(state.values())] = 0 # terminal contribution 

#         for time in reversed(range(time_horizon + 1)):
#             for state in possible_states:
#                 price = state["price"]
#                 energy_amount = state["energy_amount"]
#                 hour = state["hour"]
#                 v_list = []

#                 # valid_decisions = possible_decisions.copy()
#                 # if energy_amount == Rmax:
#                 #     valid_decisions = [d for d in valid_decisions if d["buy"] == 0]
#                 # elif energy_amount == 0:
#                 #     valid_decisions = [d for d in valid_decisions if d["sell"] == 0] same as using continue but slower

#                 for decision in possible_decisions:
#                     if energy_amount == Rmax and decision["buy"] > 0:
#                         continue
#                     elif energy_amount == 0 and decision["sell"] > 0:
#                         continue
#                     print(decision)
#                     contribution = price * (decision["buy"] * min(Rmax - energy_amount, max_load_per_hour) - decision["sell"] * min(energy_amount, max_load_per_hour))
#                     # next_state = self.state_transition(state, decision, exog_info=self.get_next_price(time, state))
#                     next_value = self.get_next_value(state, decision, values, time, self.possible_states)
#                     print(next_value)
#                     # next_value = values[time + 1].get(tuple(next_state.values()), 0) if time < time_horizon else 0

#                     v = contribution + discount_factor * next_value
#                     v_list.append(v)

#                 max_value = max(v_list) if v_list else 0
#                 values[time][tuple(state.values())] = max_value

#         # sampled_s_v_dataset = []
#         for state, value in values[time_horizon].items():
#             self.sampled_s_v_dataset.append({"state": state, "value": value})
#         # return self.sampled_s_v_dataset


#     def extract_features(self):
#         train_dataset = []

#         for index, state_value_pair in enumerate(self.sampled_s_v_dataset):
#             state = state_value_pair["state"]
#             value = state_value_pair["value"]
            
#             hour = state[0]
#             energy_amount = state[1]
#             price = state[2]
            
#             previous_price_1 = self.sampled_s_v_dataset[index - 1]["state"][2] if index > 0 else None
#             previous_price_2 = self.sampled_s_v_dataset[index - 2]["state"][2] if index > 1 else None

#             train_dataset.append(
#                 np.array([
#                     value,
#                     hour,
#                     energy_amount,
#                     energy_amount ** 2,
#                     price, 
#                     price ** 2,
#                     previous_price_1,
#                     previous_price_2,
#                     price * energy_amount
#                 ])
#             )
#         train_dataset = np.array(train_dataset)[2:, :] # remove first two rows because of None values
#         self.train_dataset = train_dataset
#         return train_dataset       

    
#     def fit_value_function(self):
#         self.model_regression.fit(self.features, self.values)
#         self.theta = self.model_regression.coef_

#     def backward_optimization(self, T):
#         for t in range(T-1, -1, -1):  # Über Tage iterieren
#             sampled_states = self.sample_states(t, T)
#             for state in sampled_states:
#                 features = []
#                 values = []
#                 for hour in range(24):  # Über Stunden iterieren
#                     energy_amount = state[hour, 0]
#                     price = state[hour, 1]
#                     feature_vector = self.extract_features(state, hour)  # Features für die Stunde
#                     features.append(feature_vector)
#                     value = self.bellman(t, hour, energy_amount, state[:, 1])  # Bellman-Aufruf
#                     values.append(value)
#                 self.features.extend(features) # Flattened to (N, feature_dim)
#                 self.values.extend(values)  # Flattened to (N, 24)

#         if len(self.features) > 0:
#             self.fit_value_function()


#     def get_decision(self, state, t, T):
#         energy_amount = state.energy_amount
#         prices = state.price  

#         decisions = []
#         for hour in range(24):
#             # Aktionen bewerten: buy, hold, sell
#             actions = {
#                 "buy": self.bellman(t + hour, T, min(self.model.init_args["Rmax"], energy_amount + self.model.init_args["max_load_per_hour"]), prices),
#                 "sell": self.bellman(t + hour, T, max(0, energy_amount - self.model.init_args["max_load_per_hour"]), prices),
#                 "hold": self.bellman(t + hour, T, energy_amount, prices),
#             }
#             best_action = max(actions, key=actions.get)
#             decision = {"buy": 0, "sell": 0, "hold": 0}
#             decision[best_action] = 1
#             decisions.append(decision)

#         return decisions







