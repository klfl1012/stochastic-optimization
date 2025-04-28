from Models.EnergyStoragePolicy import EnergyStoragePolicy
from typing import Union
from math import ceil

class BuyLowSellHigh(EnergyStoragePolicy):
    """
    Initializes high low policy from EnergyStoragePolicy.

    Params
    -------
    :theta_low: float - lower limit for the policy
    :theta_high: float - upper limit for the policy
    :verbose: bool - whether to print the hourly decisions or not (default is False)
    """
    def __init__(self, model: EnergyStoragePolicy, policy_name: str, theta_low: Union[int, float], theta_high: Union[int, float], verbose: bool=False):
        super().__init__(model, policy_name)
        assert theta_low < theta_high, "Theta_low must be smaller than theta_high"
        self.theta_low = theta_low
        self.theta_high = theta_high
        self.verbose = verbose

    def get_decision(self, state, t, T):
        lower_limit = self.theta_low
        upper_limit = self.theta_high
        Rmax = self.model.init_args["Rmax"]
        max_load_per_hour = self.model.init_args["max_load_per_hour"] * self.model.init_args["eta"]
        buy, hold, sell = self.model.possible_decisions

        decisions = []
        current_energy = state.energy_amount

        for i, price in enumerate(state.price):

            if price <= lower_limit: 
                if current_energy < Rmax:
                    decision = buy
                    current_energy = min(Rmax, current_energy + max_load_per_hour)
                else:
                    decision = hold  

            elif price >= upper_limit and current_energy > 0:  
                decision = sell
                current_energy = max(0, current_energy - max_load_per_hour)

            else:
                decision = hold  

            if t == T - 1:
                remaining_hours = 24 - i
                needed_hours = ceil(current_energy / max_load_per_hour)

                if needed_hours > remaining_hours and current_energy > 0:
                    decision = sell
                    current_energy = max(0, current_energy - max_load_per_hour)

            decisions.append(decision)

            # Logging
            if self.verbose:
                print(f"t: {t}, hour: {i}, price: {price}, energy: {current_energy}, decision: {decision}")
        
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
    def __init__(self, model, policy_name: EnergyStoragePolicy, theta_low: Union[int,float], theta_high: Union[int,float], forecast_length: int, verbose: bool=False):
        super().__init__(model, policy_name)
        assert theta_low < theta_high, "Theta_low must be smaller than theta_high"
        self.theta_low = theta_low
        self.theta_high = theta_high
        self.forecast_length = forecast_length
        self.verbose = verbose

    def get_decision(self, state, t, T):
        lower_limit = self.theta_low
        upper_limit = self.theta_high
        Rmax = self.model.init_args["Rmax"]
        max_load_per_hour = self.model.init_args["max_load_per_hour"] * self.model.init_args["eta"]
        buy, hold, sell = self.model.possible_decisions
   
        forecast = self.model.get_price_forecast(self.forecast_length)
        current_energy = state.energy_amount
        decisions = []
        for i, price in enumerate(state.price):
            if price >= upper_limit and current_energy > 0:    
                if forecast[i] < price:
                    decision = sell
                    current_energy = max(0, current_energy - max_load_per_hour)
                else:
                    decision = hold

            elif price <= lower_limit and current_energy < Rmax:
                if forecast[i] > price:
                    decision = buy
                    current_energy = min(Rmax, current_energy + max_load_per_hour)
                else:
                    decision = hold

            else:
                decision = hold
            decisions.append(decision)
            

            if t == T - 1:
                remaining_hours = 24 - i
                needed_hours = ceil(current_energy / max_load_per_hour)

                if needed_hours > remaining_hours and current_energy > 0:
                    decision = sell
                    current_energy = max(0, current_energy - max_load_per_hour)
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