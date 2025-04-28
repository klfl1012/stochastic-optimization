from Models.EnergyStoragePolicy import EnergyStoragePolicy
import numpy as np
import pulp
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
from math import ceil

class DeterministicLookahead(EnergyStoragePolicy):
    """
    Initialisiert eine deterministische Lookahead-Policy aus EnergyStoragePolicy.

    Parameters
    ----------
    :horizon: int - Anzahl der Tage, die vorausgeschaut werden (Lookahead-Horizont).
    :verbose: bool - Ob die stündlichen Entscheidungen ausgegeben werden sollen (default: False).
    """
    def __init__(self, model: EnergyStoragePolicy, policy_name: str, horizon: int, verbose: bool=False):
        super().__init__(model, policy_name)
        assert horizon > 0 and isinstance(horizon, int), "Lookahead horizon muss eine positive ganze Zahl sein."
        self.horizon = horizon
        self.verbose = verbose


    def get_decision(self, state, t, T):
        horizon_days = min(self.horizon, T - t)  # Begrenzung auf verbleibende Tage
        Rmax = self.model.init_args["Rmax"]
        eta = self.model.init_args["eta"]
        energy_amount = state.energy_amount
        max_load_per_hour = self.model.init_args["max_load_per_hour"] * eta
        current_prices = state.price  # Preise für den aktuellen Tag (1x24 Vektor)
        forecast_prices = self.model.get_price_forecast(horizon_days * 24)
        all_prices = np.concatenate([current_prices, forecast_prices])

        model = LpProblem(name=f"DL_horizon_{t}", sense=LpMaximize)

        # Entscheidungsvariablen für den gesamten Horizont
        num_hours = 24 * horizon_days
        buy_vars = [LpVariable(f"buy_hour{h}", cat="Binary") for h in range(num_hours)]
        sell_vars = [LpVariable(f"sell_hour{h}", cat="Binary") for h in range(num_hours)]
        hold_vars = [LpVariable(f"hold_hour{h}", cat="Binary") for h in range(num_hours)]
        buy_amounts = [LpVariable(f"buy_amount_hour{h}", 0, max_load_per_hour) for h in range(num_hours)]
        sell_amounts = [LpVariable(f"sell_amount_hour{h}", 0, max_load_per_hour) for h in range(num_hours)]
        energy_balance = [LpVariable(f"energy_balance_hour{h}", lowBound=0, upBound=Rmax) for h in range(num_hours)]

        # Einschränkung: Pro Stunde nur eine Aktion erlaubt (Kaufen, Verkaufen oder Halten)
        for h in range(num_hours):
            model += buy_vars[h] + sell_vars[h] + hold_vars[h] == 1, f"action_constraint_hour{h}"
            model += buy_amounts[h] <= max_load_per_hour * buy_vars[h], f"buy_constraint_hour{h}"
            model += sell_amounts[h] <= max_load_per_hour * sell_vars[h], f"sell_constraint_hour{h}"
            model += buy_amounts[h] + sell_amounts[h] <= max_load_per_hour * (1 - hold_vars[h]), f"hold_restricts_trading_hour{h}"

        # Startwert für Energielevel
        model += energy_balance[0] == energy_amount, "initial_energy_balance"

        # Energiebilanz über den Horizont
        for h in range(1, num_hours):
            model += energy_balance[h] == energy_balance[h-1] + (buy_amounts[h-1] - sell_amounts[h-1]), f"energy_update_hour{h}"
            model += energy_balance[h] >= 0, f"energy_balance_min_hour{h}"
            model += energy_balance[h] <= Rmax, f"energy_balance_max_hour{h}"

        # Zielfunktion: Maximierung der Belohnung
        immediate_reward = lpSum(
            [-buy_amounts[h] * all_prices[h] + sell_amounts[h] * all_prices[h] for h in range(num_hours)]
        )

        # Geschätzte Belohnung für die verbleibende Energie am Horizontende
        mean_future_price = np.mean(all_prices[:-24]) if len(all_prices) > 24 else np.mean(all_prices)
        future_reward = energy_balance[-1] * mean_future_price

        model += immediate_reward + future_reward, "total_reward"

        # Optimierungsproblem lösen
        solver = pulp.PULP_CBC_CMD(msg=False)
        model.solve(solver)

        # Logging für Debugging
        if self.verbose:
            print(f"|---- DL-Model für Zustand: {t}, Status: {pulp.LpStatus[model.status]}, Lösungzeit: {model.solutionTime}, Zielfunktion: {pulp.value(model.objective)} ----|")

        # Extraktion der Entscheidungen für die nächsten 24 Stunden
        if t == T - 1:
            max_energy_remaining = energy_balance[-1].varValue

        daily_decision_vector = []
        for h in range(24):
            decision = {
                "buy": buy_vars[h].varValue,
                "sell": sell_vars[h].varValue,
                "hold": hold_vars[h].varValue,
            }

            if t == T - 1:
                remaining_hours = 24 - h  # Verbleibende Stunden bis Tagesende
                needed_hours = ceil(max_energy_remaining / max_load_per_hour)  # Verkaufsstunden berechnen

                if needed_hours >= remaining_hours and max_energy_remaining > 0:
                    decision = {"buy": 0, "sell": 1, "hold": 0}
                    max_energy_remaining -= max_load_per_hour

                max_energy_remaining = max(0, max_energy_remaining)
            daily_decision_vector.append(decision)

            # Logging
            if self.verbose:
                print(f"t: {t}, hour: {h}, energy_level: {max_energy_remaining if t == T - 1 else energy_balance[h].varValue}, decision: {decision}")

        return daily_decision_vector


def benchmark(test_data: np.ndarray, Rmax: float, eta: float, initial_energy_amount: float, max_load_per_hour: float, verbose: bool=False): 
    """
    Args:
        test_data: np.ndarray, shape=(horizon_days, hours_per_day)
        Rmax: float, maximum energy capacity
        eta: float, energy loss factor
        initial_energy_amount: float, initial energy amount
        max_load_per_hour: float, maximum load per hour

    Returns:
        max_contribution: float, maximum contribution
        decision_vector: List[Dict], decision vector
    """
    horizon_days, hours_per_day = test_data.shape
    max_load_per_hour = max_load_per_hour * eta
    model = LpProblem(name="DL_horizon_benchmark", sense=LpMaximize)

    buy_vars = [[LpVariable(f"buy_day{d}_hour{h}", cat="Binary") for h in range(hours_per_day)] for d in range(horizon_days)]
    sell_vars = [[LpVariable(f"sell_day{d}_hour{h}", cat="Binary") for h in range(hours_per_day)] for d in range(horizon_days)]
    hold_vars = [[LpVariable(f"hold_day{d}_hour{h}", cat="Binary") for h in range(hours_per_day)] for d in range(horizon_days)]

    buy_amounts = [[LpVariable(f"buy_amount_day{d}_hour{h}", lowBound=0, upBound=max_load_per_hour) for h in range(hours_per_day)] for d in range(horizon_days)]
    sell_amounts = [[LpVariable(f"sell_amount_day{d}_hour{h}", lowBound=0, upBound=max_load_per_hour) for h in range(hours_per_day)] for d in range(horizon_days)]

    energy_balance = [[LpVariable(f"energy_balance_day{d}_hour{h}", lowBound=0, upBound=Rmax) for h in range(hours_per_day)] for d in range(horizon_days)]

    for d in range(horizon_days):
        for h in range(hours_per_day):
            model += buy_vars[d][h] + sell_vars[d][h] + hold_vars[d][h] == 1, f"action_constraint_day{d}_hour{h}"
            model += buy_amounts[d][h] <= max_load_per_hour * buy_vars[d][h], f"buy_limit_day{d}_hour{h}"
            model += sell_amounts[d][h] <= max_load_per_hour * sell_vars[d][h], f"sell_limit_day{d}_hour{h}"
            model += buy_amounts[d][h] + sell_amounts[d][h] <= max_load_per_hour, f"max_load_day{d}_hour{h}"

    model += energy_balance[0][0] == initial_energy_amount, "initial_energy_balance"

    for d in range(horizon_days):
        for h in range(hours_per_day):
            if h > 0:
                model += (
                    energy_balance[d][h] == energy_balance[d][h-1] + (buy_amounts[d][h-1] - sell_amounts[d][h-1]),
                    f"energy_balance_update_day{d}_hour{h}"
                )
            elif d > 0:
                model += (
                    energy_balance[d][0] == energy_balance[d-1][hours_per_day-1] + (buy_amounts[d-1][hours_per_day-1] - sell_amounts[d-1][hours_per_day-1]),
                    f"energy_balance_carryover_day{d}"
                )

    model += energy_balance[horizon_days-1][hours_per_day-1] == 0, "final_energy_empty"

    immediate_reward = lpSum(
        [-buy_amounts[d][h] * test_data[d, h] / 1000 + sell_amounts[d][h] * test_data[d, h] / 1000  
        for d in range(horizon_days) for h in range(hours_per_day)]
    )
    model += immediate_reward, "total_profit"

    solver = pulp.PULP_CBC_CMD(msg=verbose)
    model.solve(solver)

    max_contribution = model.objective.value()  

    decision_vector = []
    for d in range(horizon_days):
        for h in range(hours_per_day):
            decision_vector.append({
                "day": d + 1,
                "hour": h,
                "price": test_data[d, h],
                "buy": buy_vars[d][h].varValue,
                "sell": sell_vars[d][h].varValue,
                "hold": hold_vars[d][h].varValue,
                "buy_amount": buy_amounts[d][h].varValue / 1000,  
                "sell_amount": sell_amounts[d][h].varValue / 1000, 
                "energy_balance": energy_balance[d][h].varValue,
            })
    
    if verbose:
        for h in range(hours_per_day):
            print(f"Day 1, Hour {h}: {decision_vector[h]}")


    return max_contribution, decision_vector