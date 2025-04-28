import pandas as pd
from itertools import product
from copy import deepcopy
from BaseClasses import SDPPolicy


def grid_search(grid: dict, policy: SDPPolicy.SDPPolicy, n_iterations: int, ordered: bool = False):
    if len(grid) != 2 and ordered:
        ordered = False
        print("Warning: Grid search for ordered parameters only works if there are exactly two parameters.")
    best_performance = 0.0
    best_parameters = None
    rows = []
    params = grid.keys()
    for v in product(*grid.values()):
        if ordered:
            if v[0] >= v[1]:
                continue

        # Do a deep copy so all parameter sets get the same random numbers
        policy_copy = deepcopy(policy)

        for param, value in zip(params, v):
            setattr(policy_copy, param, value)

        performance = policy_copy.run_policy(n_iterations=n_iterations)

        row = dict(zip(params, v))
        row["performance"] = performance
        rows.append(row)
        if performance > best_performance:
            best_performance = performance
            best_parameters = dict(zip(params, v))

    return {
        "best_parameters": best_parameters,
        "best_performance": best_performance,
        "all_runs": pd.DataFrame(rows),
    }


def grid_search_PFA(grid: dict, policy: SDPPolicy.SDPPolicy, n_iterations: int, ordered: bool = False):
    if len(grid) != 2 and ordered:
        ordered = False
        print("Warning: Grid search for ordered parameters only works if there are exactly two parameters.")
    best_performance = 0.0
    best_parameters = None
    rows = []
    params = grid.keys()
    for v in zip(*grid.values()):
        policy_copy = deepcopy(policy)

        for param, value in zip(params, v):
            setattr(policy_copy, param, value)

        performance = policy_copy.run_policy(n_iterations=n_iterations)

        row = dict(zip(params, v))
        row["performance"] = performance
        rows.append(row)
        if performance > best_performance:
            best_performance = performance
            best_parameters = dict(zip(params, v))

    return {
        "best_parameters": best_parameters,
        "best_performance": best_performance,
        "all_runs": pd.DataFrame(rows),
    }

def grid_search_BADP(grid: dict, policy: SDPPolicy.SDPPolicy, n_iterations: int, ordered: bool = False):
    if len(grid) != 2 and ordered:
        ordered = False
        print("Warning: Grid search for ordered parameters only works if there are exactly two parameters.")
    best_performance = 0.0
    best_parameters = None
    rows = []
    params = grid.keys()
    for v in product(*grid.values()):
        if ordered:
            if v[0] >= v[1]:
                continue

        # Do a deep copy so all parameter sets get the same random numbers
        policy_copy = deepcopy(policy)

        for param, value in zip(params, v):
            setattr(policy_copy, param, value)
            #print(f"param: {param}, value: {value}")
            
        # Train the policy (create values)
        policy_copy.train()
        # Test the policy
        performance = policy_copy.run_policy(n_iterations=n_iterations)

        row = dict(zip(params, v))
        param_str = "_".join([f"{param}_{str(value).replace('.', '_')}" for param, value in row.items()])
        model_filename = f"/Users/florian/Documents/github/thesis/stochastic-optimization/EnergyStorage_II/vfa_models/correct_scen_comb/model_{param_str}.pkl"
        scaler_filename = f"/Users/florian/Documents/github/thesis/stochastic-optimization/EnergyStorage_II/vfa_models/correct_scen_comb/scaler_{param_str}.pkl"
        policy_copy.save_model(model_filename, scaler_filename)

        row["performance"] = performance
        rows.append(row)
        if performance > best_performance:
            best_performance = performance
            best_parameters = dict(zip(params, v))

    return {
        "best_parameters": best_parameters,
        "best_performance": best_performance,
        "all_runs": pd.DataFrame(rows),
    }