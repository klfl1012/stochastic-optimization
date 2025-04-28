import plotly.graph_objects as go
import pandas as pd
import numpy as np




def plot_decisions_over_t(df, policy_name):
    """
    Plots the decisions of the policy [buy, sell, hold] at the current price over time for the given dataframe.

    :param df: pd.DataFrame - policy.results dataframe after calling policy.run_policy() function containing the decisions and prices
    """
    all_decisions = []
    for i in range(0, len(df) - 1):
        for decision in df["decisions"][i]:
            if decision.buy > 0:
                all_decisions.append(-1)
            elif decision.sell > 0:
                all_decisions.append(1)
            else:
                all_decisions.append(0)

    
    all_prices = []
    for i in range(0, len(df) - 1):
        for price in df["price"][i]:
            all_prices.append(price)

    
    colors = ["green" if decision == 1 else "red" if decision == -1 else "orange" for decision in all_decisions]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(all_prices))), y=all_prices, mode="lines", line=dict(color="black"), name="price"))

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
        
        fig.add_trace(go.Scatter(x=[i], y=[price], mode="markers+lines", marker=dict(color=color), name=name, legendgroup=legendgroup, showlegend=showlegend))

    fig.update_layout(xaxis=dict(tickmode="array", tickvals=list(range(0, len(all_prices), 24)), ticktext=[f"t={i}" for i in range(1, len(all_prices) // 24 + 1)]))
    fig.update_layout(title=f"{policy_name}: Prices and Decisions", xaxis_title="States", yaxis_title="Price", legend_title="Decisions", yaxis2=dict(title="C_t", overlaying="y", side="right"))
    fig.show()
    