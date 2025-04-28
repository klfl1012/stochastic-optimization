import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
sys.path.append("../")


def plot_samples(data, generated_samples):
        num_sample_paths = len(generated_samples)
        rows = 3
        cols = 3
        fig = make_subplots(
            rows=rows, 
            cols=cols, 
            shared_xaxes=True, 
            shared_yaxes=True, 
            subplot_titles=[f"Sample Path {i+1}" for i in range(min(num_sample_paths, rows*cols))],
            vertical_spacing=0.1
        )

        for i in range(min(num_sample_paths, rows*cols)):
            row = i // cols + 1
            col = i % cols + 1
            
            fig.add_trace(
                go.Histogram(
                    x=data.flatten(),
                    nbinsx=50,
                    histnorm='probability',
                    name="Original Data",
                    opacity=0.6,
                    marker_color='blue'
                ),
                row=row, col=col
            )

            fig.add_trace(
                go.Histogram(
                    x=generated_samples[i].flatten(),
                    nbinsx=50,
                    histnorm='probability',
                    name=f"Generated Sample {i+1}",
                    opacity=0.6,
                    marker_color='orange'
                ),
                row=row, col=col
            )

        fig.update_layout(
            title="Comparison of Original Data and Generated Sample Paths Distribution",
            xaxis_title="Value",
            yaxis_title="Probability Density",
            showlegend=True,
            height=800,
            title_x=0.5, 
            barmode="overlay" 
        )

        fig.show()


def plot_time_series_with_bounds(data, generated_samples_list):
        fig = go.Figure()
        generated_samples_array = np.array(generated_samples_list)
        min_values = np.min(generated_samples_array, axis=0).flatten()
        max_values = np.max(generated_samples_array, axis=0).flatten()

        fig.add_trace(
            go.Scatter(
                x=list(range(len(min_values))),
                y=min_values,
                mode='lines',
                line=dict(color='grey'),
                name='Min Values',
                showlegend=False
            )
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(len(max_values))),
                y=max_values,
                mode='lines',
                fill='tonexty', 
                line=dict(color='grey'),
                name='Max Values',
                showlegend=False
            )
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(len(data))),
                y=data.flatten(),
                mode='lines',
                name="Original Data",
                line=dict(color='blue', width=2)
            )
        )

        fig.update_layout(
            title="Original Data vs Generated Sample Paths Bounds",
            xaxis_title="Time Step",
            yaxis_title="Value",
            showlegend=True
        )

        fig.show()


def plot_time_series(data, generated_samples_list):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(len(data))),
                y=data.flatten(),
                mode='lines',
                name="Original Data",
                line=dict(color='blue', width=2)
            )
        )

        for i, generated_samples in enumerate(generated_samples_list):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(generated_samples))),
                    y=generated_samples.flatten(),
                    mode='lines',
                    name=f"Generated Sample Path {i+1}",
                    line=dict(width=2),
                    opacity=0.6
                )
            )

        fig.update_layout(
            title="Original Data vs Generated Sample Paths",
            xaxis_title="Index",
            yaxis_title="Value",
            showlegend=True
        )

        fig.show()