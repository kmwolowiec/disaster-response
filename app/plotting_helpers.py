import pandas as pd
import numpy as np
from plotly.graph_objs import Scatter


def generate_colors_for_plotting_series(start_rgb=(89, 255, 13), stop_rgb=(125, 95, 92), labels=None) -> dict:
    """Generate RGB color sequence for each label from labels.

    Parameters:
        start_rgb: tuple of 3 int, (R, G, B) - first color of the sequence
        stop_rgb: tuple of 3 int, (R, G, B) - last color of the sequence
        labels: list-like object, containing ordered labels.
        Each label will be assigned to a single color.

    Return:
        Dict of {label: 'rgb(R, G, B)'}
    """
    num_colors = len(labels)

    start_red, start_green, start_blue = start_rgb
    stop_red, stop_green, stop_blue = stop_rgb

    reds = np.linspace(start_red, stop_red, num_colors)
    greens = np.linspace(start_green, stop_green, num_colors)
    blues = np.linspace(start_blue, stop_blue, num_colors)

    colors = dict()
    for label, r, g, b in zip(labels, reds, greens, blues):
        color = f'rgb({int(r)}, {int(g)}, {int(b)})'
        colors[label] = color

    return colors


def make_eval_scatter_series(df_evals: pd.DataFrame, eval_colname: str, label: str,
                             metric_pretty_name: str, label_color_dict: dict, mode: str = "markers", **kwargs):
    """Creates go.Scatter object based on df_evals

    Parameters:
        df_evals: pd.DataFrame containing training evaluation data.
        eval_colname: str, metric column name from df_evals that is going to be plotted.
        label: str, serie name.
        metric_pretty_name: metric name shown in the title and the hover
        label_color_dict: dict like {label: color} to obtain serie marker color.
                          That is output from generate_colors_for_plotting_series
        mode: str, Scatter 'mode' parameter value
        **kwargs: parameters of plotly.graph_objs.Scatter

    Return:
        plotly.graph_objs.Scatter
    """

    scatter = Scatter(
        x=df_evals['feature'].tolist(),
        y=df_evals[eval_colname],
        mode=mode,
        text=df_evals['training_timestamp'],
        name=label,
        meta=[label, metric_pretty_name],
        marker={'color': label_color_dict[label]},
        hovertemplate='Feature: %{x}<br>%{meta[1]}: %{y:.2f}<br>Date: %{meta[0]}<extra></extra>',
        hoverinfo='text',
        **kwargs
    )

    return scatter


def compose_plot(data: list, title: str):
    """Prepare input to be processed by plotly.utils.PlotlyJSONEncoder.
    Function output consists of dict of dicts.
    Two major component of output are: data and layout.

    Parameters:
        data: list of plotly.graph_objs, for instance plotly.graph_objs.Scatters
        title: str, y_axis and plot title

    Return:
        dict of dicts ready to be processed by plotly.utils.PlotlyJSONEncoder.
    """

    plot_schema = {
        'data': data,
        'layout': {
            'title': {
                    'text': title + ' across message categories',
                    'font': {'size': 22}
            },
            'yaxis': {
                'title': {
                    'text': title,
                    'font': {'size': 15}
                },
                'tickfont': {
                    'size': 15
                },
            },
            'xaxis': {
                'title': {
                    'text': "Message category",
                    'font': {'size': 15}
                },
                'tickfont': {
                    'size': 15
                },
                'tickangle': -45,
            },
            'margin': {
                'l': 50,
                'r': 50,
                'b': 170,
                't': 35,
                'pad': 2
            },
            'legend': {
                'title': {
                    'text': 'Training date<br>(the greener, the better<br>models average F1-score)',
                    'font': {'size': 15}
                },
                'font': {
                    'size': 15
                }
            },
            'hovermode': 'closest',
            'displayModeBar': False,
            'hoverlabel': {
                'bgcolor': 'black',
                'font': {'size': 15}
            }
        }
    }

    return plot_schema
