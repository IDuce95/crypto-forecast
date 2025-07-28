import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

from .frontend_config import PLOT_COLORS, PLOT_STYLE, MESSAGES

def _create_x_axis(viz_data: Dict[str, Any], total_len: int):

    if 'original_dates' in viz_data and len(viz_data['original_dates']) == total_len:
        try:
            dates = pd.to_datetime(viz_data['original_dates'])
            return dates
        except (ValueError, TypeError):
            return np.arange(total_len)
    else:
        return np.arange(total_len)

def _plot_actual_data(ax, x_axis, original_values, train_end_idx, val_end_idx):

    train_x = x_axis[:train_end_idx]
    val_x = x_axis[train_end_idx:val_end_idx]
    test_x = x_axis[val_end_idx:]

    style = PLOT_STYLE['actual_data']
    color = PLOT_COLORS['actual_data']

    ax.plot(train_x, original_values[:train_end_idx],
            color=color, linewidth=style['linewidth'],
            label='Real Data', alpha=style['alpha'])

    ax.plot(val_x, original_values[train_end_idx:val_end_idx],
            color=color, linewidth=style['linewidth'],
            alpha=style['alpha'])

    ax.plot(test_x, original_values[val_end_idx:],
            color=color, linewidth=style['linewidth'],
            alpha=style['alpha'])

    return train_x, val_x, test_x

def _plot_predictions(ax, train_x, val_x, test_x, predictions):

    style = PLOT_STYLE['predictions']
    color = PLOT_COLORS['predictions']

    train_pred_x = train_x[:len(predictions['train'])]
    ax.plot(train_pred_x, predictions['train'],
            color=color, linestyle=style['linestyle'],
            linewidth=style['linewidth'], alpha=style['alpha'],
            label='Predictions')

    val_pred_x = val_x[:len(predictions['validation'])]
    ax.plot(val_pred_x, predictions['validation'],
            color=color, linestyle=style['linestyle'],
            linewidth=style['linewidth'], alpha=style['alpha'])

    test_pred_x = test_x[:len(predictions['test'])]
    ax.plot(test_pred_x, predictions['test'],
            color=color, linestyle=style['linestyle'],
            linewidth=style['linewidth'], alpha=style['alpha'])

def _add_split_lines(ax, x_axis, train_end_idx, val_end_idx):

    style = PLOT_STYLE['split_lines']
    color = PLOT_COLORS['split_lines']

    if isinstance(x_axis[0], (pd.Timestamp, np.datetime64)):
        train_split_date = x_axis[train_end_idx] if train_end_idx < len(x_axis) else x_axis[-1]
        val_split_date = x_axis[val_end_idx] if val_end_idx < len(x_axis) else x_axis[-1]

        ax.axvline(x=train_split_date, color=color, linestyle='--',
                   linewidth=style['linewidth'], alpha=style['alpha'],
                   label='Train/Validation Split')
        ax.axvline(x=val_split_date, color=color, linestyle='--',
                   linewidth=style['linewidth'], alpha=style['alpha'],
                   label='Validation/Test Split')
    else:
        ax.axvline(x=train_end_idx, color=color, linestyle='--',
                   linewidth=style['linewidth'], alpha=style['alpha'],
                   label='Train/Validation Split')
        ax.axvline(x=val_end_idx, color=color, linestyle='--',
                   linewidth=style['linewidth'], alpha=style['alpha'],
                   label='Validation/Test Split')

def _format_axes(ax, x_axis, dataset_name):

    ax.set_title(f'{dataset_name} Price Prediction Results',
                 fontsize=PLOT_STYLE['title_fontsize'], pad=20)

    if isinstance(x_axis[0], (pd.Timestamp, np.datetime64)):
        ax.set_xlabel('Date', fontsize=PLOT_STYLE['label_fontsize'])
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax.set_xlabel('Time Index', fontsize=PLOT_STYLE['label_fontsize'])

    ax.set_ylabel('Price', fontsize=PLOT_STYLE['label_fontsize'])
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

def _add_info_text(ax, viz_data, result):

    split_ratios = viz_data['split_ratios']
    info_text = (f"Split Ratios - Train: {split_ratios['train']:.1%}, "
                 f"Val: {split_ratios['validation']:.1%}, Test: {split_ratios['test']:.1%}")

    ax.text(0.01, 0.65, info_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
            verticalalignment='top', fontsize=PLOT_STYLE['info_fontsize'])

    metrics = result['metrics']
    metrics_text = (
        f"MAPE - Train: {metrics['train']['mape']:.4f}, "
        f"Val: {metrics['validation']['mape']:.4f}, "
        f"Test: {metrics['test']['mape']:.4f}"
    )
    ax.text(0.01, 0.55, metrics_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
            verticalalignment='top', fontsize=PLOT_STYLE['info_fontsize'])

def render_results_plot(result: Dict[str, Any], dataset_name: str) -> None:

    if 'visualization_data' not in result:
        st.error(MESSAGES['no_viz_data'])
        return

    viz_data = result['visualization_data']
    predictions = result['predictions']

    fig, ax = plt.subplots(figsize=PLOT_STYLE['figure_size'])

    original_values = np.array(viz_data['original_values'])
    train_end_idx = viz_data['train_end_idx']
    val_end_idx = viz_data['val_end_idx']
    total_len = len(original_values)

    x_axis = _create_x_axis(viz_data, total_len)

    train_x, val_x, test_x = _plot_actual_data(
        ax, x_axis, original_values, train_end_idx, val_end_idx)

    _plot_predictions(ax, train_x, val_x, test_x, predictions)

    _add_split_lines(ax, x_axis, train_end_idx, val_end_idx)

    _format_axes(ax, x_axis, dataset_name)

    _add_info_text(ax, viz_data, result)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def display_optimization_results(result: Dict[str, Any], dataset_name: str = "Unknown") -> None:

    st.subheader("Optimization Results")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Best Parameters:**")
        for param, value in result['best_params'].items():
            st.write(f"- {param}: {value}")

    with col2:
        st.write(f"**Optimization Score:** {result['optimization_score']:.4f}")

    st.write("**Metrics:**")
    metrics_data = []
    for dataset_type in ['train', 'validation', 'test']:
        metrics_data.append({
            'Dataset': dataset_type.capitalize(),
            'MAPE': f"{result['metrics'][dataset_type]['mape']:.4f}",
            'R² Score': f"{result['metrics'][dataset_type]['r2']:.4f}"
        })

    df_metrics = pd.DataFrame(metrics_data)
    st.table(df_metrics.to_dict('records'))

    st.subheader("Prediction Visualization")
    render_results_plot(result, dataset_name)
