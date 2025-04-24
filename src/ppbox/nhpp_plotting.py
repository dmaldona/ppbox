"""
Visualization module for NHPP models.

This module provides functions for creating visualizations of NHPP models,
including intensity function plots and diagnostic plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import warnings
from typing import Optional, Tuple, List, Union

from .nhpp_fitter import NHPPFitter


def plot_intensity(model: NHPPFitter, 
                  ax=None, 
                  resolution: int = 100, 
                  show_events: bool = True,
                  label: Optional[str] = None,
                  color: Optional[str] = None,
                  **plot_kwargs) -> plt.Axes:
    """
    Plot the intensity function of a fitted NHPP model.
    
    Args:
        model (NHPPFitter): A fitted NHPP model.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure is created.
        resolution (int, optional): Number of points to evaluate the intensity function at.
            Defaults to 100.
        show_events (bool, optional): Whether to show event times as tick marks on the x-axis.
            Defaults to True.
        label (str, optional): Label for the intensity curve. Defaults to "Fitted Intensity".
        color (str, optional): Color for the intensity curve. If None, use default color cycle.
        **plot_kwargs: Additional keyword arguments to pass to plot.
            
    Returns:
        matplotlib.axes.Axes: The axes containing the plot.
            
    Raises:
        RuntimeError: If the model has not been fitted.
    """
    if model.fitted_params is None:
        raise RuntimeError("Model not fitted.")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create time points for plotting
    t_plot = np.linspace(0, model.end_time, resolution)
    
    # Calculate intensity at each time point
    lambda_plot = model.predict_intensity(t_plot)
    
    # Plot intensity function
    ax.plot(t_plot, lambda_plot, label=label or "Fitted Intensity", color=color, **plot_kwargs)
    
    # Show event times if requested
    if show_events and model.n_events > 0:
        ax.plot(model.event_times, np.zeros_like(model.event_times), 
                linestyle='none', marker='|', color='k', 
                markersize=10, alpha=0.7, label='_nolegend_')
    
    # Set labels and title
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Intensity (Rate)", fontsize=12)
    ax.set_title("Fitted NHPP Intensity Function", fontsize=14)
    
    # Add legend and grid
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Ensure y-axis starts at 0 or slightly below
    ylim = ax.get_ylim()
    ax.set_ylim(min(0, ylim[0]), ylim[1])
    
    return ax


def plot_diagnostics_qq(model: NHPPFitter, 
                        ax=None, 
                        **plot_kwargs) -> plt.Axes:
    """
    Create a QQ plot comparing transformed inter-arrivals to the exponential distribution.
    
    For a correctly specified NHPP model, the transformed inter-arrivals should follow
    an exponential distribution with rate 1 (Exp(1)).
    
    Args:
        model (NHPPFitter): A fitted NHPP model.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure is created.
        **plot_kwargs: Additional keyword arguments to pass to plot.
            
    Returns:
        matplotlib.axes.Axes: The axes containing the plot.
            
    Raises:
        RuntimeError: If the model has not been fitted.
    """
    if model.fitted_params is None:
        raise RuntimeError("Model not fitted.")
    
    # Calculate transformed inter-arrivals
    deltas = model.calculate_transformed_interarrivals()
    
    # Need at least 2 points for QQ plot
    if len(deltas) < 2:
        warnings.warn("Not enough events for QQ plot. At least 2 events are required.")
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        return ax
    
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create QQ plot
    (osm, osr), (slope, intercept, r) = scipy.stats.probplot(
        deltas, dist=scipy.stats.expon, plot=ax)
    
    # Customize plot appearance
    ax.get_lines()[0].set_marker('o')  # Data points as circles
    ax.get_lines()[0].set_markerfacecolor('blue')
    ax.get_lines()[0].set_markeredgecolor('blue')
    ax.get_lines()[0].set_alpha(0.7)
    ax.get_lines()[0].set_markersize(6)
    
    ax.get_lines()[1].set_color('r')  # Reference line in red
    ax.get_lines()[1].set_linewidth(1.5)
    
    # Add correlation coefficient to title
    ax.set_title(f"Exp(1) QQ Plot of Transformed Inter-arrivals (r = {r:.3f})", fontsize=14)
    ax.set_xlabel("Theoretical Quantiles (Exponential rate=1)", fontsize=12)
    ax.set_ylabel("Sample Quantiles (Transformed Inter-arrivals)", fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add a diagonal line representing perfect fit
    min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    
    return ax


def plot_cumulative_intensity(model: NHPPFitter,
                             ax=None,
                             resolution: int = 100,
                             show_events: bool = True,
                             label: Optional[str] = None,
                             color: Optional[str] = None,
                             **plot_kwargs) -> plt.Axes:
    """
    Plot the cumulative intensity function Λ(t) of a fitted NHPP model.
    
    Args:
        model (NHPPFitter): A fitted NHPP model.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure is created.
        resolution (int, optional): Number of points to evaluate the function at.
        show_events (bool, optional): Whether to show cumulative events as a step function.
        label (str, optional): Label for the curve. Defaults to "Cumulative Intensity".
        color (str, optional): Color for the curve. If None, use default color cycle.
        **plot_kwargs: Additional keyword arguments to pass to plot.
            
    Returns:
        matplotlib.axes.Axes: The axes containing the plot.
            
    Raises:
        RuntimeError: If the model has not been fitted.
    """
    if model.fitted_params is None:
        raise RuntimeError("Model not fitted.")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create time points for plotting
    t_plot = np.linspace(0, model.end_time, resolution)
    
    # Calculate cumulative intensity at each time point
    cumulative_values = np.zeros_like(t_plot)
    for i, t in enumerate(t_plot):
        cumulative_values[i] = model._cumulative_intensity(t, model.fitted_params)
    
    # Plot cumulative intensity function
    ax.plot(t_plot, cumulative_values, 
            label=label or "Cumulative Intensity", 
            color=color, **plot_kwargs)
    
    # Show cumulative events if requested
    if show_events and model.n_events > 0:
        # Create step function of cumulative event counts
        event_times = np.sort(model.event_times)
        cum_counts = np.arange(1, len(event_times) + 1)
        
        # Add a point at t=0, count=0 for the step function
        event_times = np.insert(event_times, 0, 0)
        cum_counts = np.insert(cum_counts, 0, 0)
        
        # Plot step function of cumulative event counts
        ax.step(event_times, cum_counts, where='post', 
                label='Cumulative Events', color='k', alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Cumulative Intensity / Count", fontsize=12)
    ax.set_title("Cumulative Intensity Function", fontsize=14)
    
    # Add legend and grid
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Ensure y-axis starts at 0
    ax.set_ylim(0, ax.get_ylim()[1])
    
    return ax


def plot_residual_histogram(model: NHPPFitter,
                           ax=None,
                           bins: Union[int, List, str] = 'auto',
                           density: bool = True,
                           **hist_kwargs) -> plt.Axes:
    """
    Plot a histogram of transformed inter-arrivals with the Exp(1) density.
    
    For a correctly specified NHPP model, the transformed inter-arrivals should follow
    an exponential distribution with rate 1 (Exp(1)).
    
    Args:
        model (NHPPFitter): A fitted NHPP model.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure is created.
        bins (int, sequence, or str, optional): Specification of histogram bins.
            Default is 'auto'.
        density (bool, optional): If True, the result is a normalized histogram
            that can be compared to the Exp(1) PDF. Default is True.
        **hist_kwargs: Additional keyword arguments to pass to hist.
            
    Returns:
        matplotlib.axes.Axes: The axes containing the plot.
            
    Raises:
        RuntimeError: If the model has not been fitted.
    """
    if model.fitted_params is None:
        raise RuntimeError("Model not fitted.")
    
    # Calculate transformed inter-arrivals
    deltas = model.calculate_transformed_interarrivals()
    
    if len(deltas) < 1:
        warnings.warn("No events for histogram.")
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        return ax
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    hist_kwargs.setdefault('alpha', 0.7)
    hist_kwargs.setdefault('color', 'skyblue')
    hist_kwargs.setdefault('edgecolor', 'black')
    
    ax.hist(deltas, bins=bins, density=density, label='Transformed Inter-arrivals', **hist_kwargs)
    
    # Overlay the Exp(1) density if the histogram is normalized
    if density:
        x = np.linspace(0, max(5, np.max(deltas) * 1.2), 1000)
        exp_pdf = scipy.stats.expon.pdf(x)
        ax.plot(x, exp_pdf, 'r-', linewidth=2, label='Exp(1) PDF')
    
    # Set labels and title
    ax.set_xlabel("Transformed Inter-arrival Time", fontsize=12)
    if density:
        ax.set_ylabel("Probability Density", fontsize=12)
    else:
        ax.set_ylabel("Count", fontsize=12)
    
    # Calculate KS test statistic and p-value
    ks_statistic, p_value = scipy.stats.kstest(deltas, 'expon')
    
    ax.set_title(f"Histogram of Transformed Inter-arrivals\nKS test: p-value = {p_value:.3f}", 
                fontsize=14)
    
    # Add legend and grid
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return ax


def plot_multiple_intensities(models: List[NHPPFitter], 
                             labels: List[str] = None,
                             colors: List[str] = None,
                             resolution: int = 100,
                             show_events: bool = False,
                             ax=None,
                             **plot_kwargs) -> plt.Axes:
    """
    Plot multiple intensity functions on the same axes for comparison.
    
    Args:
        models (List[NHPPFitter]): List of fitted NHPP models to compare.
        labels (List[str], optional): Labels for each model. If None, uses "Model 1", "Model 2", etc.
        colors (List[str], optional): Colors for each model. If None, uses the default color cycle.
        resolution (int, optional): Number of points to evaluate the intensity functions at.
        show_events (bool, optional): Whether to show event times as tick marks.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure is created.
        **plot_kwargs: Additional keyword arguments to pass to plot.
            
    Returns:
        matplotlib.axes.Axes: The axes containing the plot.
            
    Raises:
        RuntimeError: If any model has not been fitted.
    """
    if not models:
        raise ValueError("No models provided for comparison.")
    
    # Check if all models are fitted
    for i, model in enumerate(models):
        if model.fitted_params is None:
            raise RuntimeError(f"Model {i+1} not fitted.")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use default labels if not provided
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(models))]
    
    # Ensure we have enough labels
    if len(labels) < len(models):
        labels.extend([f"Model {i+1}" for i in range(len(labels), len(models))])
    
    # Find global end time (for consistent x-axis)
    global_end_time = max(model.end_time for model in models)
    
    # Plot each model's intensity function
    for i, model in enumerate(models):
        # Create time points for this model (up to its end_time)
        t_plot = np.linspace(0, model.end_time, resolution)
        
        # Calculate intensity at each time point
        lambda_plot = model.predict_intensity(t_plot)
        
        # Plot intensity function with specified label and color (if provided)
        color = colors[i] if colors and i < len(colors) else None
        ax.plot(t_plot, lambda_plot, label=labels[i], color=color, **plot_kwargs)
        
        # Show event times if requested
        if show_events and model.n_events > 0:
            color = colors[i] if colors and i < len(colors) else 'k'
            alpha = 0.5 / (i + 1)  # Decrease alpha for subsequent models
            ax.plot(model.event_times, np.zeros_like(model.event_times), 
                    linestyle='none', marker='|', color=color, 
                    markersize=8, alpha=alpha, label='_nolegend_')
    
    # Set labels and title
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Intensity (Rate)", fontsize=12)
    ax.set_title("Comparison of NHPP Intensity Functions", fontsize=14)
    
    # Add legend and grid
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Ensure x-axis includes all models
    ax.set_xlim(0, global_end_time)
    
    # Ensure y-axis starts at 0 or slightly below
    ylim = ax.get_ylim()
    ax.set_ylim(min(0, ylim[0]), ylim[1])
    
    return ax


def create_diagnostic_plots(model: NHPPFitter, figsize: Tuple[float, float] = (12, 10)) -> plt.Figure:
    """
    Create a comprehensive set of diagnostic plots for model assessment.
    
    Creates a 2x2 grid of plots:
    1. Intensity function with events
    2. Cumulative intensity with cumulative events
    3. QQ plot of transformed inter-arrivals against Exp(1)
    4. Histogram of transformed inter-arrivals with Exp(1) PDF
    
    Args:
        model (NHPPFitter): A fitted NHPP model.
        figsize (Tuple[float, float], optional): Figure size. Defaults to (12, 10).
            
    Returns:
        matplotlib.figure.Figure: The figure containing all diagnostic plots.
            
    Raises:
        RuntimeError: If the model has not been fitted.
    """
    if model.fitted_params is None:
        raise RuntimeError("Model not fitted.")
    
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Flatten for easier indexing
    axs = axs.flatten()
    
    # Plot 1: Intensity function
    plot_intensity(model, ax=axs[0], show_events=True)
    
    # Plot 2: Cumulative intensity
    plot_cumulative_intensity(model, ax=axs[1], show_events=True)
    
    # Plot 3: QQ plot
    plot_diagnostics_qq(model, ax=axs[2])
    
    # Plot 4: Histogram
    plot_residual_histogram(model, ax=axs[3], bins='auto')
    
    # Add a title to the entire figure
    fig.suptitle("NHPP Model Diagnostic Plots", fontsize=16)
    
    # Adjust layout
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    
    return fig