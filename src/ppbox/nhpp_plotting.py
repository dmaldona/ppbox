"""
Visualization module for NHPP models.

This module provides functions for creating visualizations of NHPP models,
including intensity function plots and diagnostic plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import warnings
from typing import Optional, Tuple, List, Union, Dict

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

def plot_uniform_order_statistic(model: NHPPFitter,
                                ax=None,
                                **plot_kwargs) -> plt.Axes:
    """
    Plot normalized transformed times against expected Uniform(0,1) order statistics.

    This diagnostic plot compares Lambda(S_i)/Lambda(T) against their
    expected values k/(n+1) under the Uniform(0,1) order statistic assumption.
    Points close to the y=x line suggest a good model fit.

    Args:
        model (NHPPFitter): A fitted NHPP model.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure/axes is created.
        **plot_kwargs: Additional keyword arguments passed to ax.plot for the points.

    Returns:
        matplotlib.axes.Axes: The axes containing the plot.

    Raises:
        RuntimeError: If the model has not been fitted.
        ValueError: If calculation of normalized times fails or yields no points.
    """
    if model.fitted_params is None:
        raise RuntimeError("Model has not been fitted yet.")

    # Calculate observed normalized transformed times
    observed_U = model.calculate_normalized_transformed_times()
    n = len(observed_U)

    if n < 1:
        warnings.warn("Not enough events for Uniform Order Statistic plot.")
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(0.5, 0.5, 'Not enough events for plot', horizontalalignment='center', verticalalignment='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return ax

    # Calculate expected values for U(0,1) order statistics: E[U_(k)] = k / (n + 1)
    expected_U = np.arange(1, n + 1) / (n + 1.0)

    # Create figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Create the scatter plot
    plot_defaults = {'marker': 'o', 'linestyle': 'none', 'alpha': 0.7, 'markersize': 6}
    plot_defaults.update(plot_kwargs) # Allow user overrides
    ax.plot(expected_U, observed_U, **plot_defaults)

    # Add y=x reference line
    ax.plot([0, 1], [0, 1], color='r', linestyle='--', linewidth=1.5, label='y=x line')

    # Customize plot appearance
    ax.set_title("Uniform Order Statistic Diagnostic Plot", fontsize=14)
    ax.set_xlabel("Expected $U(0,1)$ Order Statistics ($k/(n+1)$)", fontsize=12)
    ax.set_ylabel("Observed Normalized Transformed Times ($\\Lambda(S_k)/\\Lambda(T)$)", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box') # Ensure square plot for y=x line clarity
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend()

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

def plot_empirical_vs_fitted_rates(model: NHPPFitter,
                                   method: str = 'disjoint',
                                   interval_length: Optional[float] = None,
                                   num_intervals: Optional[int] = None,
                                   resolution: int = 100,
                                   show_confidence: bool = True,
                                   confidence_level: float = 0.95,
                                   confidence_method: str = 'transformation',
                                   ax=None,
                                   **plot_kwargs) -> plt.Axes:
    """
    Plot empirical rates vs fitted intensity function with optional confidence intervals.
    
    Args:
        model: Fitted NHPP model.
        method: 'disjoint' or 'overlapping'.
        interval_length: Length of intervals for rate calculation.
        num_intervals: Number of intervals (for disjoint method only).
        resolution: Number of points for fitted intensity evaluation.
        show_confidence: Whether to show confidence intervals for fitted intensity.
        confidence_level: Confidence level for intervals (default 0.95).
        confidence_method: Method for CI calculation ('transformation' or 'delta').
        ax: Matplotlib axes.
        **plot_kwargs: Additional plotting arguments.
        
    Returns:
        matplotlib.axes.Axes: The axes containing the plot.
    """
    if model.fitted_params is None:
        raise RuntimeError("Model not fitted.")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate empirical rates
    if method == 'disjoint':
        if interval_length is None and num_intervals is None:
            num_intervals = min(20, max(5, model.n_events // 3))  # Default heuristic
        emp_times, emp_rates = model.calculate_empirical_rates_disjoint(
            interval_length=interval_length, num_intervals=num_intervals)
        plot_type = 'step'
    elif method == 'overlapping':
        if interval_length is None:
            interval_length = model.end_time / 10  # Default heuristic
        emp_times, emp_rates = model.calculate_empirical_rates_overlapping(
            interval_length=interval_length, resolution=resolution)
        plot_type = 'line'
    else:
        raise ValueError("method must be 'disjoint' or 'overlapping'")
    
    # Plot empirical rates
    if plot_type == 'step':
        ax.step(emp_times, emp_rates, where='mid', 
                label='Empirical Rate', color='black', alpha=0.8, linewidth=1.5)
    else:
        ax.plot(emp_times, emp_rates, 
                label='Empirical Rate', color='black', alpha=0.8, linewidth=1.5)
    
    # Plot fitted intensity with confidence intervals
    t_fitted = np.linspace(0, model.end_time, resolution)
    fitted_intensity = model.predict_intensity(t_fitted)
    
    # Plot fitted intensity line
    ax.plot(t_fitted, fitted_intensity, 
            label='Fitted Intensity', color='red', linewidth=2)
    
    # Add confidence intervals if requested
    if show_confidence:
        try:
            print("caca", confidence_method)
            lower_ci, upper_ci = model.calculate_intensity_confidence_intervals(
                times=t_fitted,
                confidence_level=confidence_level,
                method=confidence_method
            )
            
            # Plot confidence band
            ax.fill_between(t_fitted, lower_ci, upper_ci, 
                          color='red', alpha=0.2, 
                          label=f'{confidence_level*100:.0f}% Confidence Band')
            
            # Optionally plot CI boundaries as lines
            ax.plot(t_fitted, lower_ci, color='red', linestyle='--', 
                   alpha=0.6, linewidth=1, label='_nolegend_')
            ax.plot(t_fitted, upper_ci, color='red', linestyle='--', 
                   alpha=0.6, linewidth=1, label='_nolegend_')
                   
        except Exception as e:
            warnings.warn(f"Could not calculate confidence intervals: {e}")
    
    # Formatting
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Rate / Intensity", fontsize=12)
    
    # Enhanced title with method info
    title_parts = [f"Empirical vs Fitted Rates ({method.title()} Method)"]
    if show_confidence:
        title_parts.append(f"with {confidence_level*100:.0f}% CI")
    ax.set_title(" ".join(title_parts), fontsize=14)
    
    # Legend and grid
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Ensure y-axis starts at 0 or slightly below minimum
    y_min = min(ax.get_ylim()[0], 0)
    if show_confidence:
        # Account for confidence intervals in y-limits
        try:
            y_min = min(y_min, np.min(lower_ci) * 0.95)
        except:
            pass
    ax.set_ylim(y_min, ax.get_ylim()[1])
    
    return ax

def plot_raw_residuals_vs_time(model: NHPPFitter,
                              method: str = 'disjoint',
                              interval_length: Optional[float] = None,
                              num_intervals: Optional[int] = None,
                              resolution: int = 100,
                              residual_type: str = 'raw',
                              show_confidence_bands: bool = True,
                              add_lowess: bool = True,
                              ax=None,
                              **plot_kwargs) -> plt.Axes:
    """
    Plot raw residuals against time.
    
    Args:
        model: Fitted NHPP model.
        method: 'disjoint' or 'overlapping'.
        interval_length: Length of intervals.
        num_intervals: Number of intervals (disjoint only).
        resolution: Number of points (overlapping only).
        residual_type: 'raw' or 'pearson'.
        show_confidence_bands: Whether to show ±2 confidence bands.
        add_lowess: Whether to add LOWESS smoother.
        ax: Matplotlib axes.
        
    Returns:
        matplotlib.axes.Axes: The axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate residuals
    if method == 'disjoint':
        if interval_length is None and num_intervals is None:
            num_intervals = min(20, max(5, model.n_events // 3))
        residual_data = model.calculate_raw_residuals_disjoint(
            interval_length=interval_length, 
            num_intervals=num_intervals,
            residual_type=residual_type)
    elif method == 'overlapping':
        if interval_length is None:
            interval_length = model.end_time / 10
        residual_data = model.calculate_raw_residuals_overlapping(
            interval_length=interval_length,
            resolution=resolution,
            residual_type=residual_type)
    else:
        raise ValueError("method must be 'disjoint' or 'overlapping'")
    
    time_points = residual_data['time_points']
    residuals = residual_data['residuals']
    interval_lengths = residual_data['interval_lengths']
    
    # Plot residuals
    ax.scatter(time_points, residuals, alpha=0.7, s=30, **plot_kwargs)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # Add confidence bands
    if show_confidence_bands:
        if residual_type == 'pearson':
            # The bands should depend on the interval length: ±2/sqrt(L)
            # This matches the reference R code: ic <- 2 / lint**0.5
            with np.errstate(divide='ignore', invalid='ignore'):
                half_width = 2.0 / np.sqrt(interval_lengths)
            
            upper_band = half_width
            lower_band = -half_width
            
            # Sort for cleaner plotting if using disjoint intervals
            sort_idx = np.argsort(time_points)

            ax.plot(time_points[sort_idx], upper_band[sort_idx], color='red', linestyle='--', alpha=0.7)
            ax.plot(time_points[sort_idx], lower_band[sort_idx], color='red', linestyle='--', alpha=0.7)
            ax.fill_between(time_points[sort_idx], lower_band[sort_idx], upper_band[sort_idx],
                            color='red', alpha=0.1, label='±2/√L bands')
        else:
            # For raw residuals: ±2*sqrt(fitted_rate/interval_length)
            fitted_rates = residual_data['fitted_rates']
            upper_band = 2 * np.sqrt(fitted_rates / interval_lengths)
            lower_band = -upper_band
            
            ax.plot(time_points, upper_band, color='red', linestyle='--', alpha=0.7)
            ax.plot(time_points, lower_band, color='red', linestyle='--', alpha=0.7)
            ax.fill_between(time_points, lower_band, upper_band, 
                          color='red', alpha=0.1, label='±2σ bands')
    
    # Add LOWESS smoother
    if add_lowess and len(residuals) > 5:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(residuals, time_points, frac=0.3)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color='blue', linewidth=2, 
                   alpha=0.8, label='LOWESS')
        except ImportError:
            warnings.warn("statsmodels not available for LOWESS smoother")
    
    # Formatting
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel(f"{residual_type.title()} Residuals", fontsize=12)
    ax.set_title(f"{residual_type.title()} Residuals vs Time ({method.title()} Method)", 
                fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend()
    
    return ax

def plot_lurking_variable_plots(model: NHPPFitter,
                               covariate_data: Dict[str, np.ndarray],
                               covariate_times: Optional[np.ndarray] = None,
                               num_intervals: int = 20,
                               residual_type: str = 'raw',
                               show_confidence_bands: bool = True,
                               time_step: float = 1.0,
                               figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
    """
    Create lurking variable plots for multiple covariates.
    
    Args:
        model: Fitted NHPP model.
        covariate_data: Dict with covariate names as keys and values as arrays.
        covariate_times: Times corresponding to covariate values.
        num_intervals: Number of intervals to divide covariate range into.
        residual_type: 'raw' or 'pearson'.
        show_confidence_bands: Whether to show variable confidence bands.
        time_step: Duration that each covariate measurement represents.
        figsize: Figure size.
        
    Returns:
        matplotlib.figure.Figure: Figure containing all lurking variable plots.
    """
    n_covariates = len(covariate_data)
    
    if n_covariates == 0:
        raise ValueError("No covariates provided")
    
    # Determine subplot layout
    if n_covariates == 1:
        nrows, ncols = 1, 1
    elif n_covariates == 2:
        nrows, ncols = 1, 2
    elif n_covariates <= 4:
        nrows, ncols = 2, 2
    elif n_covariates <= 6:
        nrows, ncols = 2, 3
    elif n_covariates <= 9:
        nrows, ncols = 3, 3
    else:
        nrows, ncols = 4, 3  # For more than 9 covariates
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Handle single subplot case
    if n_covariates == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (covariate_name, covariate_values) in enumerate(covariate_data.items()):
        if i >= len(axes):
            warnings.warn(f"Too many covariates ({n_covariates}). Only plotting first {len(axes)}.")
            break
            
        ax = axes[i]
        
        try:
            # Calculate lurking variable residuals
            lurking_data = model.calculate_lurking_variable_residuals(
                covariate_values=covariate_values,
                covariate_times=covariate_times,
                num_intervals=num_intervals,
                residual_type=residual_type,
                time_step=time_step)
            
            covariate_midpoints = lurking_data['covariate_midpoints']
            residuals = lurking_data['residuals']
            points_per_bin = lurking_data['points_per_bin']
            
            if len(residuals) == 0:
                ax.text(0.5, 0.5, f'No valid data\nfor {covariate_name}', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f"Lurking Variable Plot: {covariate_name}", fontsize=12)
                continue
            
            # Plot residuals vs covariate values
            ax.scatter(covariate_midpoints, residuals, alpha=0.7, s=40, 
                      color='blue', edgecolors='black', linewidths=0.5)
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
            
            # Add variable confidence bands based on points per bin
            if show_confidence_bands:
                if residual_type == 'pearson':
                    # Calculate confidence bands: ±2/√(points_per_bin)
                    # Following NHPoisson's approach: ic <- 2 / lintV**0.5
                    upper_bands = 2.0 / np.sqrt(points_per_bin)
                    lower_bands = -upper_bands
                    
                    # Sort by covariate value for proper line plotting
                    sort_idx = np.argsort(covariate_midpoints)
                    sorted_cov = covariate_midpoints[sort_idx]
                    sorted_upper = upper_bands[sort_idx]
                    sorted_lower = lower_bands[sort_idx]
                    
                    # Plot confidence bands
                    ax.plot(sorted_cov, sorted_upper, color='red', linestyle='--', 
                           alpha=0.7, linewidth=1.5, label='±2/√N bands' if i == 0 else '_nolegend_')
                    ax.plot(sorted_cov, sorted_lower, color='red', linestyle='--', 
                           alpha=0.7, linewidth=1.5, label='_nolegend_')
                    
                    # Fill between the bands
                    ax.fill_between(sorted_cov, sorted_lower, sorted_upper, 
                                  color='red', alpha=0.1)
                    
                elif residual_type == 'raw':
                    # For raw residuals, confidence bands are more complex
                    # They depend on fitted rates within each bin
                    # For now, show a warning that this is not implemented
                    if i == 0:  # Only warn once
                        warnings.warn("Variable confidence bands for raw residuals not implemented. "
                                    "Consider using residual_type='pearson' for confidence bands.")
            
            # Formatting
            ax.set_xlabel(covariate_name, fontsize=11)
            ax.set_ylabel(f"{residual_type.title()} Residuals", fontsize=11)
            ax.set_title(f"Lurking Variable Plot: {covariate_name}", fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add legend only to first plot to avoid clutter
            if i == 0 and show_confidence_bands and residual_type == 'pearson':
                ax.legend(fontsize=10, loc='best')
            
        except Exception as e:
            # Handle errors gracefully
            ax.text(0.5, 0.5, f'Error calculating\nresiduals for {covariate_name}\n{str(e)}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=10, color='red')
            ax.set_title(f"Lurking Variable Plot: {covariate_name} (Error)", fontsize=12)
            warnings.warn(f"Error calculating lurking variable residuals for {covariate_name}: {e}")
    
    # Hide unused subplots
    for i in range(n_covariates, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    fig.suptitle(f"Lurking Variable Plots ({residual_type.title()} Residuals)", 
                fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for suptitle
    
    return fig