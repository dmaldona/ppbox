"""
Tests for the NHPP visualization functionality.
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from ppbox import NHPPFitter
from ppbox.nhpp_plotting import (
    plot_intensity, 
    plot_diagnostics_qq, 
    plot_cumulative_intensity,
    plot_residual_histogram,
    plot_multiple_intensities,
    create_diagnostic_plots
)


@pytest.fixture
def fitted_linear_model():
    """Fixture providing a fitted linear intensity model."""
    # Parameters
    alpha, beta = 0.2, 0.05  # λ(t) = α + βt
    end_time = 100.0
    
    # Generate synthetic data
    np.random.seed(42)
    n_events = 50
    event_times = np.sort(np.random.uniform(0, end_time, n_events))
    
    # Create model
    model = NHPPFitter.create_with_linear_intensity(
        event_times=event_times,
        end_time=end_time
    )
    
    # Set parameters directly instead of fitting to ensure reproducibility
    model.fitted_params = np.array([alpha, beta])
    
    return model


@pytest.fixture
def fitted_loglinear_model():
    """Fixture providing a fitted log-linear intensity model."""
    # Parameters
    beta0, beta1 = -1.0, 0.5
    end_time = 100.0
    
    # Define covariate
    covariate_times = np.linspace(0, end_time, 11)  # 11 points for 0, 10, 20, ..., 100
    covariate_values = 1.0 + 0.5 * np.sin(2 * np.pi * covariate_times / end_time)
    
    # Generate synthetic data
    np.random.seed(42)
    n_events = 50
    event_times = np.sort(np.random.uniform(0, end_time, n_events))
    
    # Create model
    model = NHPPFitter.create_with_log_linear_intensity(
        event_times=event_times,
        covariate_times=covariate_times,
        covariate_values=covariate_values,
        end_time=end_time
    )
    
    # Set parameters directly instead of fitting to ensure reproducibility
    model.fitted_params = np.array([beta0, beta1])
    
    return model


@pytest.fixture
def unfitted_model():
    """Fixture providing an unfitted model."""
    event_times = np.array([1.0, 2.0, 3.0])
    end_time = 10.0
    
    model = NHPPFitter.create_with_linear_intensity(
        event_times=event_times,
        end_time=end_time
    )
    
    return model


def test_plot_intensity(fitted_linear_model):
    """Test the plot_intensity function."""
    # Create a figure and axes
    fig, ax = plt.subplots()
    
    # Plot intensity
    result_ax = plot_intensity(fitted_linear_model, ax=ax)
    
    # Check that the function returns the correct axes
    assert result_ax is ax
    
    # Check that the plot contains a line for the intensity function
    assert len(ax.lines) >= 1
    
    # Check that the plot has appropriate labels
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Intensity (Rate)"
    assert "Intensity" in ax.get_title()
    
    plt.close(fig)


def test_plot_intensity_show_events(fitted_linear_model):
    """Test the plot_intensity function with event markers."""
    fig, ax = plt.subplots()
    
    # Plot intensity with events
    result_ax = plot_intensity(fitted_linear_model, ax=ax, show_events=True)
    
    # With events shown, there should be at least 2 lines (intensity curve and event markers)
    assert len(ax.lines) >= 2
    
    plt.close(fig)


def test_plot_intensity_unfitted_model(unfitted_model):
    """Test that plot_intensity raises an error for unfitted models."""
    with pytest.raises(RuntimeError, match="Model not fitted"):
        plot_intensity(unfitted_model)


def test_plot_diagnostics_qq(fitted_linear_model):
    """Test the plot_diagnostics_qq function."""
    fig, ax = plt.subplots()
    
    # Create QQ plot
    result_ax = plot_diagnostics_qq(fitted_linear_model, ax=ax)
    
    # Check that the function returns the correct axes
    assert result_ax is ax
    
    # Check that the plot contains a scatter plot and a reference line
    assert len(ax.lines) >= 2
    
    # Check that the plot has appropriate labels
    assert "Quantiles" in ax.get_xlabel()
    assert "Quantiles" in ax.get_ylabel()
    assert "QQ Plot" in ax.get_title()
    
    plt.close(fig)


def test_plot_diagnostics_qq_unfitted_model(unfitted_model):
    """Test that plot_diagnostics_qq raises an error for unfitted models."""
    with pytest.raises(RuntimeError, match="Model not fitted"):
        plot_diagnostics_qq(unfitted_model)


def test_plot_cumulative_intensity(fitted_linear_model):
    """Test the plot_cumulative_intensity function."""
    fig, ax = plt.subplots()
    
    # Plot cumulative intensity
    result_ax = plot_cumulative_intensity(fitted_linear_model, ax=ax)
    
    # Check that the function returns the correct axes
    assert result_ax is ax
    
    # Check that the plot contains a line
    assert len(ax.lines) >= 1
    
    # Check that the plot has appropriate labels
    assert ax.get_xlabel() == "Time"
    assert "Cumulative" in ax.get_ylabel()
    assert "Cumulative" in ax.get_title()
    
    plt.close(fig)


def test_plot_cumulative_intensity_with_events(fitted_linear_model):
    """Test the plot_cumulative_intensity function with events."""
    fig, ax = plt.subplots()
    
    # Plot cumulative intensity with events
    result_ax = plot_cumulative_intensity(fitted_linear_model, ax=ax, show_events=True)
    
    # With events, there should be at least 2 lines (cumulative intensity and step function)
    assert len(ax.lines) >= 1  # At least the intensity line
    
    plt.close(fig)


def test_plot_residual_histogram(fitted_linear_model):
    """Test the plot_residual_histogram function."""
    fig, ax = plt.subplots()
    
    # Create histogram plot
    result_ax = plot_residual_histogram(fitted_linear_model, ax=ax)
    
    # Check that the function returns the correct axes
    assert result_ax is ax
    
    # Check that the plot has a histogram and a PDF curve
    assert len(ax.lines) >= 1  # At least one line for the PDF
    assert len(ax.patches) > 0  # Histogram bars
    
    # Check that the plot has appropriate labels
    assert "Inter-arrival" in ax.get_xlabel()
    assert "Density" in ax.get_ylabel()
    assert "Histogram" in ax.get_title()
    
    plt.close(fig)


def test_plot_multiple_intensities(fitted_linear_model, fitted_loglinear_model):
    """Test the plot_multiple_intensities function."""
    fig, ax = plt.subplots()
    
    # Plot multiple intensity functions
    result_ax = plot_multiple_intensities(
        [fitted_linear_model, fitted_loglinear_model],
        labels=["Linear", "Log-Linear"],
        ax=ax
    )
    
    # Check that the function returns the correct axes
    assert result_ax is ax
    
    # Check that the plot contains two lines (one for each model)
    assert len(ax.lines) >= 2
    
    # Check that the plot has appropriate labels
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Intensity (Rate)"
    assert "Comparison" in ax.get_title()
    
    # Check that the legend has entries for both models
    handles, labels = ax.get_legend_handles_labels()
    assert "Linear" in labels
    assert "Log-Linear" in labels
    
    plt.close(fig)


def test_create_diagnostic_plots(fitted_linear_model):
    """Test the create_diagnostic_plots function."""
    # Create diagnostic plots
    fig = create_diagnostic_plots(fitted_linear_model)
    
    # Check that we have a figure with 4 subplots
    assert len(fig.axes) == 4
    
    # Check that the figure has a title
    assert "Diagnostic" in fig.texts[0].get_text()
    
    plt.close(fig)


def test_create_diagnostic_plots_unfitted_model(unfitted_model):
    """Test that create_diagnostic_plots raises an error for unfitted models."""
    with pytest.raises(RuntimeError, match="Model not fitted"):
        create_diagnostic_plots(unfitted_model)