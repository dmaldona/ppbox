"""
Tests for the prediction functionality of the NHPP implementation.
"""

import numpy as np
import pytest

from ppbox import NHPPFitter
from ppbox.intensity_functions import LinearIntensity, LogLinearIntensity


@pytest.fixture
def linear_model_fitted():
    """Fixture providing a fitted linear intensity model."""
    # Create and fit a model with LinearIntensity
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
    
    return model, alpha, beta


@pytest.fixture
def loglinear_model_fitted():
    """Fixture providing a fitted log-linear intensity model."""
    # Create and fit a model with LogLinearIntensity
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
    
    return model, beta0, beta1


def test_predict_intensity_linear(linear_model_fitted):
    """Test predict_intensity method with LinearIntensity model."""
    # Get model and parameters from fixture
    model, alpha, beta = linear_model_fitted
    
    # Define times for prediction
    pred_times = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    
    # Predict intensities
    intensities = model.predict_intensity(pred_times)
    
    # Calculate expected intensities
    expected_intensities = alpha + beta * pred_times
    
    # Check results
    assert len(intensities) == len(pred_times)
    assert np.allclose(intensities, expected_intensities)


def test_predict_intensity_loglinear(loglinear_model_fitted):
    """Test predict_intensity method with LogLinearIntensity model."""
    # Get model and parameters from fixture
    model, beta0, beta1 = loglinear_model_fitted
    
    # Define times for prediction
    pred_times = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    
    # Predict intensities
    intensities = model.predict_intensity(pred_times)
    
    # Calculate expected intensities
    # For log-linear, λ(t) = exp(β₀ + β₁*w(t))
    w_t = model.intensity_function.get_covariate_at_time(pred_times)
    expected_intensities = np.exp(beta0 + beta1 * w_t)
    
    # Check results
    assert len(intensities) == len(pred_times)
    assert np.allclose(intensities, expected_intensities)


def test_predict_intensity_unfitted():
    """Test predict_intensity raises error when model is not fitted."""
    # Create an unfitted model
    event_times = np.array([1.0, 2.0, 3.0])
    end_time = 10.0
    
    model = NHPPFitter.create_with_linear_intensity(
        event_times=event_times,
        end_time=end_time
    )
    
    # Check that predict_intensity raises RuntimeError
    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        model.predict_intensity(np.array([1.0, 2.0]))