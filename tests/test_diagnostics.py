"""
Tests for the diagnostic functionality of the NHPP implementation.
"""

import numpy as np
import pytest
from scipy import stats

from ppbox import NHPPFitter
from ppbox.intensity_functions import LinearIntensity, LogLinearIntensity


@pytest.fixture
def simulated_hpp_model():
    """
    Fixture providing a fitted model with data simulated from HPP.
    
    For a homogeneous Poisson process (HPP), the transformed interarrivals
    should follow Exp(1) if the model is correctly specified.
    """
    # Define true parameters for HPP
    rate = 0.5  # constant rate
    end_time = 100.0
    
    # Use LogLinearIntensity with β₁ = 0 to simulate HPP
    beta0 = np.log(rate)
    beta1 = 0.0
    
    # Create covariate (constant value)
    covariate_times = np.array([0.0, end_time])
    covariate_values = np.array([1.0, 1.0])
    
    # Create simulator
    simulator = NHPPFitter.create_with_log_linear_intensity(
        event_times=np.array([]),
        covariate_times=covariate_times,
        covariate_values=covariate_values,
        end_time=end_time
    )
    
    # Simulate data
    np.random.seed(42)
    true_params = np.array([beta0, beta1])
    event_times = simulator.simulate(true_params, end_time)
    
    # Create model with simulated data
    model = NHPPFitter.create_with_log_linear_intensity(
        event_times=event_times,
        covariate_times=covariate_times,
        covariate_values=covariate_values,
        end_time=end_time
    )
    
    # Set true parameters (to ensure correct diagnostics)
    model.fitted_params = true_params
    
    return model


@pytest.fixture
def simulated_linear_model():
    """
    Fixture providing a fitted model with data simulated from a linear intensity process.
    """
    # Define true parameters for linear intensity
    alpha, beta = 0.1, 0.01  # λ(t) = α + βt
    end_time = 100.0
    
    # Create simulator
    simulator = NHPPFitter.create_with_linear_intensity(
        event_times=np.array([]),
        end_time=end_time
    )
    
    # Simulate data
    np.random.seed(42)
    true_params = np.array([alpha, beta])
    
    # For simulation, we'll use the time transformation method
    # Generate cumulative intensity at end time
    lambda_T = alpha * end_time + beta * end_time**2 / 2
    n_events = np.random.poisson(lambda_T)
    u = np.sort(np.random.uniform(0, lambda_T, n_events))
    
    # Solve for event times
    event_times = np.zeros(n_events)
    for i, u_i in enumerate(u):
        # Solve βt²/2 + αt - u = 0
        a = beta / 2
        b = alpha
        c = -u_i
        
        # Quadratic formula
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            discriminant = 0
        t_i = (-b + np.sqrt(discriminant)) / (2 * a)
        
        t_i = min(max(0, t_i), end_time)
        event_times[i] = t_i
    
    # Create model with simulated data
    model = NHPPFitter.create_with_linear_intensity(
        event_times=event_times,
        end_time=end_time
    )
    
    # Set true parameters
    model.fitted_params = true_params
    
    return model


def test_calculate_transformed_interarrivals_hpp(simulated_hpp_model):
    """Test calculation of transformed interarrivals for HPP."""
    model = simulated_hpp_model
    
    # Calculate transformed interarrivals
    deltas = model.calculate_transformed_interarrivals()
    
    # Check basic properties
    assert len(deltas) == model.n_events
    assert np.all(deltas >= 0)
    
    # For a correctly specified model, deltas should follow Exp(1)
    assert np.isclose(np.mean(deltas), 1.0, rtol=0.2)
    
    # Test distribution using Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.kstest(deltas, stats.expon(scale=1.0).cdf)
    
    # We don't want to reject H0, so p-value should be > 0.05
    assert p_value > 0.05, f"KS test failed: statistic={ks_statistic}, p={p_value}"


def test_calculate_transformed_interarrivals_linear(simulated_linear_model):
    """Test calculation of transformed interarrivals for linear intensity."""
    model = simulated_linear_model
    
    # Calculate transformed interarrivals
    deltas = model.calculate_transformed_interarrivals()
    
    # Check basic properties
    assert len(deltas) == model.n_events
    assert np.all(deltas >= 0)
    
    # For a correctly specified model, deltas should follow Exp(1)
    assert np.isclose(np.mean(deltas), 1.0, rtol=0.2)
    
    # Test distribution using Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.kstest(deltas, stats.expon(scale=1.0).cdf)
    
    # We don't want to reject H0, so p-value should be > 0.05
    assert p_value > 0.05, f"KS test failed: statistic={ks_statistic}, p={p_value}"


def test_calculate_transformed_interarrivals_empty():
    """Test calculation of transformed interarrivals with no events."""
    # Create a model with no events
    model = NHPPFitter.create_with_linear_intensity(
        event_times=np.array([]),
        end_time=10.0
    )
    
    # Set fitted_params manually (since we can't fit with no events)
    model.fitted_params = np.array([0.1, 0.01])
    
    # Calculate transformed interarrivals
    deltas = model.calculate_transformed_interarrivals()
    
    # Check that result is an empty array
    assert isinstance(deltas, np.ndarray)
    assert len(deltas) == 0


def test_calculate_transformed_interarrivals_unfitted():
    """Test that calculate_transformed_interarrivals raises error when model is not fitted."""
    # Create an unfitted model
    model = NHPPFitter.create_with_linear_intensity(
        event_times=np.array([1.0, 2.0, 3.0]),
        end_time=10.0
    )
    
    # Check that calculate_transformed_interarrivals raises RuntimeError
    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        model.calculate_transformed_interarrivals()