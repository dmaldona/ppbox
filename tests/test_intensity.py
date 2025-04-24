import numpy as np
import pytest

from ppbox import NHPPFitter
from ppbox.intensity_functions import LinearIntensity, LogLinearIntensity

def test_linear_intensity_model():
    """Test the NHPPFitter with LinearIntensity."""
    # Test data
    event_times = np.array([2.0, 5.0, 8.0])
    end_time = 10.0
    
    # Create model
    model = NHPPFitter.create_with_linear_intensity(
        event_times=event_times,
        end_time=end_time
    )
    
    # Basic assertions
    assert model.n_events == 3
    assert model.end_time == 10.0
    assert isinstance(model.intensity_function, LinearIntensity)
    
    # Fit model
    result = model.fit(verbose=False)
    
    # Check that fitting succeeded
    assert result.success
    assert model.fitted_params is not None
    assert len(model.fitted_params) == 2  # alpha, beta

def test_log_linear_intensity_model():
    """Test the NHPPFitter with LogLinearIntensity."""
    # Test data
    event_times = np.array([2.0, 5.0, 8.0])
    end_time = 10.0
    covariate_times = np.array([0.0, 5.0, 10.0])
    covariate_values = np.array([1.0, 1.5, 1.0])
    
    # Create model
    model = NHPPFitter.create_with_log_linear_intensity(
        event_times=event_times,
        covariate_times=covariate_times,
        covariate_values=covariate_values,
        end_time=end_time
    )
    
    # Basic assertions
    assert model.n_events == 3
    assert model.end_time == 10.0
    assert isinstance(model.intensity_function, LogLinearIntensity)
    
    # Fit model
    result = model.fit(verbose=False)
    
    # Check that fitting succeeded
    assert result.success
    assert model.fitted_params is not None
    assert len(model.fitted_params) == 2  # beta0, beta1