"""
Tests for mathematical correctness of the NHPP implementation.

This test suite verifies the mathematical correctness of the NHPP implementation
by checking against known analytical results for specific cases:
1. Homogeneous Poisson Process (HPP)
2. Linear Intensity
3. Log-Linear Intensity with constant covariate
"""

import numpy as np
import pytest
from scipy import stats
import math

from ppbox import NHPPFitter
from ppbox.intensity_functions import LinearIntensity, LogLinearIntensity


class TestHomogeneousPoissonProcess:
    """Tests for Homogeneous Poisson Process (constant intensity)."""
    
    @pytest.fixture
    def constant_rate(self):
        """Fixture providing a constant rate for HPP."""
        return 0.5  # Events per unit time
    
    @pytest.fixture
    def hpp_log_linear_model(self, constant_rate):
        """Create a LogLinear model that behaves like an HPP."""
        # For HPP using LogLinearIntensity, we use:
        # λ(t) = exp(β₀ + β₁*w(t)) where β₁ = 0 and w(t) = constant
        # This gives λ(t) = exp(β₀) = constant_rate
        
        # The true parameters
        beta0 = np.log(constant_rate)
        beta1 = 0.0
        
        # For covariate, we use a constant value (any value works since β₁ = 0)
        covariate_times = np.array([0.0, 100.0])
        covariate_values = np.array([1.0, 1.0])  # constant w(t) = 1
        
        # Empty event times for now (we'll simulate later)
        event_times = np.array([])
        end_time = 100.0
        
        # Create model
        model = NHPPFitter.create_with_log_linear_intensity(
            event_times=event_times,
            covariate_times=covariate_times,
            covariate_values=covariate_values,
            end_time=end_time
        )
        
        return model, beta0, beta1, end_time
    
    def test_hpp_intensity_calculation(self, hpp_log_linear_model, constant_rate):
        """Test that intensity function returns constant rate for HPP."""
        model, beta0, beta1, _ = hpp_log_linear_model
        
        # Test intensity at various times
        test_times = np.array([0.0, 10.0, 50.0, 99.9])
        params = np.array([beta0, beta1])
        
        for t in test_times:
            intensity = model._intensity_function(t, params)
            assert np.isclose(intensity, constant_rate, rtol=1e-6)
    
    def test_hpp_cumulative_intensity(self, hpp_log_linear_model, constant_rate):
        """Test cumulative intensity function for HPP."""
        model, beta0, beta1, _ = hpp_log_linear_model
        params = np.array([beta0, beta1])
        
        # For HPP, Λ(t) = λ * t
        # Test at various intervals
        test_intervals = [(0, 10), (5, 15), (20, 50), (0, 100)]
        
        for t_start, t_end in test_intervals:
            expected = constant_rate * (t_end - t_start)
            # Calculate using the model's method (subtracting to get the interval)
            actual = model._cumulative_intensity(t_end, params) - model._cumulative_intensity(t_start, params)
            assert np.isclose(actual, expected, rtol=1e-3)
    
    def test_hpp_negative_log_likelihood(self, hpp_log_linear_model, constant_rate):
        """Test negative log-likelihood calculation for HPP."""
        model, beta0, beta1, end_time = hpp_log_linear_model
        
        # Create some test event times (e.g., from a real HPP)
        np.random.seed(42)  # For reproducibility
        n_events = 50
        # For HPP, events are uniformly distributed
        event_times = np.sort(np.random.uniform(0, end_time, n_events))
        
        # Create a new model with these event times
        covariate_times = np.array([0.0, end_time])
        covariate_values = np.array([1.0, 1.0])
        
        model = NHPPFitter.create_with_log_linear_intensity(
            event_times=event_times,
            covariate_times=covariate_times,
            covariate_values=covariate_values,
            end_time=end_time
        )
        
        params = np.array([beta0, beta1])
        
        # For HPP, the negative log-likelihood is:
        # -LL = λT - n*log(λ)
        expected = constant_rate * end_time - n_events * np.log(constant_rate)
        actual = model._negative_log_likelihood(params)
        
        assert np.isclose(actual, expected, rtol=1e-3)
    
    def test_hpp_fitting(self, hpp_log_linear_model, constant_rate):
        """Test fitting of HPP data with LogLinearIntensity."""
        model, beta0, beta1, end_time = hpp_log_linear_model
        
        # Simulate data from HPP
        true_params = np.array([beta0, beta1])
        np.random.seed(42)
        event_times = model.simulate(true_params, end_time)
        n_events = len(event_times)
        
        # Create a new model with these event times
        covariate_times = np.array([0.0, end_time])
        covariate_values = np.array([1.0, 1.0])
        
        model = NHPPFitter.create_with_log_linear_intensity(
            event_times=event_times,
            covariate_times=covariate_times,
            covariate_values=covariate_values,
            end_time=end_time
        )
        
        # Fit the model
        model.fit(verbose=False)
        
        # For HPP, we expect λ ≈ n/T
        expected_rate = n_events / end_time
        fitted_rate = np.exp(model.fitted_params[0])  # exp(β₀)
        
        assert np.isclose(fitted_rate, expected_rate, rtol=0.1)
        assert np.isclose(model.fitted_params[1], 0.0, atol=0.2)  # β₁ should be close to 0
    
    def test_hpp_diagnostics(self, hpp_log_linear_model):
        """Test diagnostics for HPP using transformed inter-arrival times."""
        model, beta0, beta1, end_time = hpp_log_linear_model
        
        # Simulate data
        true_params = np.array([beta0, beta1])
        np.random.seed(42)
        event_times = model.simulate(true_params, end_time)
        
        # Create and fit model
        covariate_times = np.array([0.0, end_time])
        covariate_values = np.array([1.0, 1.0])
        
        model = NHPPFitter.create_with_log_linear_intensity(
            event_times=event_times,
            covariate_times=covariate_times,
            covariate_values=covariate_values,
            end_time=end_time
        )
        
        model.fit(verbose=False)
        
        # Calculate transformed inter-arrival times
        # Δτᵢ = Λ(tᵢ) - Λ(tᵢ₋₁)
        cum_intensities = np.zeros(len(event_times))
        for i, t in enumerate(event_times):
            cum_intensities[i] = model._cumulative_intensity(t, model.fitted_params)
        
        # Prepend 0 for the first event (time 0 has cumulative intensity 0)
        cum_intensities_with_0 = np.insert(cum_intensities, 0, 0.0)
        
        # Calculate differences
        delta_taus = np.diff(cum_intensities_with_0)
        
        # If model is correct, these should follow Exponential(1)
        # Test mean is close to 1
        assert np.isclose(np.mean(delta_taus), 1.0, rtol=0.2)
        
        # Test distribution using Kolmogorov-Smirnov test
        # H0: data comes from Exponential(1)
        ks_statistic, p_value = stats.kstest(delta_taus, stats.expon(scale=1.0).cdf)
        
        # We don't want to reject H0, so p-value should be > 0.05
        assert p_value > 0.05, f"KS test failed: statistic={ks_statistic}, p={p_value}"


class TestLinearIntensity:
    """Tests for linear intensity function: λ(t) = α + βt."""
    
    @pytest.fixture
    def linear_params(self):
        """Fixture providing parameters for linear intensity."""
        return {
            "alpha": 0.1,  # constant term
            "beta": 0.05,  # linear coefficient
            "end_time": 100.0
        }
    
    @pytest.fixture
    def linear_model(self, linear_params):
        """Create a LinearIntensity model."""
        # Empty event times for now
        event_times = np.array([])
        
        # Create model
        model = NHPPFitter.create_with_linear_intensity(
            event_times=event_times,
            end_time=linear_params["end_time"]
        )
        
        return model, linear_params
    
    def test_linear_intensity_calculation(self, linear_model):
        """Test intensity calculation for linear intensity."""
        model, params = linear_model
        alpha, beta = params["alpha"], params["beta"]
        
        # Test intensity at various times
        test_times = np.array([0.0, 10.0, 50.0, 99.9])
        model_params = np.array([alpha, beta])
        
        for t in test_times:
            expected = alpha + beta * t
            intensity = model._intensity_function(t, model_params)
            assert np.isclose(intensity, expected, rtol=1e-6)
    
    def test_linear_cumulative_intensity(self, linear_model):
        """Test cumulative intensity for linear intensity."""
        model, params = linear_model
        alpha, beta = params["alpha"], params["beta"]
        model_params = np.array([alpha, beta])
        
        # For λ(t) = α + βt, the cumulative intensity is:
        # Λ(t) = αt + βt²/2
        
        # Test at various times
        test_times = np.array([0.0, 10.0, 50.0, 99.9])
        
        for t in test_times:
            expected = alpha * t + beta * t**2 / 2
            actual = model._cumulative_intensity(t, model_params)
            assert np.isclose(actual, expected, rtol=1e-3)
    
    def test_linear_negative_log_likelihood(self, linear_model):
        """Test negative log-likelihood calculation for linear intensity."""
        model, params = linear_model
        alpha, beta = params["alpha"], params["beta"]
        end_time = params["end_time"]
        
        # Create some test event times
        np.random.seed(42)
        # For testing, we'll use a few events
        event_times = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        n_events = len(event_times)
        
        # Create a new model with these event times
        model = NHPPFitter.create_with_linear_intensity(
            event_times=event_times,
            end_time=end_time
        )
        
        model_params = np.array([alpha, beta])
        
        # For linear intensity, the negative log-likelihood is:
        # -LL = ∫₀ᵀ (α + βt) dt - ∑ᵢ log(α + βtᵢ)
        # = αT + βT²/2 - ∑ᵢ log(α + βtᵢ)
        
        integral_term = alpha * end_time + beta * end_time**2 / 2
        sum_log_term = np.sum(np.log(alpha + beta * event_times))
        expected = integral_term - sum_log_term
        
        actual = model._negative_log_likelihood(model_params)
        
        assert np.isclose(actual, expected, rtol=1e-3)
    
    def test_linear_diagnostics(self, linear_model):
        """Test diagnostics for linear intensity."""
        model, params = linear_model
        alpha, beta = params["alpha"], params["beta"]
        end_time = params["end_time"]
        model_params = np.array([alpha, beta])
        
        # Generate data as in the fitting test
        np.random.seed(42)
        lambda_T = alpha * end_time + beta * end_time**2 / 2
        n_events = np.random.poisson(lambda_T)
        u = np.sort(np.random.uniform(0, lambda_T, n_events))
        
        event_times = np.zeros(n_events)
        for i, u_i in enumerate(u):
            a = beta / 2
            b = alpha
            c = -u_i
            
            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                discriminant = 0  # Numerical error, treat as 0
            t_i = (-b + np.sqrt(discriminant)) / (2 * a)
            
            t_i = min(max(0, t_i), end_time)
            event_times[i] = t_i
        
        # Create and fit model
        model = NHPPFitter.create_with_linear_intensity(
            event_times=event_times,
            end_time=end_time
        )
        
        # Instead of fitting, use the true parameters for diagnostics
        # (to isolate diagnostics testing from fitting errors)
        model.fitted_params = model_params
        
        # Calculate transformed inter-arrival times
        cum_intensities = np.zeros(len(event_times))
        for i, t in enumerate(event_times):
            cum_intensities[i] = model._cumulative_intensity(t, model_params)
        
        cum_intensities_with_0 = np.insert(cum_intensities, 0, 0.0)
        delta_taus = np.diff(cum_intensities_with_0)
        
        # If model is correct, these should follow Exponential(1)
        assert np.isclose(np.mean(delta_taus), 1.0, rtol=0.2)
        
        # Test distribution using Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.kstest(delta_taus, stats.expon(scale=1.0).cdf)
        assert p_value > 0.05, f"KS test failed: statistic={ks_statistic}, p={p_value}"


class TestLogLinearConstantCovariate:
    """Tests for log-linear intensity with constant covariate."""
    
    @pytest.fixture
    def loglinear_params(self):
        """Fixture providing parameters for log-linear intensity."""
        return {
            "beta0": -1.0,
            "beta1": 0.5,
            "covariate_value": 2.0,  # constant w(t) = 2
            "end_time": 100.0
        }
    
    @pytest.fixture
    def loglinear_model(self, loglinear_params):
        """Create a LogLinearIntensity model with constant covariate."""
        beta0 = loglinear_params["beta0"]
        beta1 = loglinear_params["beta1"]
        w_const = loglinear_params["covariate_value"]
        end_time = loglinear_params["end_time"]
        
        # With constant covariate, λ(t) = exp(β₀ + β₁*w_const)
        # This is equivalent to an HPP with rate λ = exp(β₀ + β₁*w_const)
        
        # Empty event times for now
        event_times = np.array([])
        
        # Create a constant covariate
        covariate_times = np.array([0.0, end_time])
        covariate_values = np.array([w_const, w_const])
        
        # Create model
        model = NHPPFitter.create_with_log_linear_intensity(
            event_times=event_times,
            covariate_times=covariate_times,
            covariate_values=covariate_values,
            end_time=end_time
        )
        
        return model, loglinear_params
    
    def test_loglinear_constant_intensity(self, loglinear_model):
        """Test that intensity is constant with constant covariate."""
        model, params = loglinear_model
        beta0 = params["beta0"]
        beta1 = params["beta1"]
        w_const = params["covariate_value"]
        
        # Expected constant intensity
        expected_intensity = np.exp(beta0 + beta1 * w_const)
        
        # Test at various times
        test_times = np.array([0.0, 10.0, 50.0, 99.9])
        model_params = np.array([beta0, beta1])
        
        for t in test_times:
            intensity = model._intensity_function(t, model_params)
            assert np.isclose(intensity, expected_intensity, rtol=1e-6)
    
    def test_loglinear_constant_cumulative_intensity(self, loglinear_model):
        """Test cumulative intensity with constant covariate."""
        model, params = loglinear_model
        beta0 = params["beta0"]
        beta1 = params["beta1"]
        w_const = params["covariate_value"]
        
        # For constant intensity, Λ(t) = λt
        expected_rate = np.exp(beta0 + beta1 * w_const)
        
        # Test cumulative intensity at various intervals
        test_intervals = [(0, 10), (5, 25), (0, 100)]
        model_params = np.array([beta0, beta1])
        
        for t_start, t_end in test_intervals:
            expected = expected_rate * (t_end - t_start)
            actual = model._cumulative_intensity(t_end, model_params) - model._cumulative_intensity(t_start, model_params)
            assert np.isclose(actual, expected, rtol=1e-3)
    
    def test_loglinear_constant_negative_log_likelihood(self, loglinear_model):
        """Test negative log-likelihood with constant covariate."""
        model, params = loglinear_model
        beta0 = params["beta0"]
        beta1 = params["beta1"]
        w_const = params["covariate_value"]
        end_time = params["end_time"]
        
        # Expected constant intensity
        expected_rate = np.exp(beta0 + beta1 * w_const)
        
        # Create some test event times
        np.random.seed(42)
        n_events = 50
        event_times = np.sort(np.random.uniform(0, end_time, n_events))
        
        # Create a new model with these event times
        covariate_times = np.array([0.0, end_time])
        covariate_values = np.array([w_const, w_const])
        
        model = NHPPFitter.create_with_log_linear_intensity(
            event_times=event_times,
            covariate_times=covariate_times,
            covariate_values=covariate_values,
            end_time=end_time
        )
        
        model_params = np.array([beta0, beta1])
        
        # For constant intensity, -LL = λT - n*log(λ)
        expected = expected_rate * end_time - n_events * np.log(expected_rate)
        actual = model._negative_log_likelihood(model_params)
        
        assert np.isclose(actual, expected, rtol=1e-3)
    
    def test_loglinear_constant_fitting(self, loglinear_model):
        """Test fitting with constant covariate."""
        model, params = loglinear_model
        beta0 = params["beta0"]
        beta1 = params["beta1"]
        w_const = params["covariate_value"]
        end_time = params["end_time"]
        
        # Expected constant intensity
        expected_rate = np.exp(beta0 + beta1 * w_const)
        
        # Simulate data
        true_params = np.array([beta0, beta1])
        np.random.seed(42)
        event_times = model.simulate(true_params, end_time)
        n_events = len(event_times)
        
        # Create a new model with these event times
        covariate_times = np.array([0.0, end_time])
        covariate_values = np.array([w_const, w_const])
        
        model = NHPPFitter.create_with_log_linear_intensity(
            event_times=event_times,
            covariate_times=covariate_times,
            covariate_values=covariate_values,
            end_time=end_time
        )
        
        # Fit the model
        model.fit(verbose=False)
        
        # For constant intensity, we expect the fitted parameters to give
        # exp(β₀ + β₁*w_const) ≈ n/T
        observed_rate = n_events / end_time
        fitted_rate = np.exp(model.fitted_params[0] + model.fitted_params[1] * w_const)
        
        assert np.isclose(fitted_rate, observed_rate, rtol=0.1)
        
        # The individual values of β₀ and β₁ can vary as long as 
        # β₀ + β₁*w_const remains close to log(n/T)
        # So we test their sum directly
        expected_sum = np.log(observed_rate)
        actual_sum = model.fitted_params[0] + model.fitted_params[1] * w_const
        
        assert np.isclose(actual_sum, expected_sum, rtol=0.1)