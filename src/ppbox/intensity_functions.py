import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, Any, Dict
import warnings

class IntensityFunction(ABC):
    """
    Abstract base class for intensity functions in Non-Homogeneous Poisson Processes.
    
    An intensity function λ(t) defines the instantaneous rate of events at time t,
    which may depend on parameters and covariates.
    
    All intensity functions must implement:
    - evaluate(t, params): Calculate λ(t) for given parameters
    - get_param_count(): Number of parameters needed
    - get_param_names(): Names of the parameters
    - initial_params(events, duration): Suggest initial parameter values
    """
    
    @abstractmethod
    def evaluate(self, t: Union[float, np.ndarray], params: np.ndarray) -> Union[float, np.ndarray]:
        """
        Evaluate the intensity function at time(s) t with given parameters.
        
        Args:
            t (Union[float, np.ndarray]): Time point(s) at which to evaluate the intensity.
            params (np.ndarray): Parameter vector of appropriate length.
            
        Returns:
            Union[float, np.ndarray]: Intensity value(s) at time(s) t.
        """
        pass
    
    @abstractmethod
    def get_param_count(self) -> int:
        """
        Get the number of parameters for this intensity function.
        
        Returns:
            int: Number of parameters required.
        """
        pass
    
    @abstractmethod
    def get_param_names(self) -> List[str]:
        """
        Get the names of parameters for this intensity function.
        
        Returns:
            List[str]: List of parameter names.
        """
        pass
    
    @abstractmethod
    def initial_params(self, event_times: np.ndarray, end_time: float) -> np.ndarray:
        """
        Generate reasonable initial parameter values based on observed events.
        
        Args:
            event_times (np.ndarray): Array of observed event times.
            end_time (float): End of observation period.
            
        Returns:
            np.ndarray: Array of initial parameter values.
        """
        pass


class LinearIntensity(IntensityFunction):
    """
    Implements a linear intensity function: λ(t) = α + βt
    
    The linear intensity function models a NHPP with constant baseline rate α
    and linear time trend β.
    """
    
    def __init__(self) -> None:
        """Initialize the LinearIntensity function."""
        pass
    
    def evaluate(self, t: Union[float, np.ndarray], params: np.ndarray) -> Union[float, np.ndarray]:
        """
        Evaluate the linear intensity function λ(t) = α + βt.
        
        Args:
            t (Union[float, np.ndarray]): Time point(s) at which to evaluate the intensity.
            params (np.ndarray): Parameter vector [α, β].
            
        Returns:
            Union[float, np.ndarray]: Intensity value(s) at time(s) t.
        """
        alpha, beta = params
        return alpha + beta * t
    
    def get_param_count(self) -> int:
        """
        Get the number of parameters (2).
        
        Returns:
            int: 2 (for α and β).
        """
        return 2
    
    def get_param_names(self) -> List[str]:
        """
        Get the parameter names.
        
        Returns:
            List[str]: ['alpha', 'beta']
        """
        return ['alpha', 'beta']
    
    def initial_params(self, event_times: np.ndarray, end_time: float) -> np.ndarray:
        """
        Generate initial parameter values.
        
        Args:
            event_times (np.ndarray): Array of observed event times.
            end_time (float): End of observation period.
            
        Returns:
            np.ndarray: [α₀, β₀] where α₀ is the average rate and β₀ is 0.
        """
        n_events = len(event_times)
        avg_rate = max(0.1, n_events / end_time)
        # Start with constant rate (β=0)
        return np.array([avg_rate, 0.0])


class LogLinearIntensity(IntensityFunction):
    """
    Implements a log-linear intensity function with covariates: λ(t) = exp(β₀ + β₁w(t))
    
    The log-linear intensity function models a NHPP where covariates affect
    the log of the intensity multiplicatively.
    
    Attributes:
        covariate_times (np.ndarray): Times at which covariates are measured.
        covariate_values (np.ndarray): Values of covariates at measurement times.
        end_time (float): End of observation period.
        grid_size (int): Size of pre-computed covariate grid.
        time_grid (np.ndarray): Grid of time points for fast interpolation.
        covariate_grid (np.ndarray): Pre-computed covariate values on time_grid.
    """
    
    def __init__(self, 
                covariate_times: np.ndarray, 
                covariate_values: np.ndarray, 
                end_time: float,
                grid_size: int = 1000) -> None:
        """
        Initialize the LogLinearIntensity function.
        
        Args:
            covariate_times (np.ndarray): Times at which covariates are measured.
            covariate_values (np.ndarray): Covariate values. Can be:
                - 1D array for single covariate
                - 2D array (n_times, n_covariates) for multiple covariates
            end_time (float): End of observation period.
            grid_size (int, optional): Size of pre-computed grid. Defaults to 1000.
        """
        # Convert to 2D array for consistent handling
        covariate_values = np.atleast_2d(covariate_values)
        if covariate_values.shape[0] == 1 and len(covariate_times) > 1:
            # Handle case where single covariate was passed as row vector
            covariate_values = covariate_values.T
        
        if len(covariate_times) != covariate_values.shape[0]:
            raise ValueError("covariate_times and covariate_values must have compatible dimensions")
        if len(covariate_times) == 0:
            raise ValueError("Covariate data cannot be empty")
        
        # Store sorted covariate data
        idx = np.argsort(covariate_times)
        self.covariate_times = covariate_times[idx]
        self.covariate_values = covariate_values[idx, :]
        self.n_covariates = covariate_values.shape[1]
        self.end_time = end_time
        
        # Check coverage
        if self.covariate_times[0] > 0 or self.covariate_times[-1] < self.end_time:
            warnings.warn(f"Covariate range [{self.covariate_times[0]}, {self.covariate_times[-1]}] "
                        f"does not fully cover [0, {self.end_time}]. Extrapolation may occur.")
        
        # Pre-compute covariate grid
        self.grid_size = grid_size
        self.time_grid = np.linspace(0, self.end_time, self.grid_size)
        self.covariate_grid = self._interpolate_covariate(self.time_grid)
    
    def _interpolate_covariate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Interpolate covariate values at time(s) t.
        
        Returns:
            Union[float, np.ndarray]: Interpolated covariate value(s).
                For multiple covariates, returns array of shape (len(t), n_covariates)
        """
        if self.n_covariates == 1:
            # Single covariate - return 1D
            return np.interp(t, self.covariate_times, self.covariate_values[:, 0],
                            left=self.covariate_values[0, 0], right=self.covariate_values[-1, 0])
        else:
            # Multiple covariates - interpolate each one
            if np.isscalar(t):
                result = np.zeros(self.n_covariates)
                for i in range(self.n_covariates):
                    result[i] = np.interp(t, self.covariate_times, self.covariate_values[:, i],
                                        left=self.covariate_values[0, i], right=self.covariate_values[-1, i])
                return result
            else:
                t = np.asarray(t)
                result = np.zeros((len(t), self.n_covariates))
                for i in range(self.n_covariates):
                    result[:, i] = np.interp(t, self.covariate_times, self.covariate_values[:, i],
                                        left=self.covariate_values[0, i], right=self.covariate_values[-1, i])
                return result
    
    def get_covariate_at_time(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get covariate value at time(s) t, using pre-computed grid when possible.
        
        Args:
            t (Union[float, np.ndarray]): Time point(s) at which to get covariate value.
            
        Returns:
            Union[float, np.ndarray]: Covariate value(s) at time(s) t.
        """
        if isinstance(t, (list, np.ndarray)):
            return self._interpolate_covariate(t)
        
        # For single time point, try grid lookup
        if 0 <= t <= self.end_time:
            idx = int(t / self.end_time * (self.grid_size - 1))
            idx = max(0, min(idx, self.grid_size - 1))
            return self.covariate_grid[idx]
        else:
            return self._interpolate_covariate(t)
    
    def evaluate(self, t: Union[float, np.ndarray], params: np.ndarray) -> Union[float, np.ndarray]:
        """
        Evaluate the intensity function λ(t) = exp(β₀ + Σᵢ βᵢwᵢ(t)).
        
        Args:
            params (np.ndarray): Parameter vector [β₀, β₁, β₂, ..., βₖ].
        """
        if len(params) != self.n_covariates + 1:
            raise ValueError(f"Expected {self.n_covariates + 1} parameters, got {len(params)}")
        
        beta0 = params[0]
        betas = params[1:]  # β₁, β₂, ..., βₖ
        
        # Get covariate values
        w_t = self.get_covariate_at_time(t)
        
        if self.n_covariates == 1:
            # Single covariate (backward compatibility)
            return np.exp(beta0 + betas[0] * w_t)
        else:
            # Multiple covariates
            if np.isscalar(t):
                linear_combination = beta0 + np.dot(betas, w_t)
            else:
                linear_combination = beta0 + np.dot(w_t, betas)  # Broadcasting
            return np.exp(linear_combination)
    
    def get_param_count(self) -> int:
        """Return number of parameters: 1 (intercept) + n_covariates."""
        return self.n_covariates + 1

    def get_param_names(self) -> List[str]:
        """Return parameter names."""
        names = ['beta0']
        for i in range(self.n_covariates):
            names.append(f'beta{i+1}')
        return names

    def initial_params(self, event_times: np.ndarray, end_time: float) -> np.ndarray:
        """Generate initial parameter values."""
        n_events = len(event_times)
        avg_rate = max(0.1, n_events / end_time)
        beta0_guess = np.log(avg_rate)
        # Start all covariate effects at 0
        return np.array([beta0_guess] + [0.0] * self.n_covariates)