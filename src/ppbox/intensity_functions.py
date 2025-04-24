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
            covariate_times (np.ndarray): Times at which covariate is measured.
            covariate_values (np.ndarray): Covariate values at measurement times.
            end_time (float): End of observation period.
            grid_size (int, optional): Size of pre-computed grid for faster interpolation.
                Defaults to 1000.
                
        Raises:
            ValueError: If covariate_times and covariate_values have different lengths.
            ValueError: If covariate data is empty.
        """
        if len(covariate_times) != len(covariate_values):
            raise ValueError("covariate_times and covariate_values must have the same length")
        if len(covariate_times) == 0:
            raise ValueError("Covariate data cannot be empty")
        
        # Store sorted covariate data
        idx = np.argsort(covariate_times)
        self.covariate_times = covariate_times[idx]
        self.covariate_values = covariate_values[idx]
        self.end_time = end_time
        
        # Check if observation interval is covered by covariate data
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
        
        Args:
            t (Union[float, np.ndarray]): Time point(s) at which to interpolate.
            
        Returns:
            Union[float, np.ndarray]: Interpolated covariate value(s).
        """
        return np.interp(
            t,
            self.covariate_times,
            self.covariate_values,
            left=self.covariate_values[0],
            right=self.covariate_values[-1]
        )
    
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
        Evaluate the intensity function λ(t) = exp(β₀ + β₁w(t)).
        
        Args:
            t (Union[float, np.ndarray]): Time point(s) at which to evaluate intensity.
            params (np.ndarray): Parameter vector [β₀, β₁].
            
        Returns:
            Union[float, np.ndarray]: Intensity value(s) at time(s) t.
        """
        beta0, beta1 = params
        
        # Vectorized implementation
        if isinstance(t, (list, np.ndarray)):
            w_t = self.get_covariate_at_time(t)
            return np.exp(beta0 + beta1 * w_t)
        
        # Single point calculation
        w_t = self.get_covariate_at_time(t)
        return np.exp(beta0 + beta1 * w_t)
    
    def get_param_count(self) -> int:
        """
        Get the number of parameters (2).
        
        Returns:
            int: 2 (for β₀ and β₁).
        """
        return 2
    
    def get_param_names(self) -> List[str]:
        """
        Get the parameter names.
        
        Returns:
            List[str]: ['beta0', 'beta1'].
        """
        return ['beta0', 'beta1']
    
    def initial_params(self, event_times: np.ndarray, end_time: float) -> np.ndarray:
        """
        Generate initial parameter values.
        
        Args:
            event_times (np.ndarray): Array of observed event times.
            end_time (float): End of observation period.
            
        Returns:
            np.ndarray: [β₀, β₁] where β₀ is log(avg_rate) and β₁ is 0.
        """
        n_events = len(event_times)
        avg_rate = max(0.1, n_events / end_time)
        beta0_guess = np.log(avg_rate)
        return np.array([beta0_guess, 0.0])