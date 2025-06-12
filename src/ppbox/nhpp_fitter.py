import numpy as np
import pandas as pd
import copy
from tqdm.auto import tqdm
import scipy.optimize
import scipy.integrate
import warnings
import time
from typing import List, Optional, Tuple, Union, Any, Dict, Type

from .intensity_functions import IntensityFunction, LinearIntensity, LogLinearIntensity

class NHPPFitter:
    """
    Non-Homogeneous Poisson Process (NHPP) model for event data.
    
    This class provides methods for:
    - Maximum likelihood estimation of intensity function parameters
    - Simulation of event times based on the fitted model
    - Calculation of intensity and cumulative intensity functions
    
    Attributes:
        event_times (np.ndarray): Array of observed event times.
        n_events (int): Number of events in the observation window.
        end_time (float): End time of observation window [0, T].
        intensity_function (IntensityFunction): The intensity function object.
        grid_size (int): Size of the time grid for integration.
        time_grid (np.ndarray): Grid of time points for integration.
        fitted_params (Optional[np.ndarray]): Fitted parameters after MLE.
        mle_result (Optional[scipy.optimize.OptimizeResult]): Full optimization result.
    """
    
    def __init__(self, 
                 event_times: np.ndarray,
                 intensity_function: IntensityFunction,
                 end_time: float,
                 grid_size: int = 1000,
                 discrete_time = False) -> None:
        """
        Initialize the NHPP model.
        
        Args:
            event_times (np.ndarray): Array of observed event times.
            intensity_function (IntensityFunction): Intensity function object.
            end_time (float): End time of observation window [0, T].
            grid_size (int, optional): Size of time grid for integration. Defaults to 1000.
            
        Raises:
            ValueError: If end_time is not positive.
        """
        # Ensure event_times is an array, even if empty
        event_times = np.asarray(event_times)

        # Store likelihood calculation mode
        self.discrete_time = discrete_time

        # Filter and sort event times
        self.event_times = np.sort(event_times[(event_times >= 0) & (event_times <= end_time)])
        self.n_events = len(self.event_times)
        
        # Only warn if this is a fitter (not a simulator with intentionally empty events)
        if self.n_events == 0 and len(event_times) > 0:
            warnings.warn("No events in the specified interval [0, T]. Fitting might be problematic.")
        
        self.end_time = float(end_time)
        if self.end_time <= 0:
            raise ValueError("end_time (T) must be positive.")
        
        # Store the intensity function
        self.intensity_function = intensity_function
        
        # Pre-compute time grid for integration
        self.grid_size = grid_size
        self.time_grid = np.linspace(0, self.end_time, self.grid_size)
        
        # Initialize fitting results
        self.fitted_params: Optional[np.ndarray] = None
        self.mle_result: Optional[scipy.optimize.OptimizeResult] = None
    
    def _intensity_function(self, t: Union[float, np.ndarray], params: np.ndarray) -> Union[float, np.ndarray]:
        """
        Evaluate the intensity function at time(s) t.
        
        Args:
            t (Union[float, np.ndarray]): Time point(s) at which to evaluate the intensity.
            params (np.ndarray): Parameter vector for the intensity function.
            
        Returns:
            Union[float, np.ndarray]: Intensity value(s) at time(s) t.
        """
        return self.intensity_function.evaluate(t, params)
    
    def _negative_log_likelihood(self, params: np.ndarray) -> float:
        """
        Calculate negative log-likelihood for parameter estimation.
        
        The log-likelihood for NHPP has the form:
            LL = ∑log(λ(tᵢ)) - ∫₀ᵀ λ(t)dt
        
        This method returns the negative log-likelihood (-LL) for minimization.
        
        Args:
            params (np.ndarray): Parameter vector to evaluate.
            
        Returns:
            float: Negative log-likelihood value.
            
        Notes:
            Returns np.inf for invalid parameter combinations that lead to numerical
            issues, such as negative or zero intensities at event times.
        """
        # 1. Sum of log-intensities at event times
        intensities_at_events = self._intensity_function(self.event_times, params)
        
        # Check for non-positive intensities
        if np.any(intensities_at_events <= 0):
            return np.inf
        
        sum_log_lambda = np.sum(np.log(intensities_at_events))
        
        # 2. Integral term - use trapezoid rule
        try:
            if self.discrete_time:
                # Assume λ(t) is constant within each unit interval [i-1, i)
                # Sum from i=1 to floor(end_time)
                discrete_times = np.arange(1, int(np.floor(self.end_time)) + 1)
                if len(discrete_times) > 0:
                    lambda_discrete = self._intensity_function(discrete_times, params)
                    integral_lambda = np.sum(lambda_discrete)
                else:
                    integral_lambda = 0.0
            else:
                # Original continuous integration using trapezoid rule
                lambda_grid = self._intensity_function(self.time_grid, params)
                integral_lambda = np.trapz(lambda_grid, self.time_grid)
            
            if np.isnan(integral_lambda) or np.isinf(integral_lambda):
                warnings.warn(f"Integral calculation resulted in {integral_lambda} for params {params}")
                return np.inf
        
        except Exception as e:
            warnings.warn(f"Error during integration for params {params}: {e}")
            return np.inf
        
        # 3. Combine terms
        neg_ll = -sum_log_lambda + integral_lambda
        
        # Check final result validity
        if np.isnan(neg_ll) or np.isinf(neg_ll):
            warnings.warn(f"Negative log-likelihood is invalid ({neg_ll}) for params {params}")
            return np.inf
            
        return neg_ll
    
    def fit(self, 
            initial_params: Optional[np.ndarray] = None, 
            method: str = 'BFGS', 
            options: Optional[Dict[str, Any]] = None, 
            verbose: bool = True) -> scipy.optimize.OptimizeResult:
        """
        Fit the model parameters using Maximum Likelihood Estimation.
        
        Uses numerical optimization to find the parameter values that
        maximize the log-likelihood of the observed event times.
        
        Args:
            initial_params (Optional[np.ndarray], optional): Initial parameter values for optimization.
                If None, values from intensity_function.initial_params() are used. Defaults to None.
            method (str, optional): Optimization method ('BFGS', 'Nelder-Mead', etc.).
                See scipy.optimize.minimize for available methods. Defaults to 'BFGS'.
            options (Optional[Dict[str, Any]], optional): Additional options for the optimizer. 
                Defaults to None.
            verbose (bool, optional): Whether to print progress information. Defaults to True.
            
        Returns:
            scipy.optimize.OptimizeResult: Full optimization result object.
            
        Notes:
            After fitting, the optimal parameters are stored in self.fitted_params
            and the full optimization result in self.mle_result.
        """
        start_time = time.time()
        
        if initial_params is None:
            # Use intensity function's suggestion for initial parameters
            initial_params = self.intensity_function.initial_params(self.event_times, self.end_time)
        else:
            initial_params = np.array(initial_params)
            
        if options is None:
            options = {'disp': verbose}
            
        if verbose:
            print(f"Starting optimization with method '{method}' and initial guess {initial_params}...")
            
        result = scipy.optimize.minimize(
            fun=self._negative_log_likelihood,
            x0=initial_params,
            method=method,
            options=options
        )
        
        if result.success:
            self.fitted_params = result.x
            self.mle_result = result
            if verbose:
                print(f"Optimization successful in {time.time() - start_time:.2f} seconds.")
                print(f"  Log-Likelihood at solution: {-result.fun}")
                param_names = self.intensity_function.get_param_names()
                print(f"  Fitted parameters ({', '.join(param_names)}): {self.fitted_params}")
        else:
            if verbose:
                print(f"Optimization failed: {result.message}")
            self.fitted_params = None
            self.mle_result = result
            
        return self.mle_result
    
    def _cumulative_intensity(self, t: float, params: np.ndarray) -> float:
        """
        Calculate the cumulative intensity function Λ(t) = ∫₀ᵗ λ(u) du.
        
        The cumulative intensity function gives the expected number of events
        in the interval [0, t] for the specified parameters.
        
        Args:
            t (float): Upper limit of integration.
            params (np.ndarray): Parameter vector for the intensity function.
            
        Returns:
            float: Cumulative intensity value Λ(t).
            
        Raises:
            RuntimeWarning: If integration produces invalid results.
            
        Notes:
            Uses efficient trapezoidal integration with pre-computed grids
            where possible to improve performance.
        """
        if t <= 0:
            return 0.0
            
        if t >= self.end_time:
            # For t beyond pre-computed grid, extend the grid
            extended_grid = np.linspace(0, t, self.grid_size)
            lambda_values = self._intensity_function(extended_grid, params)
            return np.trapz(lambda_values, extended_grid)
            
        # For t within pre-computed grid, use part of the grid
        idx = min(self.grid_size - 1, max(0, int(t / self.end_time * (self.grid_size - 1))))
        
        # Create sub-grid from 0 to t
        sub_grid = self.time_grid[:idx+1]
        if sub_grid[-1] < t:
            sub_grid = np.append(sub_grid, t)
            
        # Calculate intensity values at sub-grid points
        lambda_values = self._intensity_function(sub_grid, params)
        
        # Use trapezoidal rule for integration
        result = np.trapz(lambda_values, sub_grid)
        
        # Check validity
        if np.isnan(result) or np.isinf(result) or result < 0:
            warnings.warn(f"Cumulative intensity calculation resulted in {result} at t={t}")
            
        return result
    
    def simulate(self, 
                sim_params: np.ndarray, 
                duration: float, 
                max_attempts: int = 1000) -> np.ndarray:
        """
        Simulate event times using the time transformation method.
        
        The time transformation algorithm:
        1. Calculate Λ(duration) = expected number of events
        2. Generate N ~ Poisson(Λ(duration)) as the number of events
        3. Generate N uniform random variables τᵢ ~ U[0, Λ(duration)]
        4. Invert τᵢ = Λ(tᵢ) to find each event time tᵢ
        
        Args:
            sim_params (np.ndarray): Parameter vector for simulation.
            duration (float): The time duration over which to simulate.
            max_attempts (int, optional): Maximum number of root-finding attempts. 
                Defaults to 1000.
            
        Returns:
            np.ndarray: Array of simulated event times, sorted in ascending order.
            
        Raises:
            RuntimeError: If cumulative intensity calculation fails.
            
        Notes:
            The inversion step uses a numerical root-finding method since
            Λ⁻¹(τ) typically has no closed-form solution.
        """
        sim_params = np.asarray(sim_params)
        if duration <= 0:
            return np.array([])
            
        # 1. Calculate Λ(duration)
        lambda_max = self._cumulative_intensity(duration, sim_params)
        if np.isnan(lambda_max) or lambda_max < 0:
            raise RuntimeError(f"Failed to calculate cumulative intensity at end time {duration}")
        if lambda_max == 0:
            return np.array([])
            
        # 2. Simulate number of events from Poisson(Λ(duration))
        num_events = np.random.poisson(lam=lambda_max)
        if num_events == 0:
            return np.array([])
            
        # 3. Simulate event times uniformly from [0, Λ(duration)]
        tau_events = np.sort(np.random.uniform(0, lambda_max, num_events))
        
        # 4. Invert Λ(t) to find times
        simulated_event_times = np.zeros(num_events)
        
        # Pre-compute a finer grid for better initial guesses
        fine_grid_size = min(10000, max(1000, num_events * 10))
        fine_time_grid = np.linspace(0, duration, fine_grid_size)
        fine_cum_intensity = np.zeros(fine_grid_size)
        
        # Calculate cumulative intensity at each grid point
        for i, t in enumerate(fine_time_grid):
            fine_cum_intensity[i] = self._cumulative_intensity(t, sim_params)
            
        # For each tau_i, find corresponding t_i
        for i, tau_i in enumerate(tau_events):
            if tau_i <= 1e-10:
                simulated_event_times[i] = 0.0
                continue
                
            # Use grid to find a good initial bracket for root finding
            idx = np.searchsorted(fine_cum_intensity, tau_i)
            
            # Ensure valid bracket within grid bounds
            if idx == 0:
                lower_bound = 0
                upper_bound = fine_time_grid[1]
            elif idx >= len(fine_time_grid):
                lower_bound = fine_time_grid[-2]
                upper_bound = duration
            else:
                lower_bound = fine_time_grid[idx-1]
                upper_bound = fine_time_grid[idx]
                
            # Define objective function for root finding
            objective_func = lambda t: self._cumulative_intensity(t, sim_params) - tau_i
            
            try:
                # Use brentq with improved initial bracket
                t_i = scipy.optimize.brentq(
                    f=objective_func,
                    a=lower_bound,
                    b=upper_bound,
                    xtol=1e-6,
                    rtol=1e-6,
                    maxiter=100
                )
                simulated_event_times[i] = t_i
            except Exception as e:
                warnings.warn(f"Root finding failed for tau_{i}={tau_i}. Error: {e}")
                simulated_event_times[i] = np.nan
                
        # Filter out NaNs from failed root finding
        valid_times = simulated_event_times[~np.isnan(simulated_event_times)]
        if len(valid_times) < num_events:
            warnings.warn(f"Simulation produced {len(valid_times)} valid events out of {num_events} attempted.")
            
        return np.sort(valid_times)
    
    def predict_intensity(self, times: np.ndarray) -> np.ndarray:
        """
        Predict intensity values at specified times using fitted model.
        
        Args:
            times (np.ndarray): Times at which to predict intensity values.
            
        Returns:
            np.ndarray: Predicted intensity values at specified times.
            
        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        # Calculate intensity values using fitted parameters
        intensity_values = self._intensity_function(times, self.fitted_params)
        
        return intensity_values
    
    def calculate_transformed_interarrivals(self) -> np.ndarray:
        """
        Calculate transformed interarrival times for model diagnostics.
        
        For a correctly specified NHPP model, the transformed interarrival times
        should follow an exponential distribution with unit mean (Exp(1)).
        
        Returns:
            np.ndarray: Array of transformed interarrival times.
            
        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        if self.n_events == 0:
            return np.array([])
        
        # Calculate cumulative intensity values at event times
        tau_values = np.zeros(self.n_events)
        for i, t_i in enumerate(self.event_times):
            tau_values[i] = self._cumulative_intensity(t_i, self.fitted_params)
        
        # Check for NaNs or other issues
        if np.any(np.isnan(tau_values)):
            raise ValueError("Cumulative intensity calculation produced NaN values.")
        
        # Calculate interarrival times by taking differences
        # Prepend 0 to represent start time (t=0)
        deltas = np.diff(np.insert(tau_values, 0, 0.0))
        
        return deltas

    def calculate_normalized_transformed_times(self) -> np.ndarray:
        """
        Calculate normalized transformed event times for diagnostics.

        Computes U_i = Lambda(S_i) / Lambda(T), where S_i are event times
        and T is the end_time. Under the correct model, these should be
        distributed as order statistics from a Uniform(0, 1) distribution.

        Returns:
            np.ndarray: Array of normalized transformed event times U_i.

        Raises:
            RuntimeError: If the model has not been fitted yet.
            ValueError: If cumulative intensity calculations fail.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet.")

        if self.n_events == 0:
            return np.array([])

        # Calculate cumulative intensity at end time Lambda(T)
        lambda_T = self._cumulative_intensity(self.end_time, self.fitted_params)
        if lambda_T <= 0 or np.isnan(lambda_T) or np.isinf(lambda_T):
            warnings.warn(f"Total cumulative intensity Lambda(T) = {lambda_T} is invalid. Cannot normalize.")
            # Return empty or handle error appropriately, maybe raise ValueError
            return np.array([]) 

        # Calculate cumulative intensity values at each event time Lambda(S_i)
        tau_values = np.zeros(self.n_events)
        for i, t_i in enumerate(self.event_times):
            tau_values[i] = self._cumulative_intensity(t_i, self.fitted_params)

        # Check for NaNs or other issues in individual Lambda(S_i)
        if np.any(np.isnan(tau_values)) or np.any(tau_values < 0):
            # Handle potential issues from _cumulative_intensity
            valid_mask = ~np.isnan(tau_values) & (tau_values >= 0)
            if not np.all(valid_mask):
                warnings.warn("Some cumulative intensity values at event times were invalid.")
                # Decide how to proceed: filter or raise error
                # For now, filter, but raising might be better
                tau_values = tau_values[valid_mask]
                if len(tau_values) == 0:
                    return np.array([])


        # Normalize: U_i = Lambda(S_i) / Lambda(T)
        normalized_times = tau_values / lambda_T

        # Ensure values are within [0, 1] due to potential numerical inaccuracies
        normalized_times = np.clip(normalized_times, 0.0, 1.0)

        return np.sort(normalized_times) # Ensure sorted
    
    def bootstrap_parameter_uncertainty(self,
                                        num_replicates: int = 100,
                                        fit_method: str = 'BFGS',
                                        fit_options: Optional[Dict[str, Any]] = None,
                                        show_progress: bool = True,
                                        random_seed: Optional[int] = None) -> pd.DataFrame:
        """
        Estimate parameter uncertainty using parametric bootstrap.

        This method performs the following steps:
        1. Simulates multiple event datasets from the currently fitted model.
        2. Refits the model to each simulated dataset.
        3. Collects the parameter estimates from these refits.
        4. Returns the collection of bootstrap parameter estimates.

        Args:
            num_replicates (int, optional): Number of bootstrap datasets to generate.
                Defaults to 100.
            fit_method (str, optional): Optimization method used for fitting bootstrap models.
                Defaults to 'BFGS'.
            fit_options (Optional[Dict[str, Any]], optional): Options passed to the optimizer
                for bootstrap fits. Defaults to None (typically {'disp': False}).
            show_progress (bool, optional): Whether to display a progress bar.
                Defaults to True (requires tqdm).
            random_seed (Optional[int], optional): Seed for reproducibility of simulations.
                Defaults to None.

        Returns:
            pd.DataFrame: A pandas DataFrame where each row corresponds to a successful
                bootstrap replicate and columns correspond to the fitted parameters.
                Returns an empty DataFrame if the original model is not fitted or
                if no bootstrap replicates succeed.

        Raises:
            RuntimeError: If the model has not been fitted yet.
            ImportError: If show_progress is True and tqdm is not installed.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model must be fitted before running bootstrap.")

        if show_progress:
            try:
                from tqdm.auto import tqdm
            except ImportError:
                raise ImportError("tqdm must be installed to show progress bar. "
                                  "Install with 'pip install tqdm' or set show_progress=False.")

        if random_seed is not None:
            np.random.seed(random_seed)

        original_params = self.fitted_params
        param_names = self.intensity_function.get_param_names()
        bootstrap_estimates = []

        # Default fit options for bootstrap (usually non-verbose)
        if fit_options is None:
            fit_options = {'disp': False}
        else:
            # Ensure verbosity is off unless explicitly requested
            fit_options.setdefault('disp', False)

        # Determine how to create new instances for refitting
        # This needs access to the original configuration
        # We assume the intensity_function object carries necessary config
        # (like covariate data in LogLinearIntensity)
        intensity_func_config = self.intensity_function # Assumes state (covariates) is here

        iterator = range(num_replicates)
        if show_progress:
            iterator = tqdm(iterator, desc="Bootstrap Replicates")

        successful_fits = 0
        for _ in iterator:
            # 1. Simulate new data from the fitted model
            simulated_event_times = self.simulate(sim_params=original_params,
                                                  duration=self.end_time)

            # 2. Create a new fitter instance for the simulated data
            # We need to ensure the new instance has the same intensity function type
            # and configuration (like covariates, end_time).
            # A simple way is to create a new instance using the stored intensity function object.
            # This assumes the intensity function object is self-contained or can be deep-copied.
            try:
                # Attempt to deepcopy the intensity function to avoid side effects if it's mutable
                current_intensity_func = copy.deepcopy(intensity_func_config)
            except TypeError:
                 # If deepcopy fails (e.g., for complex objects), use the original reference cautiously
                 # or implement a specific clone/factory method later.
                 current_intensity_func = intensity_func_config
                 warnings.warn("Could not deepcopy intensity function; using original reference.")

            bootstrap_fitter = NHPPFitter(
                event_times=simulated_event_times,
                intensity_function=current_intensity_func,
                end_time=self.end_time,
                grid_size=self.grid_size # Use same grid size
            )

            # 3. Fit the model to the simulated data
            # Use original fitted parameters as initial guess for speed/stability
            try:
                result = bootstrap_fitter.fit(
                    initial_params=original_params,
                    method=fit_method,
                    options=fit_options,
                    verbose=False # Ensure verbose is off for bootstrap fits
                )

                # 4. Store results if successful
                if result.success and bootstrap_fitter.fitted_params is not None:
                    bootstrap_estimates.append(bootstrap_fitter.fitted_params)
                    successful_fits += 1
                # Optional: Add logging here for failed fits if desired

            except Exception as e:
                # Catch potential errors during fitting on simulated data
                warnings.warn(f"Bootstrap replicate fit failed with error: {e}")
                continue # Skip to the next replicate

        if show_progress:
             print(f"Bootstrap completed. {successful_fits}/{num_replicates} fits succeeded.")

        if not bootstrap_estimates:
            warnings.warn("No bootstrap replicates succeeded.")
            return pd.DataFrame(columns=param_names)

        # Convert results to DataFrame
        bootstrap_df = pd.DataFrame(bootstrap_estimates, columns=param_names)

        return bootstrap_df

    def get_parameter_summary(self, bootstrap_results: Optional[pd.DataFrame] = None,
                              num_replicates: int = 100,
                              alpha: float = 0.05,
                              **bootstrap_kwargs) -> Optional[pd.DataFrame]:
        """
        Provides a summary table of parameters including MLE estimates,
        bootstrap standard errors, and confidence intervals.

        Args:
            bootstrap_results (Optional[pd.DataFrame], optional): Pre-computed bootstrap
                results from bootstrap_parameter_uncertainty(). If None, bootstrap
                will be run first. Defaults to None.
            num_replicates (int, optional): Number of replicates if bootstrap needs to be run.
                Defaults to 100.
            alpha (float, optional): Significance level for confidence intervals (e.g., 0.05 for 95% CI).
                 Defaults to 0.05.
            **bootstrap_kwargs: Additional keyword arguments passed to
                 bootstrap_parameter_uncertainty if it needs to be run.

        Returns:
            Optional[pd.DataFrame]: A DataFrame summarizing parameter estimates and uncertainty,
                 or None if the model is not fitted or bootstrap fails.
        """
        if self.fitted_params is None:
            print("Model is not fitted.")
            return None

        if bootstrap_results is None:
            print(f"Running bootstrap with {num_replicates} replicates...")
            bootstrap_results = self.bootstrap_parameter_uncertainty(
                num_replicates=num_replicates,
                **bootstrap_kwargs
            )

        if bootstrap_results is None or bootstrap_results.empty:
            print("Bootstrap failed or yielded no successful replicates.")
            return None

        # Calculate summary statistics
        mle_series = pd.Series(self.fitted_params, index=bootstrap_results.columns, name='MLE_Estimate')
        mean_series = bootstrap_results.mean().rename('Bootstrap_Mean')
        std_series = bootstrap_results.std().rename('Bootstrap_StdErr')

        # Calculate percentile confidence intervals
        lower_quantile = alpha / 2.0
        upper_quantile = 1.0 - lower_quantile
        lower_ci = bootstrap_results.quantile(lower_quantile).rename(f'CI_{lower_quantile*100:.1f}%')
        upper_ci = bootstrap_results.quantile(upper_quantile).rename(f'CI_{upper_quantile*100:.1f}%')

        # Combine into a summary DataFrame
        summary_df = pd.concat([mle_series, mean_series, std_series, lower_ci, upper_ci], axis=1)

        return summary_df
    
    def calculate_empirical_rates_disjoint(self, 
                                        interval_length: Optional[float] = None,
                                        num_intervals: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate empirical rates using disjoint intervals.
        
        Args:
            interval_length: Length of each interval. If None, calculated from num_intervals.
            num_intervals: Number of intervals. If None, calculated from interval_length.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (time_points, empirical_rates)
        """
        if interval_length is None and num_intervals is None:
            raise ValueError("Either interval_length or num_intervals must be specified")
        if interval_length is not None and num_intervals is not None:
            raise ValueError("Only one of interval_length or num_intervals can be specified")
        
        if interval_length is None:
            interval_length = self.end_time / num_intervals
        else:
            num_intervals = int(np.ceil(self.end_time / interval_length))
        
        # Create interval boundaries
        intervals = np.linspace(0, self.end_time, num_intervals + 1)
        time_points = (intervals[:-1] + intervals[1:]) / 2  # Midpoints
        empirical_rates = np.zeros(num_intervals)
        
        # Calculate empirical rate for each interval
        for i in range(num_intervals):
            start_time = intervals[i]
            end_time = intervals[i + 1]
            
            # Count events in this interval
            events_in_interval = np.sum((self.event_times >= start_time) & 
                                    (self.event_times < end_time))
            
            # Calculate rate
            empirical_rates[i] = events_in_interval / (end_time - start_time)
        
        return time_points, empirical_rates

    def calculate_empirical_rates_overlapping(self, 
                                            interval_length: float,
                                            resolution: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate empirical rates using overlapping intervals (sliding window).
        
        Args:
            interval_length: Length of the sliding window.
            resolution: Number of time points to evaluate at.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (time_points, empirical_rates)
        """
        # Create time points for evaluation
        time_points = np.linspace(interval_length/2, 
                                self.end_time - interval_length/2, 
                                resolution)
        empirical_rates = np.zeros(len(time_points))
        
        # Calculate empirical rate at each time point
        for i, t in enumerate(time_points):
            start_time = max(0, t - interval_length/2)
            end_time = min(self.end_time, t + interval_length/2)
            
            # Count events in this interval
            events_in_interval = np.sum((self.event_times >= start_time) & 
                                    (self.event_times < end_time))
            
            # Calculate rate
            actual_length = end_time - start_time
            empirical_rates[i] = events_in_interval / actual_length
        
        return time_points, empirical_rates
    
    @classmethod
    def create_with_log_linear_intensity(cls, 
                                        event_times: np.ndarray,
                                        covariate_times: np.ndarray,
                                        covariate_values: np.ndarray,
                                        end_time: float,
                                        grid_size: int = 1000,
                                        discrete_time: bool = False) -> 'NHPPFitter':
        """
        Factory method to create an NHPPFitter with LogLinearIntensity.
        
        Args:
            event_times (np.ndarray): Array of observed event times.
            covariate_times (np.ndarray): Times at which covariate is measured.
            covariate_values (np.ndarray): Covariate values. Can be:
                - 1D array for single covariate
                - 2D array (n_times, n_covariates) for multiple covariates
            end_time (float): End time of observation window.
            grid_size (int, optional): Size of grid for integration. Defaults to 1000.
            
        Returns:
            NHPPFitter: NHPPFitter instance with LogLinearIntensity.
        """
        intensity = LogLinearIntensity(
            covariate_times=covariate_times,
            covariate_values=covariate_values,
            end_time=end_time,
            grid_size=grid_size
        )
        return cls(
            event_times=event_times,
            intensity_function=intensity,
            end_time=end_time,
            grid_size=grid_size,
            discrete_time=discrete_time
        )
    
    @classmethod
    def create_with_linear_intensity(cls,
                                    event_times: np.ndarray,
                                    end_time: float,
                                    grid_size: int = 1000,
                                    discrete_time: bool = False) -> 'NHPPFitter':
        """
        Factory method to create an NHPPFitter with LinearIntensity.
        
        Args:
            event_times (np.ndarray): Array of observed event times.
            end_time (float): End time of observation window.
            grid_size (int, optional): Size of grid for integration. Defaults to 1000.
            
        Returns:
            NHPPFitter: NHPPFitter instance with LinearIntensity.
        """
        intensity = LinearIntensity()
        return cls(
            event_times=event_times,
            intensity_function=intensity,
            end_time=end_time,
            grid_size=grid_size,
            discrete_time=discrete_time  # ADD THIS LINE
        )