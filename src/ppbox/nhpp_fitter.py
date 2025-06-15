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
    
    def calculate_parameter_covariance(self) -> np.ndarray:
        """
        Calculate the variance-covariance matrix of fitted parameters.
        
        Uses the Fisher Information Matrix approach: Var(β̂) = I(β̂)⁻¹
        where I(β̂) is the Fisher Information Matrix.
        
        Returns:
            np.ndarray: Variance-covariance matrix of parameters.
            
        Raises:
            RuntimeError: If model has not been fitted.
            ValueError: If covariance matrix cannot be computed.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        # Get number of parameters
        n_params = len(self.fitted_params)
        
        # Initialize Fisher Information Matrix
        fisher_matrix = np.zeros((n_params, n_params))
        
        # For log-linear models: ∂log λ(t)/∂β₀ = 1, ∂log λ(t)/∂βᵢ = wᵢ(t)
        # Fisher Information: I_ij = ∫₀ᵀ λ(t) * (∂log λ/∂βᵢ) * (∂log λ/∂βⱼ) dt
        
        # Calculate intensity values at grid points
        lambda_values = self._intensity_function(self.time_grid, self.fitted_params)
        
        # Get gradient of log-likelihood at each time point
        if hasattr(self.intensity_function, 'n_covariates'):
            # Multiple covariates case
            n_covariates = self.intensity_function.n_covariates
            
            # Get covariate values at grid points
            if n_covariates == 1:
                # Single covariate - backward compatibility
                w_values = self.intensity_function.get_covariate_at_time(self.time_grid)
                w_matrix = np.column_stack([np.ones(len(self.time_grid)), w_values])
            else:
                # Multiple covariates
                w_values = self.intensity_function.get_covariate_at_time(self.time_grid)
                w_matrix = np.column_stack([np.ones(len(self.time_grid)), w_values])
        else:
            # Linear intensity case: λ(t) = α + βt
            # log λ(t) gradients are more complex, need special handling
            # NOT SUPPORTED YET
            raise NotImplementedError("Covariance calculation for linear intensity models is not implemented yet.")
        
        # Calculate Fisher Information Matrix elements
        for i in range(n_params):
            for j in range(i, n_params):  # Only upper triangle, then mirror
                # I_ij = ∫₀ᵀ λ(t) * (∂log λ/∂βᵢ) * (∂log λ/∂βⱼ) dt
                integrand = lambda_values * w_matrix[:, i] * w_matrix[:, j]
                fisher_matrix[i, j] = np.trapz(integrand, self.time_grid)
                
                # Mirror to lower triangle
                if i != j:
                    fisher_matrix[j, i] = fisher_matrix[i, j]
        
        # Invert Fisher Information Matrix to get covariance matrix
        try:
            # Try standard matrix inversion
            cov_matrix = np.linalg.inv(fisher_matrix)
            cov_method = 'standard_inverse'
        except np.linalg.LinAlgError:
            raise ValueError("Cannot compute parameter covariance matrix. "
                            "Fisher Information Matrix is singular.")
        
        # Store method used for debugging
        if not hasattr(self, '_cov_method'):
            self._cov_method = cov_method
        
        return cov_matrix
    
    def calculate_parameter_confidence_intervals(self, 
                                            confidence_level: float = 0.95) -> pd.DataFrame:
        """
        Calculate confidence intervals for model parameters using asymptotic normality.
        
        Args:
            confidence_level (float): Confidence level (default 0.95 for 95% CI).
            
        Returns:
            pd.DataFrame: DataFrame with columns ['Parameter', 'Estimate', 'Std_Error', 
                        'Lower_CI', 'Upper_CI']
            
        Raises:
            RuntimeError: If model has not been fitted.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        # Get parameter covariance matrix
        param_cov = self.calculate_parameter_covariance()
        
        # Extract standard errors (square root of diagonal elements)
        std_errors = np.sqrt(np.diag(param_cov))
        
        # Calculate confidence intervals: β̂ᵢ ± z * SE(β̂ᵢ)
        z_score = scipy.stats.norm.ppf(1 - (1 - confidence_level) / 2)
        margin = z_score * std_errors
        
        lower_ci = self.fitted_params - margin
        upper_ci = self.fitted_params + margin
        
        # Create results DataFrame
        param_names = self.intensity_function.get_param_names()
        
        results_df = pd.DataFrame({
            'Parameter': param_names,
            'Estimate': self.fitted_params,
            'Std_Error': std_errors,
            'Lower_CI': lower_ci,
            'Upper_CI': upper_ci
        })
        
        return results_df
    
    def calculate_intensity_confidence_intervals(self, 
                                            times: np.ndarray,
                                            confidence_level: float = 0.95,
                                            method: str = 'transformation') -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence intervals for the fitted intensity λ(t).
        
        Args:
            times (np.ndarray): Time points at which to calculate confidence intervals.
            confidence_level (float): Confidence level (default 0.95 for 95% CI).
            method (str): Method to use - 'transformation' (default) or 'delta'.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (lower_bounds, upper_bounds)
            
        Raises:
            RuntimeError: If model has not been fitted.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        # Get parameter covariance matrix
        param_cov = self.calculate_parameter_covariance()
        
        # Get fitted intensity values at specified times
        fitted_intensity = self.predict_intensity(times)
        
        # Calculate variance of linear predictor X(t)ᵀβ at each time point
        linear_predictor_var = np.zeros(len(times))
        
        for i, t in enumerate(times):
            if hasattr(self.intensity_function, 'n_covariates'):
                # Log-linear case
                n_covariates = self.intensity_function.n_covariates
                if n_covariates == 1:
                    w_t = self.intensity_function.get_covariate_at_time(t)
                    x_t = np.array([1.0, w_t])  # [1, w(t)]
                else:
                    w_t = self.intensity_function.get_covariate_at_time(t)
                    x_t = np.concatenate([[1.0], w_t])  # [1, w₁(t), w₂(t), ...]
            else:
                # Linear case: λ(t) = α + βt
                # This needs special handling since it's not log-linear
                x_t = np.array([1.0, t])
            
            # Var(X(t)ᵀβ̂) = X(t)ᵀ Var(β̂) X(t)
            linear_predictor_var[i] = x_t.T @ param_cov @ x_t
        
        # Calculate confidence intervals based on method
        z_score = scipy.stats.norm.ppf(1 - (1 - confidence_level) / 2)
        
        if method == 'transformation':
            # Transformation method (preferred for log-linear models)
            # CI for log(λ(t)) = log(λ̂(t)) ± z * sqrt(Var(X(t)ᵀβ̂))
            # Then transform: CI for λ(t) = exp(CI for log(λ(t)))
            
            if hasattr(self.intensity_function, 'n_covariates'):
                # Log-linear case
                log_intensity = np.log(fitted_intensity)
                log_margin = z_score * np.sqrt(linear_predictor_var)
                
                lower_bounds = np.exp(log_intensity - log_margin)
                upper_bounds = np.exp(log_intensity + log_margin)
            else:
                # Linear case: use delta method since transformation doesn't apply
                method = 'delta'  # Fall back to delta method

        elif method == 'delta':
            # Delta method
            # Var(λ̂(t)) ≈ λ̂(t)² * Var(X(t)ᵀβ̂) for log-linear
            # For linear case, Var(λ̂(t)) = Var(X(t)ᵀβ̂) directly
            
            if hasattr(self.intensity_function, 'n_covariates'):
                # Log-linear: Var(λ̂(t)) ≈ λ̂(t)² * Var(X(t)ᵀβ̂)
                intensity_var = fitted_intensity**2 * linear_predictor_var
            else:
                # Linear: Var(λ̂(t)) = Var(X(t)ᵀβ̂)
                intensity_var = linear_predictor_var
            
            intensity_std = np.sqrt(intensity_var)
            margin = z_score * intensity_std
            
            lower_bounds = np.maximum(0, fitted_intensity - margin)  # Ensure non-negative
            upper_bounds = fitted_intensity + margin
        
        else:
            raise ValueError("method must be 'transformation' or 'delta'")
        
        return lower_bounds, upper_bounds
        
    def calculate_raw_residuals_disjoint(self, 
                                    interval_length: Optional[float] = None,
                                    num_intervals: Optional[int] = None,
                                    residual_type: str = 'raw') -> Dict[str, np.ndarray]:
        """
        Calculate raw residuals using disjoint intervals.
        
        Raw residual = (empirical_rate - fitted_rate) for each interval
        Pearson residual = raw_residual / sqrt(fitted_rate / interval_length)
        
        Args:
            interval_length: Length of each interval.
            num_intervals: Number of intervals.
            residual_type: 'raw' or 'pearson'.
            
        Returns:
            Dict containing: 'residuals', 'time_points', 'empirical_rates', 
            'fitted_rates', 'interval_lengths'
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        # Get empirical rates (reuse existing method)
        time_points, empirical_rates = self.calculate_empirical_rates_disjoint(
            interval_length=interval_length, num_intervals=num_intervals)
        
        # Calculate fitted rates at interval midpoints
        fitted_rates = self.predict_intensity(time_points)
        
        # Calculate raw residuals
        raw_residuals = empirical_rates - fitted_rates
        
        # Calculate interval lengths (most will be the same, but last might differ)
        if interval_length is None:
            interval_length = self.end_time / len(time_points)
        interval_lengths = np.full(len(time_points), interval_length)
        
        # Adjust last interval if needed
        if len(time_points) * interval_length > self.end_time:
            interval_lengths[-1] = self.end_time - (len(time_points) - 1) * interval_length
        
        # Calculate scaled residuals if requested
        if residual_type == 'pearson':
            # Pearson residuals: standardized by sqrt(variance)
            # For Poisson: Var = fitted_rate * interval_length / interval_length = fitted_rate
            residuals = raw_residuals / np.sqrt(fitted_rates / interval_lengths)
        elif residual_type == 'raw':
            residuals = raw_residuals
        else:
            raise ValueError("residual_type must be 'raw' or 'pearson'")
        
        return {
            'residuals': residuals,
            'time_points': time_points,
            'empirical_rates': empirical_rates,
            'fitted_rates': fitted_rates,
            'interval_lengths': interval_lengths,
            'residual_type': residual_type
        }

    def calculate_raw_residuals_overlapping(self, 
                                        interval_length: float,
                                        resolution: int = 100,
                                        residual_type: str = 'raw') -> Dict[str, np.ndarray]:
        """
        Calculate raw residuals using overlapping intervals (sliding window).
        
        Args:
            interval_length: Length of the sliding window.
            resolution: Number of time points to evaluate at.
            residual_type: 'raw' or 'pearson'.
            
        Returns:
            Dict containing: 'residuals', 'time_points', 'empirical_rates', 
            'fitted_rates', 'interval_lengths'
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        # Get empirical rates (reuse existing method)
        time_points, empirical_rates = self.calculate_empirical_rates_overlapping(
            interval_length=interval_length, resolution=resolution)
        
        # Calculate fitted rates at time points
        fitted_rates = self.predict_intensity(time_points)
        
        # Calculate raw residuals
        raw_residuals = empirical_rates - fitted_rates
        
        # All intervals have the same length for overlapping method
        interval_lengths = np.full(len(time_points), interval_length)
        
        # Calculate scaled residuals if requested
        if residual_type == 'pearson':
            residuals = raw_residuals / np.sqrt(fitted_rates / interval_lengths)
        elif residual_type == 'raw':
            residuals = raw_residuals
        else:
            raise ValueError("residual_type must be 'raw' or 'pearson'")
        
        return {
            'residuals': residuals,
            'time_points': time_points,
            'empirical_rates': empirical_rates,
            'fitted_rates': fitted_rates,
            'interval_lengths': interval_lengths,
            'residual_type': residual_type
        }

    def calculate_lurking_variable_residuals(self, 
                                        covariate_values: np.ndarray,
                                        covariate_times: Optional[np.ndarray] = None,
                                        num_intervals: int = 20,
                                        residual_type: str = 'raw',
                                        time_step: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Calculate residuals for lurking variable plots.
        
        Divides covariate range into quantile-based intervals and calculates
        residuals for each interval by aggregating over scattered time points.
        
        Args:
            covariate_values: Values of the covariate to analyze.
            covariate_times: Times corresponding to covariate values. If None,
                            assumes regular grid from 0 to end_time.
            num_intervals: Number of intervals to divide covariate range into.
            residual_type: 'raw' or 'pearson'.
            time_step: Duration that each covariate measurement represents.
            
        Returns:
            Dict containing: 'residuals', 'covariate_midpoints', 'points_per_bin', 
                            'interval_bounds', 'residual_type', 'num_valid_intervals'
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        if covariate_times is None:
            # Create regular grid assuming each measurement represents time_step duration
            covariate_times = np.arange(0, len(covariate_values) * time_step, time_step)
            covariate_times = covariate_times[:len(covariate_values)]  # Ensure same length
        
        if len(covariate_times) != len(covariate_values):
            raise ValueError("covariate_times and covariate_values must have same length")
        
        # Calculate percentiles to define intervals
        percentiles = np.linspace(0, 100, num_intervals + 1)
        interval_bounds = np.percentile(covariate_values, percentiles)
        
        # Initialize results
        residuals = np.full(num_intervals, np.nan)
        covariate_midpoints = np.full(num_intervals, np.nan)
        points_per_bin = np.full(num_intervals, 0)
        
        for i in range(num_intervals):
            # Define interval bounds for this covariate bin
            lower_bound = interval_bounds[i]
            upper_bound = interval_bounds[i + 1]
            
            # Handle edge cases for the last interval
            if i == num_intervals - 1:
                mask = (covariate_values >= lower_bound) & (covariate_values <= upper_bound)
            else:
                mask = (covariate_values >= lower_bound) & (covariate_values < upper_bound)
            
            # Get the number of discrete time points in this bin
            num_points_in_bin = np.sum(mask)
            points_per_bin[i] = num_points_in_bin
            
            if num_points_in_bin == 0:
                continue  # Skip empty intervals
            
            # Get the specific time points and their corresponding covariate values
            times_in_bin = covariate_times[mask]
            
            # The total exposure time is the number of points times the time step
            total_exposure_time = num_points_in_bin * time_step
            
            # Sum the events over all discrete intervals in the bin
            total_events = 0
            for t in times_in_bin:
                # Count events in the interval [t, t + time_step)
                events_in_this_interval = np.sum(
                    (self.event_times >= t) & (self.event_times < t + time_step)
                )
                total_events += events_in_this_interval
            
            # Sum the FITTED intensity over all points in the bin
            # This is the sum of intensity values at each time point, multiplied by time_step
            fitted_intensities_at_points = self.predict_intensity(times_in_bin)
            total_fitted_intensity = np.sum(fitted_intensities_at_points) * time_step
            
            # Calculate the average rates for the entire bin
            empirical_rate = total_events / total_exposure_time
            fitted_rate = total_fitted_intensity / total_exposure_time
            
            # Calculate the residual
            raw_residual = empirical_rate - fitted_rate
            
            if residual_type == 'pearson':
                # Pearson residual: standardize by sqrt(fitted_rate / total_exposure_time)
                if fitted_rate > 0:
                    residuals[i] = raw_residual / np.sqrt(fitted_rate / total_exposure_time)
                else:
                    residuals[i] = np.nan
            else:
                residuals[i] = raw_residual
            
            # Store midpoint of covariate values in this bin
            covariate_midpoints[i] = np.mean(covariate_values[mask])
        
        # Filter out NaN values
        valid_mask = ~np.isnan(residuals)
        
        return {
            'residuals': residuals[valid_mask],
            'covariate_midpoints': covariate_midpoints[valid_mask],
            'points_per_bin': points_per_bin[valid_mask],
            'interval_bounds': interval_bounds,
            'residual_type': residual_type,
            'num_valid_intervals': np.sum(valid_mask)
        }
    
    def simulate_batch(self, 
                    sim_params: np.ndarray,
                    duration: float,
                    n_simulations: int,
                    max_attempts: int = 1000,
                    random_seed: Optional[int] = None,
                    show_progress: bool = False) -> List[np.ndarray]:
        """
        Simulate multiple event time trajectories.
        
        Args:
            sim_params: Parameter vector for simulation.
            duration: Time duration for each simulation.
            n_simulations: Number of trajectories to simulate.
            max_attempts: Maximum root-finding attempts per event.
            random_seed: Seed for reproducibility.
            show_progress: Whether to show progress bar.
            
        Returns:
            List[np.ndarray]: List of simulated event time arrays.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        if show_progress:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(range(n_simulations), desc="Simulating trajectories")
            except ImportError:
                iterator = range(n_simulations)
        else:
            iterator = range(n_simulations)
        
        trajectories = []
        for _ in iterator:
            trajectory = self.simulate(sim_params, duration, max_attempts)
            trajectories.append(trajectory)
        
        return trajectories

    def calculate_simulation_based_residual_qqplot(self,
                                                residual_method: str = 'disjoint',
                                                n_simulations: int = 100,
                                                confidence_level: float = 0.95,
                                                interval_length: Optional[float] = None,
                                                num_intervals: Optional[int] = None,
                                                residual_type: str = 'pearson',
                                                fit_method: str = 'BFGS',
                                                fit_options: Optional[Dict[str, Any]] = None,
                                                random_seed: Optional[int] = None,
                                                show_progress: bool = True,
                                                n_cores: int = 1) -> Dict[str, np.ndarray]:
        """
        Calculate simulation-based QQ plot data for residual diagnostics.
        
        This method:
        1. Simulates multiple datasets from the fitted model
        2. Fits a new model to each simulated dataset
        3. Calculates residuals for each fitted model
        4. Compares observed residuals against the distribution of simulated residuals
        
        Args:
            residual_method: 'disjoint' or 'overlapping' residuals.
            n_simulations: Number of simulations to perform.
            confidence_level: Confidence level for envelopes (e.g., 0.95).
            interval_length: Length of intervals for residual calculation.
            num_intervals: Number of intervals (disjoint method only).
            residual_type: 'raw' or 'pearson' residuals.
            fit_method: Optimization method for refitting models.
            fit_options: Options for the optimizer.
            random_seed: Seed for reproducibility.
            show_progress: Whether to show progress bar.
            n_cores: Number of cores for parallel processing (not implemented yet).
            
        Returns:
            Dict containing:
                'observed_residuals': Original residuals from fitted model
                'expected_quantiles': Expected quantiles under the model
                'lower_envelope': Lower confidence envelope
                'upper_envelope': Upper confidence envelope
                'sorted_residuals': Sorted observed residuals for plotting
        """
        if self.fitted_params is None:
            raise RuntimeError("Model must be fitted before running simulation-based QQ plot.")
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Set default intervals if not provided
        if residual_method == 'disjoint':
            if interval_length is None and num_intervals is None:
                # Default: use approximately 20 intervals or at least 5 based on number of events
                num_intervals = min(20, max(5, self.n_events // 3))
        elif residual_method == 'overlapping':
            if interval_length is None:
                interval_length = self.end_time / 10
        
        # Calculate observed residuals
        if residual_method == 'disjoint':
            obs_residual_data = self.calculate_raw_residuals_disjoint(
                interval_length=interval_length,
                num_intervals=num_intervals,
                residual_type=residual_type
            )
        elif residual_method == 'overlapping':
            obs_residual_data = self.calculate_raw_residuals_overlapping(
                interval_length=interval_length,
                residual_type=residual_type
            )
        else:
            raise ValueError("residual_method must be 'disjoint' or 'overlapping'")
        
        observed_residuals = obs_residual_data['residuals']
        n_residuals = len(observed_residuals)
        
        if n_residuals < 2:
            raise ValueError("Not enough residuals for QQ plot analysis.")
        
        # Sort observed residuals for QQ plot
        sorted_observed = np.sort(observed_residuals)
        
        # Default fit options for simulations
        if fit_options is None:
            fit_options = {'disp': False}
        
        # Store residuals from all simulations
        all_simulated_residuals = []
        
        # Progress tracking
        if show_progress:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(range(n_simulations), desc="Simulating and fitting")
            except ImportError:
                iterator = range(n_simulations)
                warnings.warn("tqdm not available. Install with 'pip install tqdm' for progress bars.")
        else:
            iterator = range(n_simulations)
        
        successful_sims = 0
        
        for _ in iterator:
            try:
                # 1. Simulate new trajectory from fitted model
                sim_events = self.simulate(self.fitted_params, self.end_time)
                
                # Skip if no events generated
                if len(sim_events) == 0:
                    continue
                
                # 2. Create new fitter with simulated data
                sim_fitter = NHPPFitter(
                    event_times=sim_events,
                    intensity_function=copy.deepcopy(self.intensity_function),
                    end_time=self.end_time,
                    grid_size=self.grid_size,
                    discrete_time=self.discrete_time
                )
                
                # 3. Fit model to simulated data
                result = sim_fitter.fit(
                    initial_params=self.fitted_params,  # Use fitted params as initial guess
                    method=fit_method,
                    options=fit_options,
                    verbose=False
                )
                
                if result.success:
                    # 4. Calculate residuals from simulated data
                    if residual_method == 'disjoint':
                        sim_residual_data = sim_fitter.calculate_raw_residuals_disjoint(
                            interval_length=interval_length,
                            num_intervals=num_intervals,
                            residual_type=residual_type
                        )
                    else:
                        sim_residual_data = sim_fitter.calculate_raw_residuals_overlapping(
                            interval_length=interval_length,
                            residual_type=residual_type
                        )
                    
                    sim_residuals = sim_residual_data['residuals']
                    
                    # Ensure same number of residuals (might differ slightly due to events)
                    if len(sim_residuals) >= n_residuals:
                        # Sort and take first n_residuals values
                        sorted_sim_residuals = np.sort(sim_residuals)[:n_residuals]
                    else:
                        # Pad with NaN if fewer residuals
                        sorted_sim_residuals = np.full(n_residuals, np.nan)
                        sorted_sim_residuals[:len(sim_residuals)] = np.sort(sim_residuals)
                    
                    all_simulated_residuals.append(sorted_sim_residuals)
                    successful_sims += 1
                    
            except Exception as e:
                warnings.warn(f"Simulation failed: {e}")
                continue
        
        if successful_sims == 0:
            raise RuntimeError("All simulations failed.")
        
        if show_progress:
            print(f"Successfully completed {successful_sims}/{n_simulations} simulations.")
        
        # Convert to array for easier manipulation
        simulated_residuals_array = np.array(all_simulated_residuals)
        
        # Calculate expected quantiles and envelopes for each order statistic
        expected_quantiles = np.nanmean(simulated_residuals_array, axis=0)
        
        # Calculate confidence envelopes
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        
        lower_envelope = np.nanpercentile(simulated_residuals_array, lower_quantile * 100, axis=0)
        upper_envelope = np.nanpercentile(simulated_residuals_array, upper_quantile * 100, axis=0)
        
        return {
            'observed_residuals': observed_residuals,
            'sorted_residuals': sorted_observed,
            'expected_quantiles': expected_quantiles,
            'lower_envelope': lower_envelope,
            'upper_envelope': upper_envelope,
            'n_simulations': successful_sims,
            'confidence_level': confidence_level,
            'residual_method': residual_method,
            'residual_type': residual_type
        }

    def simulate_inference(self,
                        statistic_func: callable,
                        duration: Optional[float] = None,
                        n_simulations: int = 1000,
                        confidence_level: float = 0.95,
                        func_args: Optional[Dict[str, Any]] = None,
                        random_seed: Optional[int] = None,
                        show_progress: bool = True) -> Dict[str, Any]:
        """
        Perform simulation-based inference for arbitrary statistics.
        
        This function simulates multiple trajectories from the fitted model and
        calculates a user-defined statistic for each trajectory. It returns the
        mean value and confidence envelope for the statistic.
        
        Args:
            statistic_func: Function that takes event times array and returns a scalar.
                        Signature: func(event_times, **func_args) -> scalar
            duration: Duration for simulation. If None, uses self.end_time.
            n_simulations: Number of simulations to perform.
            confidence_level: Confidence level for envelope (e.g., 0.95 for 95%).
            func_args: Additional arguments to pass to statistic_func.
            random_seed: Seed for reproducibility.
            show_progress: Whether to show progress bar.
            
        Returns:
            Dict containing:
                'mean': Mean value of the statistic
                'median': Median value of the statistic
                'lower_bound': Lower confidence bound
                'upper_bound': Upper confidence bound
                'values': All simulated values (for custom analysis)
                'n_valid': Number of valid (non-NaN) results
                
        Examples:
            # Number of events in a time period
            result = model.simulate_inference(
                statistic_func=lambda events: len(events),
                duration=30
            )
            
            # Time of kth event
            def kth_event_time(events, k=5):
                return events[k-1] if len(events) >= k else np.nan
            
            result = model.simulate_inference(
                statistic_func=kth_event_time,
                func_args={'k': 5}
            )
        """
        if self.fitted_params is None:
            raise RuntimeError("Model must be fitted before running inference.")
        
        if duration is None:
            duration = self.end_time
        
        if func_args is None:
            func_args = {}
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Simulate trajectories
        trajectories = self.simulate_batch(
            sim_params=self.fitted_params,
            duration=duration,
            n_simulations=n_simulations,
            show_progress=show_progress
        )
        
        # Calculate statistic for each trajectory
        statistic_values = []
        
        for trajectory in trajectories:
            try:
                value = statistic_func(trajectory, **func_args)
                statistic_values.append(value)
            except Exception as e:
                # Handle cases where statistic cannot be calculated
                statistic_values.append(np.nan)
        
        # Convert to array and remove NaNs for statistics
        values_array = np.array(statistic_values)
        valid_values = values_array[~np.isnan(values_array)]
        n_valid = len(valid_values)
        
        if n_valid == 0:
            warnings.warn("No valid statistics could be calculated from simulations.")
            return {
                'mean': np.nan,
                'median': np.nan,
                'lower_bound': np.nan,
                'upper_bound': np.nan,
                'values': values_array,
                'n_valid': 0
            }
        
        # Calculate summary statistics
        mean_value = np.mean(valid_values)
        median_value = np.median(valid_values)
        
        # Calculate confidence bounds
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(valid_values, lower_percentile)
        upper_bound = np.percentile(valid_values, upper_percentile)
        
        if show_progress and n_valid < n_simulations:
            print(f"Valid simulations: {n_valid}/{n_simulations}")
        
        return {
            'mean': mean_value,
            'median': median_value,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'values': values_array,
            'n_valid': n_valid,
            'confidence_level': confidence_level
        }


    # Convenience methods for common statistics
    def predict_event_count(self, 
                        start_time: float = 0,
                        end_time: Optional[float] = None,
                        n_simulations: int = 1000,
                        confidence_level: float = 0.95,
                        **kwargs) -> Dict[str, Any]:
        """
        Predict the number of events in a time interval.
        
        Args:
            start_time: Start of prediction interval.
            end_time: End of prediction interval (defaults to self.end_time).
            n_simulations: Number of simulations.
            confidence_level: Confidence level for prediction interval.
            **kwargs: Additional arguments passed to simulate_inference.
            
        Returns:
            Dict with mean, median, and prediction interval for event count.
        """
        if end_time is None:
            end_time = self.end_time
        
        def count_in_interval(events):
            return np.sum((events >= start_time) & (events <= end_time))
        
        return self.simulate_inference(
            statistic_func=count_in_interval,
            duration=end_time,
            n_simulations=n_simulations,
            confidence_level=confidence_level,
            **kwargs
        )


    def predict_time_to_kth_event(self,
                                k: int,
                                n_simulations: int = 1000,
                                confidence_level: float = 0.95,
                                **kwargs) -> Dict[str, Any]:
        """
        Predict the time until the kth event occurs.
        
        Args:
            k: Which event to predict (1 for first event, etc.).
            n_simulations: Number of simulations.
            confidence_level: Confidence level for prediction interval.
            **kwargs: Additional arguments passed to simulate_inference.
            
        Returns:
            Dict with mean, median, and prediction interval for time to kth event.
        """
        def time_of_kth_event(events):
            return events[k-1] if len(events) >= k else np.nan
        
        return self.simulate_inference(
            statistic_func=time_of_kth_event,
            n_simulations=n_simulations,
            confidence_level=confidence_level,
            **kwargs
        )
        
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