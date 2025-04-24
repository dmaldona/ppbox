import numpy as np
import scipy.optimize
import scipy.integrate
import warnings
import time

class NHPPLogLinearCovariate:
    def __init__(self, event_times, covariate_times, covariate_values, end_time, grid_size=1000):
        """
        Initializes the NHPP model with log-linear covariates.

        Args:
            event_times (np.ndarray): 1D array of observed event times.
            covariate_times (np.ndarray): 1D array of times at which covariate is measured.
            covariate_values (np.ndarray): 1D array of covariate values corresponding to covariate_times.
            end_time (float): The end time (T) of the observation interval [0, T].
            grid_size (int): Size of the pre-computed grid for faster integration.
        """
        # Ensure event_times is an array, even if empty
        event_times = np.asarray(event_times)
        
        # Ensure event times are sorted and within the observation window
        self.event_times = np.sort(event_times[(event_times >= 0) & (event_times <= end_time)])
        self.n_events = len(self.event_times)
        
        # Only warn if this is a fitter (not a simulator with intentionally empty events)
        if self.n_events == 0 and len(event_times) > 0:
            warnings.warn("No events in the specified interval [0, T]. Fitting might be problematic.")

        self.end_time = float(end_time)
        if self.end_time <= 0:
            raise ValueError("end_time (T) must be positive.")

        # --- Covariate Handling ---
        if len(covariate_times) != len(covariate_values):
            raise ValueError("covariate_times and covariate_values must have the same length.")
        if len(covariate_times) == 0:
            raise ValueError("Covariate data cannot be empty.")

        # Store sorted covariate data in numpy arrays
        idx = np.argsort(covariate_times)
        self.covariate_times = covariate_times[idx]
        self.covariate_values = covariate_values[idx]
        
        # Check if observation interval is covered by covariate data
        if self.covariate_times[0] > 0 or self.covariate_times[-1] < self.end_time:
            warnings.warn(f"Covariate time range [{self.covariate_times[0]}, {self.covariate_times[-1]}] "
                        f"does not fully cover observation interval [0, {self.end_time}]. Extrapolation may occur.")
        
        # Pre-compute covariate values on a grid for faster interpolation
        self.grid_size = grid_size
        self.time_grid = np.linspace(0, self.end_time, self.grid_size)
        self.covariate_grid = self._interpolate_covariate(self.time_grid)
        
        self.fitted_params = None
        self.mle_result = None
        self._param_names = ['beta0', 'beta1']

    def _interpolate_covariate(self, t):
        """
        Fast interpolation of covariate values at given time points.
        
        Args:
            t (float or np.ndarray): Time point(s) at which to interpolate.
            
        Returns:
            float or np.ndarray: Interpolated covariate value(s).
        """
        return np.interp(
            t,
            self.covariate_times,
            self.covariate_values,
            left=self.covariate_values[0],    # Extrapolate with boundary values
            right=self.covariate_values[-1]
        )

    def _get_covariate_at_time(self, t):
        """
        Get covariate value at time t using pre-computed grid when possible.
        
        Args:
            t (float): Time point.
            
        Returns:
            float: Covariate value at time t.
        """
        if isinstance(t, (list, np.ndarray)):
            return self._interpolate_covariate(t)
        
        # For single time point, check if grid lookup is possible
        if 0 <= t <= self.end_time:
            # Find nearest grid point (faster than full interpolation)
            idx = int(t / self.end_time * (self.grid_size - 1))
            idx = max(0, min(idx, self.grid_size - 1))  # Ensure valid index
            
            # For better accuracy near event times, do actual interpolation
            # if we're at an event time or if high precision is needed
            if t in self.event_times:
                return self._interpolate_covariate(t)
            return self.covariate_grid[idx]
        else:
            # For times outside observation window, use interpolation with extrapolation
            return self._interpolate_covariate(t)

    def _intensity_function(self, t, params):
        """
        Calculates intensity function λ(t) = exp(β₀ + β₁*w(t)).
        
        Args:
            t (float or np.ndarray): Time point(s).
            params (np.ndarray): Parameters [β₀, β₁].
            
        Returns:
            float or np.ndarray: Intensity value(s).
        """
        beta0, beta1 = params
        
        # Vectorized implementation for arrays
        if isinstance(t, (list, np.ndarray)):
            w_t = self._get_covariate_at_time(t)
            return np.exp(beta0 + beta1 * w_t)
        
        # Single point calculation
        w_t = self._get_covariate_at_time(t)
        return np.exp(beta0 + beta1 * w_t)

    def _negative_log_likelihood(self, params):
        """
        Calculates the negative log-likelihood: -∑log(λ(tᵢ)) + ∫λ(t)dt.
        
        Args:
            params (np.ndarray): Parameters [β₀, β₁].
            
        Returns:
            float: Negative log-likelihood value.
        """
        beta0, beta1 = params

        # 1. Sum of log-intensities at event times
        intensities_at_events = self._intensity_function(self.event_times, params)
        
        # Check for non-positive intensities
        if np.any(intensities_at_events <= 0):
            return np.inf
        
        sum_log_lambda = np.sum(np.log(intensities_at_events))

        # 2. Integral term - use trapezoid rule for efficiency
        try:
            # Pre-compute intensity at grid points (vectorized)
            lambda_grid = self._intensity_function(self.time_grid, params)
            integral_lambda = np.trapz(lambda_grid, self.time_grid)
            
            # Cross-check with scipy.integrate.quad for accuracy if needed
            # (can be commented out in production for speed)
            # func_to_integrate = lambda t: self._intensity_function(t, params)
            # quad_result, _ = scipy.integrate.quad(func_to_integrate, 0, self.end_time)
            # if abs(integral_lambda - quad_result) > 0.1 * quad_result:
            #     warnings.warn(f"Trapezoid integration may be inaccurate: {integral_lambda} vs {quad_result}")
            #     integral_lambda = quad_result
            
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

    def fit(self, initial_params=None, method='BFGS', options=None, verbose=True):
        """
        Fits the model using Maximum Likelihood Estimation (MLE).

        Args:
            initial_params (list or np.ndarray, optional): Initial guess for [β₀, β₁].
            method (str): Optimization method ('BFGS', 'Nelder-Mead', etc.).
            options (dict, optional): Options for the optimizer.
            verbose (bool): Whether to print optimization progress.
            
        Returns:
            scipy.optimize.OptimizeResult: Optimization result object.
        """
        start_time = time.time()
        
        if initial_params is None:
            # Use a reasonable guess: log(avg_rate) for beta0, 0 for beta1
            avg_rate = max(0.1, self.n_events / self.end_time)
            beta0_guess = np.log(avg_rate)
            initial_params = np.array([beta0_guess, 0.0])
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
                print(f"  Fitted parameters ({', '.join(self._param_names)}): {self.fitted_params}")
        else:
            if verbose:
                print(f"Optimization failed: {result.message}")
            self.fitted_params = None
            self.mle_result = result

        return self.mle_result
    
    def _cumulative_intensity(self, t, params):
        """
        Calculates Λ(t) = ∫₀ᵗ λ(u) du using trapezoidal integration.
        
        Args:
            t (float): Upper limit of integration.
            params (np.ndarray): Parameters [β₀, β₁].
            
        Returns:
            float: Cumulative intensity at time t.
        """
        if t <= 0:
            return 0.0
            
        if t >= self.end_time:
            # For t beyond our pre-computed grid, extend the grid
            extended_grid = np.linspace(0, t, self.grid_size)
            lambda_values = self._intensity_function(extended_grid, params)
            return np.trapz(lambda_values, extended_grid)
        
        # For t within our pre-computed grid, we can use part of the grid
        # Find the index in the grid corresponding to time t
        idx = min(self.grid_size - 1, max(0, int(t / self.end_time * (self.grid_size - 1))))
        
        # Create sub-grid from 0 to t
        sub_grid = self.time_grid[:idx+1]
        if sub_grid[-1] < t:
            sub_grid = np.append(sub_grid, t)
            
        # Calculate intensity values at sub-grid points
        lambda_values = self._intensity_function(sub_grid, params)
        
        # Use trapezoidal rule for integration
        return np.trapz(lambda_values, sub_grid)

    def simulate(self, sim_params, duration, max_attempts=1000):
        """
        Simulates event times using the time transformation method.

        Args:
            sim_params (list or np.ndarray): Parameters [β₀, β₁] for simulation.
            duration (float): The time duration over which to simulate.
            max_attempts (int): Maximum number of root-finding attempts.
            
        Returns:
            np.ndarray: Array of simulated event times.
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


if __name__ == "__main__":
    print("=== Testing Optimized NHPP Implementation ===")
    
    # Basic Test: Log-Likelihood Calculation
    print("\n--- Test 1: Log-Likelihood Calculation ---")
    
    # Define simple test data
    test_end_time = 10.0
    test_event_times = np.array([2.0, 5.0, 8.0])
    test_covariate_times = np.array([0.0, test_end_time])
    test_covariate_values = np.array([1.0, 1.0])  # Constant w(t) = 1
    
    # True parameters: beta0 = log(0.5), beta1 = log(2.0) makes lambda(t) = 1
    true_beta0 = np.log(0.5)
    true_beta1 = np.log(2.0)
    true_params = np.array([true_beta0, true_beta1])
    
    # For these parameters, we expect negative log-likelihood = 10.0 - 3*log(1) = 10.0
    expected_neg_ll = 10.0
    
    # Create model
    print("Creating model...")
    start_time = time.time()
    nhpp_model = NHPPLogLinearCovariate(
        event_times=test_event_times,
        covariate_times=test_covariate_times,
        covariate_values=test_covariate_values,
        end_time=test_end_time,
        grid_size=100  # Small grid for quick test
    )
    print(f"Model created in {time.time() - start_time:.4f} seconds")
    
    # Calculate negative log-likelihood
    print("Calculating negative log-likelihood...")
    start_time = time.time()
    calculated_neg_ll = nhpp_model._negative_log_likelihood(true_params)
    print(f"Calculation completed in {time.time() - start_time:.4f} seconds")
    print(f"Expected: {expected_neg_ll}, Calculated: {calculated_neg_ll}")
    print(f"{'SUCCESS' if np.isclose(calculated_neg_ll, expected_neg_ll, rtol=1e-3) else 'FAILURE'}")
    
    # Simulation and Fitting Test with Time-Varying Covariate
    print("\n--- Test 2: Simulation and Fitting ---")
    
    # Define parameters for simulation
    true_beta0 = -2.0  # Base log-rate
    true_beta1 = 0.5   # Effect of covariate
    true_params_sim = np.array([true_beta0, true_beta1])
    sim_duration = 5000.0  # Shorter duration for quick testing
    
    # Define covariate w(t) = 1 + sin(2*pi*t / 50)
    sim_covariate_times = np.linspace(0, sim_duration, int(sim_duration*2) + 1)  # 2 points per unit time
    sim_covariate_values = 1 + np.sin(2 * np.pi * sim_covariate_times / 50.0)
    
    print(f"Simulating with params: beta0={true_beta0}, beta1={true_beta1}, duration={sim_duration}")
    print("Using covariate w(t) = 1 + sin(2*pi*t / 50)")
    
    # Create simulator object
    start_time = time.time()
    simulator = NHPPLogLinearCovariate(
        event_times=np.array([]),  # No events needed for simulator
        covariate_times=sim_covariate_times,
        covariate_values=sim_covariate_values,
        end_time=sim_duration,
        grid_size=1000  # Larger grid for simulation
    )
    print(f"Simulator created in {time.time() - start_time:.4f} seconds")
    
    # Run simulation
    print("Running simulation...")
    start_time = time.time()
    simulated_events = simulator.simulate(sim_params=true_params_sim, duration=sim_duration)
    sim_time = time.time() - start_time
    print(f"Simulation completed in {sim_time:.4f} seconds")
    print(f"Generated {len(simulated_events)} events")
    
    # Create fitter object with simulated data
    print("\nFitting model to simulated data...")
    fitter = NHPPLogLinearCovariate(
        event_times=simulated_events,
        covariate_times=sim_covariate_times,
        covariate_values=sim_covariate_values,
        end_time=sim_duration,
        grid_size=1000
    )
    
    # Run fitting
    start_time = time.time()
    initial_guess = np.array([-1.5, 0.3])  # Slightly off from true values
    method_name = 'BFGS'
    fit_result = fitter.fit(initial_params=initial_guess, method=method_name)
    fit_time = time.time() - start_time
    print(f"Fitting completed in {fit_time:.4f} seconds")
    
    # Compare results
    print("\nResults comparison:")
    print(f"True parameters: {true_params_sim}")
    print(f"Fitted parameters: {fitter.fitted_params}")
    if fitter.fitted_params is not None:
        relative_error = np.abs((fitter.fitted_params - true_params_sim) / true_params_sim)
        print(f"Relative error: {relative_error}")
        success = np.all(relative_error < 0.2)  # Allow up to 20% error
        print(f"{'SUCCESS' if success else 'FAILURE'}: Parameters recovered with acceptable accuracy")
    
    print("\n=== Performance Report ===")
    print(f"Simulation time: {sim_time:.4f} seconds for {len(simulated_events)} events")
    print(f"Fitting time: {fit_time:.4f} seconds using {method_name} optimizer")
    print("Done!")