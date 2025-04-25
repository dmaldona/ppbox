# ppbox: Point Process Modeling Toolkit

A Python library for modeling and simulating Non-Homogeneous Poisson Processes (NHPP).

## Features

* Fit NHPP models using Maximum Likelihood Estimation (MLE).
* Support for different intensity functions:
    * Linear Intensity: $\lambda(t) = \alpha + \beta t$
    * Log-Linear Intensity with covariates: $\lambda(t) = \exp(\beta_0 + \beta_1 w(t))$
* Simulate event times from a fitted model.
* Visualize intensity functions, cumulative intensity, and diagnostic plots.
* Model diagnostics using transformed inter-arrival times (residuals).

## Requirements

* numpy>=1.20.0 [cite: 2]
* scipy>=1.7.0 [cite: 2]
* matplotlib (for plotting)

## Installation

(Provide installation instructions here, e.g., using pip if packaged)

```bash
# Example if packaged:
# pip install ppbox 
```

## Examples

### Example 1: Fitting a Log-Linear NHPP Model

Fit an NHPP model where the intensity depends on a time-varying covariate $w(t)$.

```python
import numpy as np
from ppbox import NHPPFitter
from ppbox.nhpp_plotting import plot_intensity
import matplotlib.pyplot as plt

# --- 1. Define Data and Covariate --- 
np.random.seed(42)
end_time = 100.0

# Example event times (replace with your actual data)
event_times = np.sort(np.random.uniform(0, end_time, 50)) 

# Example covariate: w(t) = 1 + sin(2*pi*t/50)
covariate_times = np.linspace(0, end_time, 101)
covariate_values = 1.0 + np.sin(2 * np.pi * covariate_times / 50.0)

# --- 2. Create and Fit the Model ---
# Use the factory method for LogLinearIntensity
log_linear_model = NHPPFitter.create_with_log_linear_intensity(
    event_times=event_times,
    covariate_times=covariate_times,
    covariate_values=covariate_values,
    end_time=end_time
)

# Fit the model using MLE
result = log_linear_model.fit(verbose=True) 
print(f"Fitted parameters (beta0, beta1): {log_linear_model.fitted_params}")

# --- 3. Plot the Fitted Intensity ---
if log_linear_model.fitted_params is not None:
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_intensity(log_linear_model, ax=ax, show_events=True) #
    ax.set_title("Fitted Log-Linear Intensity Function")
    plt.show()
```

### Example 2: Model Diagnostics

Assess the model fit using diagnostic plots (QQ-plot of residuals against Exp(1) distribution, histogram).

```python
import matplotlib.pyplot as plt
from ppbox.nhpp_plotting import create_diagnostic_plots

# Assuming 'log_linear_model' is the fitted model from Example 1

if log_linear_model.fitted_params is not None:
   # Create a 2x2 grid of diagnostic plots
   fig_diag = create_diagnostic_plots(log_linear_model, figsize=(12, 10)) 
   fig_diag.suptitle("Model Diagnostic Plots", fontsize=16)
   plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
   plt.show()
```

## Code Structure

* `src/ppbox/nhpp_fitter.py`: Core `NHPPFitter` class for modeling and simulation.
* `src/ppbox/intensity_functions.py`: Defines intensity function classes (`LinearIntensity`, `LogLinearIntensity`).
* `src/ppbox/nhpp_plotting.py`: Utility functions for visualization.
* `tests/`: Unit and integration tests. [cite: 1]
* `examples/`: Usage examples.