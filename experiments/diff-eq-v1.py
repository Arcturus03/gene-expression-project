"""
gene_expression_time_series.py
---------------------------------
Simulates a basic gene expression model using a differential equation.
Then, it trains a simple Linear Regression model to predict future values.
Author: Hrithik
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------
# 1. Define the differential equation: dy/dt = 4t + 3
# ---------------------------------------------------
def dydt(t):
    """Derivative of y with respect to t."""
    return 4 * t + 3


# ---------------------------------------------------
# 2. Numerical integration to generate y(t)
# ---------------------------------------------------
def simulate_gene_expression(t_start=0, t_end=10, dt=0.1, y0=2):
    """
    Simulates gene expression over time using simple forward integration.
    Args:
        t_start: starting time (default 0)
        t_end: ending time (default 10)
        dt: step size (smaller = more accurate)
        y0: initial gene expression level
    Returns:
        t_vals (numpy array): time values
        y_vals (numpy array): simulated gene expression values
    """
    t_vals = np.arange(t_start, t_end, dt)  # numpy array arranging from t_start to t_end with step dt
    y_vals = np.zeros_like(t_vals)  # initialize y values array
    y_vals[0] = y0

    for i in range(1, len(t_vals)):
        y_vals[i] = y_vals[i-1] + dydt(t_vals[i-1]) * dt  # Euler's Method

    return t_vals, y_vals


# ---------------------------------------------------
# 3. Generate the time series data
# ---------------------------------------------------
t, y = simulate_gene_expression()
print("Simulated gene expression time series generated!")


# ---------------------------------------------------
# 4. Plot the simulated time series
# ---------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(t, y, label="Gene Expression (y)", color="purple")
plt.xlabel("Time")
plt.ylabel("Gene Expression Level")
plt.title("Simulated Gene Expression Over Time")
plt.legend()
plt.grid(True)
plt.show()


# ---------------------------------------------------
# 5. Prepare data for Linear Regression
# ---------------------------------------------------
# The goal: predict y_next from y_current
X = y[:-1].reshape(-1, 1)  # current expression
y_next = y[1:]             # next expression

# Train simple linear regression
model = LinearRegression()
model.fit(X, y_next)

# Predict using the trained model
y_pred = model.predict(X)

# ---------------------------------------------------
# 6. Compare real vs predicted
# ---------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(t[1:], y_next, label="Actual Next Value", color="blue")
plt.plot(t[1:], y_pred, label="Predicted Next Value", color="red", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Gene Expression")
plt.title("Linear Regression Prediction of Gene Expression Dynamics")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------------------------------
# 7. Print model coefficients
# ---------------------------------------------------
print("Linear Regression Results:")
print("Equation: y_next =", round(model.coef_[0], 3), "* y_current +", round(model.intercept_, 3))
