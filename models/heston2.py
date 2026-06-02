# MC_test

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


def heston_monte_carlo(start_date, S0, r, maturity_months, num_reps,
                       Vo, theta=0.04, kappa=2.0, sigma_v=0.3, rho=-0.7,
                       q=0.0, tr_days=252, antithetic=True, seed=None):
    """Modele de Heston avec volatilite stochastique."""
    V0 = Vo ** 2

    # Grille de dates : start -> end, jours ouvrables
    start_dt = pd.to_datetime(start_date, format="%d/%m/%Y")
    end_dt   = start_dt + relativedelta(months=int(maturity_months))
    dates    = pd.date_range(start=start_dt, end=end_dt, freq='B')
    N        = len(dates) - 1
    dt       = 1.0 / tr_days

    # Seed
    if seed is not None:
        np.random.seed(seed)

    # Chocs correles
    half = num_reps // 2 if antithetic else num_reps
    Z1   = np.random.normal(size=(half, N))
    Z2   = np.random.normal(size=(half, N))
    W1   = Z1
    W2   = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z2

    if antithetic:
        if num_reps % 2 == 1:
            Z1x = np.random.normal(size=(1, N))
            Z2x = np.random.normal(size=(1, N))
            W1  = np.vstack([W1, -W1, Z1x])
            W2  = np.vstack([W2, -W2, rho * Z1x + np.sqrt(1.0 - rho ** 2) * Z2x])
        else:
            W1 = np.vstack([W1, -W1])
            W2 = np.vstack([W2, -W2])

    # Simulation Euler-Maruyama
    S = np.zeros((num_reps, N + 1))
    V = np.zeros((num_reps, N + 1))
    S[:, 0] = S0
    V[:, 0] = V0

    for t in range(N):
        V_pos  = np.maximum(V[:, t], 0.0)
        sqrtV  = np.sqrt(V_pos)
        dW1    = np.sqrt(dt) * W1[:, t]
        dW2    = np.sqrt(dt) * W2[:, t]

        # Variance (CIR)
        V[:, t + 1] = (V[:, t]
                       + kappa * (theta - V[:, t]) * dt
                       + sigma_v * sqrtV * dW2)

        # Prix (drift risk-neutral : r - q)
        S[:, t + 1] = (S[:, t]
                       + (r - q) * S[:, t] * dt
                       + sqrtV * S[:, t] * dW1)

    # DataFrame indexe par les jours ouvrables
    df = pd.DataFrame(S.T, index=dates,
                      columns=[f"Path_{i + 1}" for i in range(num_reps)])
    df.index.name = "Date"
    return df
