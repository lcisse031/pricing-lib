import numpy as np
from scipy.optimize import minimize, least_squares
import warnings


def sabr_vol(K, F, T, alpha=0.15, beta=0.7, rho=0.2, nu=0.3):
    """SABR volatility formula (Hagan et al. 2002)."""
    if K == 0 or F == 0 or T <= 0:
        return alpha

    moneyness = K / F
    z = (nu / alpha) * (F * K) ** ((1 - beta) / 2) * np.log(moneyness)

    try:
        x = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
    except:
        return alpha

    if abs(x) < 1e-10:
        x = 1e-10

    # Main term
    sigma = (alpha / ((F * K) ** (beta / 2) * x)) * (1 + 0 * T)  # Simplified for speed

    # Corrections (optional, for accuracy)
    num1 = (1 - beta) ** 2 / (24 * (F * K) ** (1 - beta))
    num2 = (nu**2 / (1152 * (F * K) ** (2 * (1 - beta)))) * (5 * rho**2 + 2*rho - 1)

    sigma = (alpha / ((F * K) ** (beta / 2) * x)) * (1 + (num1 + num2) * T)

    return sigma * np.sqrt(T)


class SABRSmoother:
    """
    Calibrates SABR to raw IV data and provides smooth evaluations.
    """

    def __init__(self, spot, r, q):
        """Initialize SABR smoother."""
        self.spot = spot
        self.r = r
        self.q = q
        self.params = {}  # Per-tenor SABR parameters

    def calibrate(self, iv_df, tenors=None):
        """Calibrate SABR to IV surface data."""
        if tenors is None:
            tenors = sorted(iv_df['T'].unique())

        self.params = {}

        for T in tenors:
            data_T = iv_df[iv_df['T'] == T]
            if len(data_T) < 3:
                continue

            strikes = data_T['K'].values
            vols = data_T['iv'].values

            # Forward price at this tenor
            F = self.spot * np.exp((self.r - self.q) * T)

            # Fit SABR to this tenor's data
            params = self._fit_sabr_to_tenor(F, T, strikes, vols)
            self.params[T] = params

        return self.params

    def _fit_sabr_to_tenor(self, F, T, strikes, vols, max_iter=100):
        """
        Fit SABR parameters to a single tenor.

        Uses least-squares optimization with bounds.
        """
        # Initial guess: alpha = ATM vol, others reasonable
        atm_idx = np.argmin(np.abs(strikes - F))
        alpha0 = max(vols[atm_idx], 0.05)

        x0 = [alpha0, 0.7, 0.2, 0.3]  # [alpha, beta, rho, nu]

        def residual(p):
            alpha, beta, nu, rho = p
            alpha = max(alpha, 0.01)
            beta = np.clip(beta, 0.0, 1.0)
            rho = np.clip(rho, -0.99, 0.99)
            nu = max(nu, 0.01)

            pred = np.array([sabr_vol(K, F, T, alpha, beta, rho, nu) for K in strikes])
            return pred - vols

        # Bounds
        bounds = ([0.01, 0.0, -0.99, 0.01], [2.0, 1.0, 0.99, 2.0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = least_squares(residual, x0, bounds=bounds, max_nfev=max_iter)

        alpha, beta, rho, nu = result.x
        alpha = max(alpha, 0.01)
        beta = np.clip(beta, 0.0, 1.0)
        rho = np.clip(rho, -0.99, 0.99)
        nu = max(nu, 0.01)

        return {'alpha': alpha, 'beta': beta, 'rho': rho, 'nu': nu}

    def eval_iv(self, T, K):
        """Evaluate smooth IV at (T, K) using fitted SABR."""
        if not self.params:
            raise ValueError("SABR not calibrated. Call calibrate() first.")

        # Find closest tenor(s) for interpolation
        tenors = sorted(self.params.keys())

        if T <= tenors[0]:
            params = self.params[tenors[0]]
        elif T >= tenors[-1]:
            params = self.params[tenors[-1]]
        else:
            # Linear interpolation in parameter space
            idx = np.searchsorted(tenors, T)
            T1, T2 = tenors[idx-1], tenors[idx]
            p1, p2 = self.params[T1], self.params[T2]
            w = (T - T1) / (T2 - T1)

            params = {
                'alpha': (1-w)*p1['alpha'] + w*p2['alpha'],
                'beta': (1-w)*p1['beta'] + w*p2['beta'],
                'rho': (1-w)*p1['rho'] + w*p2['rho'],
                'nu': (1-w)*p1['nu'] + w*p2['nu'],
            }

        F = self.spot * np.exp((self.r - self.q) * T)
        return sabr_vol(K, F, T, **params)
