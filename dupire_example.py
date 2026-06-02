import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import math

# Paramètres
S = 70.0
r = 0.024888
q = 0.057
T_vals = np.linspace(0.1, 1.0, 10)
K_vals = np.linspace(40, 100, 15)


def sabr_vol(T, K, F, alpha=0.15, beta=0.7, rho=0.2, nu=0.3):
    """Formule SABR pour une surface lisse."""
    if K == 0:
        return alpha

    moneyness = K / F
    z = (nu / alpha) * (F * K) ** ((1 - beta) / 2) * np.log(moneyness)

    x = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))

    num1 = (1 - beta)**2 / (24 * (F * K)**(1 - beta))
    num2 = (nu**2 / (1152 * (F * K)**(2 * (1 - beta)))) * (5 * rho**2 + 2*rho - 1)

    sigma = (alpha / ((F * K)**(beta / 2) * x)) * (1 + (num1 + num2) * T)

    return sigma * np.sqrt(T)


print("=" * 70)
print("SCENARIO 1: Surface LISSE (SABR)")
print("=" * 70)

# Crée la surface SABR
vi_lisse = np.zeros((len(T_vals), len(K_vals)))
for i, T in enumerate(T_vals):
    for j, K in enumerate(K_vals):
        F = S * np.exp((r - q) * T)
        vi_lisse[i, j] = sabr_vol(T, K, F)

# Spline lisse
spline_lisse = RectBivariateSpline(T_vals, K_vals, vi_lisse, kx=3, ky=3)

# Calcule Dupire sur la surface lisse
print(f"\nVI moyennes: {vi_lisse.mean():.4f}")

T_test = 0.25
K_test = S  # ATM

def bs_price(S, K, T, r, q, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    from scipy.stats import norm
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def dupire_calc(spline, T, K, S, r, q, dT=1/252, dK_frac=0.01):
    """Calcule vol locale Dupire."""
    dK = K * dK_frac

    # Récupère vol de la spline
    sigma_TK = float(spline(T, K)[0][0])

    # Dérivées numériques
    sigma_TdT = float(spline(T + dT, K)[0][0])
    sigma_KpDK = float(spline(T, K + dK)[0][0])
    sigma_KmDK = float(spline(T, K - dK)[0][0])

    # Calcule prix et dérivées
    C_TK = bs_price(S, K, T, r, q, sigma_TK)
    C_TdT = bs_price(S, K, T + dT, r, q, sigma_TdT)
    C_KpDK = bs_price(S, K + dK, T, r, q, sigma_KpDK)
    C_KmDK = bs_price(S, K - dK, T, r, q, sigma_KmDK)

    dCdT = (C_TdT - C_TK) / dT
    d2CdK2 = (C_KpDK - 2 * C_TK + C_KmDK) / dK**2
    dCdK = (C_KpDK - C_KmDK) / (2 * dK)

    # Dupire
    num = dCdT + q * C_TK + (r - q) * K * dCdK
    den = 0.5 * K**2 * d2CdK2

    if den <= 1e-10:
        return float(spline(T, K)[0][0])

    local_var = num / den
    if local_var <= 0:
        return float(spline(T, K)[0][0])

    return np.sqrt(local_var)

sigma_vi_lisse = float(spline_lisse(T_test, K_test)[0][0])
sigma_loc_lisse = dupire_calc(spline_lisse, T_test, K_test, S, r, q)

print(f"\nT={T_test:.2f}y, K={K_test:.0f} (ATM):")
print(f"  VI lisse:     {sigma_vi_lisse:.4f} ({sigma_vi_lisse*100:.2f}%)")
print(f"  σ_loc Dupire: {sigma_loc_lisse:.4f} ({sigma_loc_lisse*100:.2f}%)")
print(f"  Ratio σ_loc/VI: {sigma_loc_lisse/sigma_vi_lisse:.2f}")


print("\n" + "=" * 70)
print("SCENARIO 2: Surface BRUTE (VI + bruit)")
print("=" * 70)

np.random.seed(42)
bruit = np.random.normal(0, 0.02, vi_lisse.shape)  # Bruit 2%
vi_brute = vi_lisse + bruit
vi_brute = np.clip(vi_brute, 0.05, 0.50)  # Clipse les extrêmes

spline_brute = RectBivariateSpline(T_vals, K_vals, vi_brute, kx=3, ky=3)

print(f"\nVI moyennes: {vi_brute.mean():.4f}")
print(f"Écart-type du bruit: {bruit.std():.4f}")

sigma_vi_brute = float(spline_brute(T_test, K_test)[0][0])
sigma_loc_brute = dupire_calc(spline_brute, T_test, K_test, S, r, q)

print(f"\nT={T_test:.2f}y, K={K_test:.0f} (ATM):")
print(f"  VI brute:     {sigma_vi_brute:.4f} ({sigma_vi_brute*100:.2f}%)")
print(f"  σ_loc Dupire: {sigma_loc_brute:.4f} ({sigma_loc_brute*100:.2f}%)")
print(f"  Ratio σ_loc/VI: {sigma_loc_brute/sigma_vi_brute:.2f}")


print("\n" + "=" * 70)
print("COMPARAISON")
print("=" * 70)
print(f"\nVol locale lisse:  {sigma_loc_lisse*100:.2f}%")
print(f"Vol locale brute:  {sigma_loc_brute*100:.2f}%")
print(f"Différence:        {abs(sigma_loc_lisse - sigma_loc_brute)*100:.2f}%")
print(f"\n⚠️  DUPIRE SUR SURFACE BRUTE = INSTABLE!")
print(f"✅ DUPIRE SUR SURFACE LISSE = STABLE")


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# VI Lisse
ax = axes[0, 0]
T_plot, K_plot = np.meshgrid(T_vals, K_vals, indexing='ij')
cs = ax.contourf(K_plot, T_plot, vi_lisse, levels=20, cmap='viridis')
ax.set_title('VI Lisse (SABR)', fontsize=12, fontweight='bold')
ax.set_xlabel('Strike')
ax.set_ylabel('Maturité (ans)')
plt.colorbar(cs, ax=ax)

# VI Brute
ax = axes[0, 1]
cs = ax.contourf(K_plot, T_plot, vi_brute, levels=20, cmap='viridis')
ax.set_title('VI Brute (SABR + Bruit)', fontsize=12, fontweight='bold')
ax.set_xlabel('Strike')
ax.set_ylabel('Maturité (ans)')
plt.colorbar(cs, ax=ax)

# Coupes T=0.25
ax = axes[1, 0]
vi_coupe_lisse = [float(spline_lisse(0.25, k)[0][0]) for k in K_vals]
vi_coupe_brute = [float(spline_brute(0.25, k)[0][0]) for k in K_vals]
ax.plot(K_vals, np.array(vi_coupe_lisse)*100, 'b-', linewidth=2, label='Lisse')
ax.plot(K_vals, np.array(vi_coupe_brute)*100, 'r-', linewidth=1, label='Brute')
ax.axvline(S, color='k', linestyle='--', alpha=0.5, label='ATM')
ax.set_title('VI à T=0.25 (3M)', fontsize=12, fontweight='bold')
ax.set_xlabel('Strike')
ax.set_ylabel('VI (%)')
ax.legend()
ax.grid(True, alpha=0.3)

# Stabilité Dupire
ax = axes[1, 1]
sigma_loc_lisse_vals = [dupire_calc(spline_lisse, 0.25, k, S, r, q) for k in K_vals]
sigma_loc_brute_vals = [dupire_calc(spline_brute, 0.25, k, S, r, q) for k in K_vals]
ax.plot(K_vals, np.array(sigma_loc_lisse_vals)*100, 'b-', linewidth=2, marker='o', label='Dupire (lisse)')
ax.plot(K_vals, np.array(sigma_loc_brute_vals)*100, 'r-', linewidth=1, marker='x', label='Dupire (brute)')
ax.axvline(S, color='k', linestyle='--', alpha=0.5, label='ATM')
ax.set_title('Vol locale Dupire à T=0.25', fontsize=12, fontweight='bold')
ax.set_xlabel('Strike')
ax.set_ylabel('σ_loc (%)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dupire_example.png', dpi=100, bbox_inches='tight')
print("\n📊 Graphique sauvegardé: dupire_example.png")
plt.show()
