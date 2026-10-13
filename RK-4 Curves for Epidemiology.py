import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# ============================================================
# 1. PINN ESTIMATED PARAMETERS
# ============================================================
beta1      = 0.44027846
beta2      = 0.19036518
mu         = 0.01066501
sigma      = 0.23901682
p          = 0.43090740
gamma1     = 0.08062782
gamma2     = 0.06086202
delta1     = 0.11071707
delta2     = 0.04088546
eta        = 0.04040855
alpha_est      = 0.80092542

# ============================================================
# 2. INITIAL CONDITIONS (Percentages, N0 = 1)
# ============================================================
S0  = 0.946421
E0  = 0.050000
I10 = 0.001539
I20 = 0.002040
R0  = 0.00000

y0 = np.array([S0, E0, I10, I20, R0], dtype=float)

# ============================================================
# 3. RIGHT-HAND SIDE (Standard Incidence)
# ============================================================
def rhs(y):
    S, E, I1, I2, R = y
    N = S + E + I1 + I2 + R
    
    incidence_1 = (beta1 * S * I1) / N
    incidence_2 = (beta2 * S * I2) / N

    dS  = mu * N - incidence_1 - incidence_2 + eta * R - mu * S
    dE  = incidence_1 + incidence_2 - (sigma + mu) * E
    dI1 = p * sigma * E - (gamma1 + mu + delta1) * I1
    dI2 = (1.0 - p) * sigma * E - (gamma2 + mu + delta2) * I2
    dR  = gamma1 * I1 + gamma2 * I2 - (eta + mu) * R

    return np.array([dS, dE, dI1, dI2, dR])

# ============================================================
# 4. FRACTIONAL RK-2 SOLVER
# ============================================================
def solve_fractional_rk2(alpha, y0, t):
    N_steps = len(t)
    h = t[1] - t[0]
    y = np.zeros((N_steps, len(y0)))
    y[0] = y0

    gamma_alpha = gamma(alpha + 1.0)
    dt_factor = (h**alpha) / gamma_alpha

    for k in range(N_steps - 1):
        # ---- k1 ----
        k1 = rhs(y[k])

        # ---- Midpoint ----
        y_mid = y[k] + 0.5 * dt_factor * k1

        # ---- k2 ----
        k2 = rhs(y_mid)

        # ---- Update ----
        y[k + 1] = y[k] + dt_factor * k2

    return y

# ============================================================
# 5. SIMULATION EXECUTION & VISUALIZATION
# ============================================================
T = 20.0
h = 0.0001
t = np.arange(0, T + h, h)

# Solve fractional system
fractional_solution = solve_fractional_rk2(alpha_est, y0, t)

S  = fractional_solution[:, 0]
E  = fractional_solution[:, 1]
I1 = fractional_solution[:, 2]
I2 = fractional_solution[:, 3]
R  = fractional_solution[:, 4]

# Plot Fractional Model
plt.figure(figsize=(9, 6), facecolor='w')
plt.plot(t, S,  linewidth=2, label='S')
plt.plot(t, E,  linewidth=2, label='E')
plt.plot(t, I1, linewidth=2, label=r'$I_1$ (Symptomatic)')
plt.plot(t, I2, linewidth=2, label=r'$I_2$ (Asymptomatic)')
plt.plot(t, R,  linewidth=2, label='R')

plt.xlabel('Time (years)', fontsize=12, fontweight='bold')
plt.ylabel('Population per 100,000', fontsize=12, fontweight='bold')
plt.title(f'Fractional-Order Meningitis Model (alpha={alpha_est:.3f})', fontsize=14, fontweight='bold')

plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim([0, T])
plt.ylim([0, np.max(fractional_solution) * 1.1])
plt.tight_layout()
plt.show()
