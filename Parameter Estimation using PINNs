import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_percentage_error

# ============================================================
# 1. REPRODUCIBILITY, DEVICE & DTYPE
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

EPS = 1e-10
POP = 100000.0

# ============================================================
# 2. DATA LOAD & TRAIN/TEST SPLIT
# ============================================================
# Cases per 100,000 population
I_cases_all = np.array([
    357.9, 351.8, 348.0, 342.8, 373.3, 333.1, 341.3, 324.5,
    316.7, 310.4, 304.2, 296.6, 288.7, 283.6, 278.6, 273.7,
    269.2, 262.0, 250.4, 241.5, 231.1, 221.1, 210.5, 210.4,
    191.9, 181.4, 173.4, 165.6, 157.5, 150.5, 145.6
], dtype=float)

# Scale to population fractions
I_frac_all = I_cases_all / POP
N_total = len(I_frac_all)

N_TEST = 3
N_TRAIN = N_total - N_TEST

t_all_np = np.linspace(0.0, 1.0, N_total)
t_train_np = t_all_np[:N_TRAIN]
t_test_np = t_all_np[N_TRAIN:]

I_train_np = I_frac_all[:N_TRAIN]
I_test_np = I_frac_all[N_TRAIN:]

# Tensors
t_train_tensor = torch.tensor(t_train_np.reshape(-1, 1), device=device)
I_train_tensor = torch.tensor(I_train_np.reshape(-1, 1), device=device)

t_test_tensor = torch.tensor(t_test_np.reshape(-1, 1), device=device)
I_test_tensor = torch.tensor(I_test_np.reshape(-1, 1), device=device)

t_all_tensor = torch.tensor(t_all_np.reshape(-1, 1), device=device)

# ============================================================
# 3. INITIAL CONDITIONS
# ============================================================
I0 = I_train_np[0]
p_initial = 0.43

I10 = p_initial * I0
I20 = (1.0 - p_initial) * I0
E0 = 0.05
R0 = 0.0
S0 = 1.0 - E0 - I10 - I20 - R0

# ============================================================
# 4. CAPUTO FRACTIONAL DERIVATIVE (GRÜNWALD–LETNIKOV)
# ============================================================
def frac_binomial_coeffs(alpha, nmax):
    alpha = torch.clamp(alpha, 0.01, 1.99)
    j = torch.arange(0, nmax + 1, device=device)
    return (-1)**j * torch.exp(
        torch.lgamma(alpha + 1)
        - torch.lgamma(j + 1)
        - torch.lgamma(alpha - j + 1)
    )

def caputo_GL(u, alpha):
    N = len(u)
    d = torch.zeros_like(u)
    coeffs = frac_binomial_coeffs(alpha, N - 1)
    for n in range(1, N):
        d[n] = (coeffs[:n+1] @ u[:n+1].flip(0)) / (1.0**alpha + EPS)
    return d

# ============================================================
# 5. PINN ARCHITECTURE
# ============================================================
class MeningitisPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 5)
        )

        # Baseline Epidemiological Parameters & Order Alpha
        self.base = {
            'Lambda': 0.04, 'beta1': 0.44, 'beta2': 0.19,
            'mu': 0.01, 'sigma': 0.24, 'p': 0.43,
            'gamma1': 0.08, 'gamma2': 0.06, 'delta1': 0.11,
            'delta2': 0.04, 'eta': 0.04, 'alpha': 0.80
        }

        # Small perturbations for stable learning
        self.raw = nn.ParameterDict({k: nn.Parameter(torch.zeros(1)) for k in self.base})

    def forward(self, t):
        x = self.net(t)
        # Sigmoid keeps state vectors strictly within valid physical bounds [0, 1]
        S = torch.sigmoid(x[:, 0:1])
        E = torch.sigmoid(x[:, 1:2])
        I1 = torch.sigmoid(x[:, 2:3])
        I2 = torch.sigmoid(x[:, 3:4])
        R = torch.sigmoid(x[:, 4:5])
        return S, E, I1, I2, R

    def params(self):
        dev = 1e-3
        return {k: torch.clamp(self.base[k] + dev * torch.tanh(self.raw[k]), min=1e-8) for k in self.base}

# ============================================================
# 6. SYSTEM RHS EQUATIONS
# ============================================================
def rhs(S, E, I1, I2, R, p):
    res_S = p['Lambda'] - p['beta1']*S*I1 - p['beta2']*S*I2 + p['eta']*R - p['mu']*S
    res_E = p['beta1']*S*I1 + p['beta2']*S*I2 - (p['sigma'] + p['mu'])*E
    res_I1 = p['p']*p['sigma']*E - (p['gamma1'] + p['mu'] + p['delta1'])*I1
    res_I2 = (1.0 - p['p'])*p['sigma']*E - (p['gamma2'] + p['mu'] + p['delta2'])*I2
    res_R = p['gamma1']*I1 + p['gamma2']*I2 - (p['eta'] + p['mu'])*R
    return res_S, res_E, res_I1, res_I2, res_R

# ============================================================
# 7. TRAINING SETUP
# ============================================================
model = MeningitisPINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

w_phys, w_data, w_ic = 1.0, 100.0, 200.0
epochs = 20000

best_loss = 1e18
best_state = None

loss_history = []
data_history = []
physics_history = []
ic_history = []
alpha_history = []

# ============================================================
# 8. TRAINING LOOP
# ============================================================
print("=" * 65)
print("STARTING FRACTIONAL PINN PARAMETER ESTIMATION")
print("=" * 65)

for ep in range(1, epochs + 1):
    optimizer.zero_grad()

    S, E, I1, I2, R = model(t_train_tensor)
    p = model.params()
    alpha = p['alpha']

    # Fractional Caputo Derivatives
    dS = caputo_GL(S.squeeze(), alpha)
    dE = caputo_GL(E.squeeze(), alpha)
    dI1 = caputo_GL(I1.squeeze(), alpha)
    dI2 = caputo_GL(I2.squeeze(), alpha)
    dR = caputo_GL(R.squeeze(), alpha)

    # Differential System Residuals
    rS, rE, rI1, rI2, rR = rhs(S.squeeze(), E.squeeze(), I1.squeeze(), I2.squeeze(), R.squeeze(), p)

    phys_loss = ((dS - rS)[1:].pow(2).mean() +
                 (dE - rE)[1:].pow(2).mean() +
                 (dI1 - rI1)[1:].pow(2).mean() +
                 (dI2 - rI2)[1:].pow(2).mean() +
                 (dR - rR)[1:].pow(2).mean())

    # Data Loss (Observed Total Infected I = I1 + I2)
    I_pred = I1 + I2
    data_loss = (I_pred - I_train_tensor).pow(2).mean()

    # Initial Condition Loss
    ic_loss = ((S[0] - S0)**2 + (E[0] - E0)**2 +
               (I1[0] - I10)**2 + (I2[0] - I20)**2 + (R[0] - R0)**2)

    total_loss = w_phys * phys_loss + w_data * data_loss + w_ic * ic_loss
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # History Logging
    loss_history.append(total_loss.item())
    data_history.append(data_loss.item())
    physics_history.append(phys_loss.item())
    ic_history.append(ic_loss.item())
    alpha_history.append(alpha.item())

    if total_loss.item() < best_loss:
        best_loss = total_loss.item()
        best_state = model.state_dict()

    if ep % 1000 == 0:
        print(f"Epoch {ep:5d} | Total: {total_loss.item():.4e} | Data: {data_loss.item():.4e} | "
              f"Phys: {phys_loss.item():.4e} | IC: {ic_loss.item():.4e} | Alpha: {alpha.item():.6f}")

model.load_state_dict(best_state)

# ============================================================
# 9. ESTIMATED PARAMETERS DISPLAY
# ============================================================
print("\n" + "=" * 65)
print("FINAL ESTIMATED PARAMETERS")
print("=" * 65)
param_df = []
with torch.no_grad():
    p_final = model.params()
    for k, v in p_final.items():
        val = v.item()
        base_val = model.base[k]
        param_df.append({"Parameter": k, "Initial Value": base_val, "Estimated Value": val})
        print(f"{k:10s} = {val:.8f}")

df_params = pd.DataFrame(param_df)

# ============================================================
# 10. EVALUATION & METRICS
# ============================================================
def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual

    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))
    mape = mean_absolute_percentage_error(actual, predicted) * 100.0

    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else np.nan
    max_err = np.max(np.abs(error))

    return rmse, mae, mape, r2, max_err

model.eval()
with torch.no_grad():
    _, _, I1_tr, I2_tr, _ = model(t_train_tensor)
    I_train_pred = ((I1_tr + I2_tr).cpu().numpy().squeeze()) * POP

    _, _, I1_te, I2_te, _ = model(t_test_tensor)
    I_test_pred = ((I1_te + I2_te).cpu().numpy().squeeze()) * POP

I_train_actual = I_cases_all[:N_TRAIN]
I_test_actual = I_cases_all[N_TRAIN:]

tr_rmse, tr_mae, tr_mape, tr_r2, tr_max = calculate_metrics(I_train_actual, I_train_pred)
te_rmse, te_mae, te_mape, te_r2, te_max = calculate_metrics(I_test_actual, I_test_pred)

print("\n" + "=" * 65)
print("TRAINING PERFORMANCE (IN-SAMPLE)")
print("=" * 65)
print(f"RMSE = {tr_rmse:.4f} | MAE = {tr_mae:.4f} | MAPE = {tr_mape:.2f}% | R² = {tr_r2:.4f}")

print("\n" + "=" * 65)
print("UNSEEN HOLD-OUT PERFORMANCE (OUT-OF-SAMPLE)")
print("=" * 65)
print(f"RMSE = {te_rmse:.4f} | MAE = {te_mae:.4f} | MAPE = {te_mape:.2f}% | R² = {te_r2:.4f}")

# Hold-Out Prediction Table
df_pred = pd.DataFrame({
    "Time Index": np.arange(N_TRAIN, N_total),
    "Observed Cases": I_test_actual,
    "PINN Predicted": I_test_pred,
    "Absolute Error": np.abs(I_test_actual - I_test_pred)
})
print("\n" + df_pred.to_string(index=False))

# ============================================================
# 11. PLOTTING RESULTS
# ============================================================
# 1. Training Loss History
plt.figure(figsize=(8, 4))
plt.plot(loss_history, label="Total Loss")
plt.plot(data_history, label="Data Loss")
plt.plot(physics_history, label="Physics Loss")
plt.plot(ic_history, label="IC Loss")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("PINN Training Loss History")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("Meningitis_Loss.eps", format="eps", bbox_inches="tight")
plt.show()

# 2. Convergence of Alpha
plt.figure(figsize=(8, 4))
plt.plot(alpha_history, color='purple', label="Estimated Alpha")
plt.axhline(p_final['alpha'].item(), color='black', linestyle='--', label=f"Final α = {p_final['alpha'].item():.4f}")
plt.xlabel("Epoch")
plt.ylabel(r"Fractional Order ($\alpha$)")
plt.title("Fractional Order Alpha Convergence")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 3. Trajectory Plot: Observed vs PINN Fit & Prediction
with torch.no_grad():
    _, _, I1_all, I2_all, _ = model(t_all_tensor)
    I_all_pred = ((I1_all + I2_all).cpu().numpy().squeeze()) * POP

plt.figure(figsize=(10, 5))
plt.plot(np.arange(N_TRAIN), I_train_actual, 'ko', label="Observed (Train)")
plt.plot(np.arange(N_TRAIN, N_total), I_test_actual, 'ro', label="Observed (Hold-out Test)")
plt.plot(np.arange(N_total), I_all_pred, 'b--', linewidth=2, label="PINN Trajectory")
plt.axvline(N_TRAIN - 0.5, color='gray', linestyle=':', label="Train/Test Split Boundary")
plt.xlabel("Time Step (Years)")
plt.ylabel("Infected Population (per 100,000)")
plt.title("Fractional Meningitis PINN: Fitting & Hold-Out Forecast")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("Meningitis_Trajectory.eps", format="eps", bbox_inches="tight")
plt.show()
