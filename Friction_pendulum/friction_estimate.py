import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- USER SETTINGS ----------------
CSV_IN = "log.csv"   # your logged data
I = 0.037322         # inertia (kg*m^2)

# Column names in your CSV:
COL_T = "t"                  # seconds
COL_W = "theta_dot"          # rad/s
COL_TAU_CMD = "command_torque"  # Nm
# Optional (if you already logged alpha):
COL_ALPHA = "alpha"          # rad/s^2  (optional)
# ------------------------------------------------


def finite_diff(y, t):
    """Simple numerical derivative dy/dt with gradient (handles nonuniform dt)."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    return np.gradient(y, t)

def sgn(x, eps=1e-4):
    """Sign with deadband around zero to avoid noise blowing up Coulomb fit."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    out[x > eps] = 1.0
    out[x < -eps] = -1.0
    return out

def main():
    df = pd.read_csv(CSV_IN)

    t = df[COL_T].to_numpy(dtype=float)
    w = df[COL_W].to_numpy(dtype=float)               # theta_dot
    tau_cmd = df[COL_TAU_CMD].to_numpy(dtype=float)

    if COL_ALPHA in df.columns:
        alpha = df[COL_ALPHA].to_numpy(dtype=float)
    else:
        alpha = finite_diff(w, t)

    # tau_d = command_torque - I*alpha
    tau_d = tau_cmd - I * alpha

    # Build linear model: tau_d = fc*sgn(w) - fv*w
    S = sgn(w)
    A = np.column_stack([S, -w])   # params = [fc, fv]

    # Ignore near-zero speeds (Coulomb term ambiguous + noisy)
    mask = np.abs(w) > 0.02   # rad/s threshold (tune if needed)
    A_fit = A[mask]
    b_fit = tau_d[mask]

    # Least squares fit
    params, *_ = np.linalg.lstsq(A_fit, b_fit, rcond=None)
    fc_hat, fv_hat = params[0], params[1]

    # Predicted curve
    tau_hat = A @ params

    print(f"Estimated Coulomb friction fc = {fc_hat:.6g} Nm")
    print(f"Estimated viscous friction fv = {fv_hat:.6g} Nm/(rad/s)")

    # Plot: motor velocity vs tau_d (and fitted model)
    plt.figure()
    plt.scatter(w, tau_d, s=8, label="tau_d data")
    # overlay fitted curve as a line by sorting w
    idx = np.argsort(w)
    plt.plot(w[idx], tau_hat[idx], linewidth=2, label="fit: fc*sgn(w) - fv*w")
    plt.xlabel("theta_dot (rad/s)")
    plt.ylabel("tau_d = tau_cmd - I*alpha (Nm)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save result CSV for analysis
    out = pd.DataFrame({
        "t": t,
        "theta_dot": w,
        "alpha": alpha,
        "command_torque": tau_cmd,
        "tau_d": tau_d,
        "tau_fit": tau_hat,
    })
    out.to_csv("friction_fit_output.csv", index=False)
    print("Wrote: friction_fit_output.csv")

if __name__ == "__main__":
    main()
