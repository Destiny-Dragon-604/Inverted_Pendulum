import pyodrivecan
import asyncio
import numpy as np
import smbus
import time
import csv
from datetime import datetime, timedelta
from pathlib import Path
from PendulumEKF import PendulumEKF # Importing your separate file

# ─────────────────────────────────────────────────────────────────────────────
# RLS ESTIMATOR CLASS
# ─────────────────────────────────────────────────────────────────────────────
class RecursiveLeastSquares:
    def __init__(self, n_params: int, lam: float = 1.0, P0: float = 10.0, xi0=None):
        self.lam = float(lam)
        self.n = int(n_params)
        # P0=10.0 allows the filter to actually move; 1e-7 was "locking" it.
        self.P = np.eye(self.n) * float(P0)
        self.xi = (np.zeros(self.n) if xi0 is None 
                   else np.asarray(xi0, dtype=float).flatten())

    def update(self, phi: np.ndarray, epsilon: float):
        phi = np.asarray(phi, dtype=float).reshape(-1, 1)
        denom = self.lam + (phi.T @ self.P @ phi)
        K = (self.P @ phi) / denom
        
        self.xi = self.xi + (K.flatten() * float(epsilon))
        
        # Joseph-stabilized covariance update
        I_KphiT = np.eye(self.n) - (K @ phi.T)
        self.P = (I_KphiT @ self.P @ I_KphiT.T + (K @ K.T) * 1e-6) / self.lam
        return self.xi.copy(), K.flatten()

    @property
    def P_trace(self):
        return float(np.trace(self.P))

# ─────────────────────────────────────────────────────────────────────────────
# HARDWARE & UTILS
# ─────────────────────────────────────────────────────────────────────────────
bus = smbus.SMBus(1)
def read_raw_angle():
    try:
        data = bus.read_i2c_block_data(0x40, 0xFE, 2)
        return data[0] / 255 + data[1] / 64 / 255
    except: return 0.0

class LowPass1:
    def __init__(self, fc_hz: float):
        self.tau = 1.0 / (2.0 * np.pi * fc_hz)
        self.y, self.init = 0.0, False
    def update(self, x, dt):
        if not self.init: self.y, self.init = x, True
        self.y += (dt / (self.tau + dt)) * (x - self.y)
        return self.y

# ─────────────────────────────────────────────────────────────────────────────
# CONTROL LOOP
# ─────────────────────────────────────────────────────────────────────────────
async def controller(odrive):
    await asyncio.sleep(1)
    odrive.set_controller_mode("torque_control")
    
    # Dynamics & EKF
    I = 0.037322
    C, k_rho = 6, 1
    ekf = PendulumEKF(I)
    
    # RLS Tuning
    # lam=1.0 is safer for RLS benchmarks to prevent P-windup
    rls = RecursiveLeastSquares(n_params=3, lam=1.0, P0=10) 
    
    rest_pos = read_raw_angle()
    val, hturns = rest_pos, 0
    t0 = time.monotonic()
    t_prev = asyncio.get_running_loop().time()
    u = 0.0

    print("\n--- RLS Controller Active ---")
    
    try:
        while True:
            t_now = asyncio.get_running_loop().time()
            dt = max(t_now - t_prev, 1e-4)
            t_prev = t_now
            t_s = time.monotonic() - t0

            # 1. Encoder & State
            prev_val = val
            val = read_raw_angle()
            if (val - prev_val) < -0.5: hturns += 1
            elif (val - prev_val) > 0.5: hturns -= 1
            p_meas = (val + hturns - rest_pos) * np.pi
            p_hat, v = ekf.update(p_meas, u, dt)

            # 2. Reference
            w = 0.8 * 2 * np.pi
            A = (5/180.0) * np.pi
            x_d = A * np.sin(w * t_s)
            xd_d = A * w * np.cos(w * t_s)
            xdd_d = -A * (w**2) * np.sin(w * t_s)

            # 3. RLS Update
            phi_vec = np.array([np.sin(p_hat), -np.tanh(v/0.1), -v])
            u_ad_predicted = float(phi_vec @ rls.xi)
            
            # Prediction error fix: (Actual Torque Req) - (Predicted)
            theta_dd_est = ekf.dynamics(ekf.x, u)[1, 0]
            epsilon = (u - I * theta_dd_est) - u_ad_predicted

            # Only update RLS if moving to prevent math exploding (Persistence of Excitation)
            if abs(v) > 0.05:
                xi, _ = rls.update(phi_vec, epsilon)
                ekf.update_parameters(xi[0], xi[1], xi[2])
            else:
                xi = rls.xi

            # 4. Control Law
            rho = (v - xd_d) + C * (p_hat - x_d)
            u_ad = float(phi_vec @ xi)
            u = (-xdd_d * I) + (k_rho * rho) + (I * C * (v - xd_d)) + u_ad
            
            odrive.set_torque(u)
            
            if int(t_s * 10) % 5 == 0: # Print at 2Hz
                print(f"T:{t_s:.1f}s | a:{xi[0]:.3f} | P_tr:{rls.P_trace:.4f}", end='\r')

            await asyncio.sleep(0.001)

    finally:
        odrive.set_torque(0)
        print("\nController stopped safely.")

# ─────────────────────────────────────────────────────────────────────────────
# ODRIVE BOILERPLATE
# ─────────────────────────────────────────────────────────────────────────────
odrive = pyodrivecan.ODriveCAN(0)

async def main():
    odrive.initCanBus()
    odrive.clear_errors()
    await asyncio.gather(odrive.loop(), controller(odrive))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
