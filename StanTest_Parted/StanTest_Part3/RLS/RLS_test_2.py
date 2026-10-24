import pyodrivecan
import asyncio
import numpy as np
import smbus
import time
from PendulumEKF import PendulumEKF

# ═════════════════════════════════════════════════════════════════════════════
# ⚙️ ADJUSTABLE PARAMETERS (CONFIG SPACE)
# ═════════════════════════════════════════════════════════════════════════════
class CONFIG:
    # --- RLS Learning Parameters ---
    RLS_LAMBDA = 0.995      # Forgetting factor (0.98-1.0). 1.0 = infinite memory.
    RLS_P0     = 50.0       # Initial confidence. Increase if 'a' is too slow to move.
    
    # --- Physics & Control Gains ---
    I_INERTIA  = 0.037322   # kg*m^2
    CTRL_C     = 6.0        # Slidings surface slope (rad/s)
    CTRL_K_RHO = 1.5        # Reaching law gain (Robustness)

    # --- Trajectory ---
    TRAJ_FREQ  = 0.8        # Hz
    TRAJ_AMP   = 10.0       # Degrees (Make it large enough for RLS to see gravity change)

# ═════════════════════════════════════════════════════════════════════════════
# RLS ESTIMATOR CLASS
# ═════════════════════════════════════════════════════════════════════════════
class RecursiveLeastSquares:
    def __init__(self, n_params: int, lam: float, P0: float):
        self.lam = float(lam)
        self.n = int(n_params)
        self.P = np.eye(self.n) * float(P0)
        self.xi = np.zeros(self.n)

    def update(self, phi: np.ndarray, epsilon: float):
        phi = np.asarray(phi, dtype=float).reshape(-1, 1)
        denom = self.lam + (phi.T @ self.P @ phi)
        K = (self.P @ phi) / denom
        
        self.xi = self.xi + (K.flatten() * float(epsilon))
        
        # Joseph-stabilized covariance update for numerical robustness
        I_KphiT = np.eye(self.n) - (K @ phi.T)
        self.P = (I_KphiT @ self.P @ I_KphiT.T + (K @ K.T) * 1e-6) / self.lam
        return self.xi.copy()

    @property
    def P_trace(self):
        return float(np.trace(self.P))

# ═════════════════════════════════════════════════════════════════════════════
# HARDWARE INTERFACE
# ═════════════════════════════════════════════════════════════════════════════
bus = smbus.SMBus(1)
def read_raw_angle():
    try:
        data = bus.read_i2c_block_data(0x40, 0xFE, 2)
        return data[0] / 255 + data[1] / 64 / 255
    except Exception: 
        return 0.0

# ═════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═════════════════════════════════════════════════════════════════════════════
async def controller(odrive):
    await asyncio.sleep(1)
    odrive.set_controller_mode("torque_control")
    
    ekf = PendulumEKF(CONFIG.I_INERTIA)
    rls = RecursiveLeastSquares(n_params=1, lam=CONFIG.RLS_LAMBDA, P0=CONFIG.RLS_P0) 
    
    # State tracking
    rest_pos = read_raw_angle()
    val, hturns = rest_pos, 0
    v_prev = 0.0
    u = 0.0

    t0 = time.monotonic()
    t_prev = asyncio.get_running_loop().time()

    print("\n--- RLS Controller Active ---")
    print(f"Tracking: {CONFIG.TRAJ_AMP}deg at {CONFIG.TRAJ_FREQ}Hz")
    
    try:
        while True:
            t_now = asyncio.get_running_loop().time()
            dt = max(t_now - t_prev, 1e-4)
            t_prev = t_now
            t_s = time.monotonic() - t0

            # 1. Encoder Unwrapping & EKF State
            prev_val = val
            val = read_raw_angle()
            if (val - prev_val) < -0.5: hturns += 1
            elif (val - prev_val) > 0.5: hturns -= 1
            p_meas = (val + hturns - rest_pos) * np.pi
            p_hat, v = ekf.update(p_meas, u, dt)

            # 2. Reference Trajectory
            w = CONFIG.TRAJ_FREQ * 2 * np.pi
            A = np.radians(CONFIG.TRAJ_AMP)
            x_d = A * np.sin(w * t_s)
            xd_d = A * w * np.cos(w * t_s)
            xdd_d = -A * (w**2) * np.sin(w * t_s)

            # 3. RLS Parameter Estimation (Gravity Only)
            phi_vec = np.array([np.sin(p_hat)])
            u_ad_predicted = float(phi_vec @ rls.xi)
            
            # Use raw numerical derivative for acceleration to break circularity
            accel_raw = (v - v_prev) / dt
            v_prev = v
            
            # epsilon = Actual Physics Force - Predicted Force
            epsilon = (u - CONFIG.I_INERTIA * accel_raw) - u_ad_predicted

            # Update parameters
            xi = rls.update(phi_vec, epsilon)
            
            # Sync EKF (Friction set to 0 as requested)
            ekf.update_parameters(a=xi[0], fc=0.0, fv=0.0)

            # 4. Control Law (Sliding Mode + Adaptive Compensation)
            e = p_hat - x_d
            ed = v - xd_d
            rho = ed + CONFIG.CTRL_C * e
            
            u_ad = float(phi_vec @ xi) # Gravity compensation
            u_ff = -xdd_d * CONFIG.I_INERTIA # Feedforward
            u_pd = (CONFIG.CTRL_K_RHO * rho) + (CONFIG.I_INERTIA * CONFIG.CTRL_C * ed)
            
            u = u_ff + u_pd + u_ad
            
            odrive.set_torque(u)
            
            # Monitor Output
            if int(t_s * 10) % 2 == 0: 
                print(f"T:{t_s:.1f}s | a_est:{xi[0]:.4f} | Err:{e:.3f} | P_tr:{rls.P_trace:.2f}", end='\r')

            await asyncio.sleep(0.001)

    finally:
        odrive.set_torque(0)
        print("\nSafety Stop: Torque zeroed.")

# ═════════════════════════════════════════════════════════════════════════════
# CAN BUS SETUP
# ═════════════════════════════════════════════════════════════════════════════
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
