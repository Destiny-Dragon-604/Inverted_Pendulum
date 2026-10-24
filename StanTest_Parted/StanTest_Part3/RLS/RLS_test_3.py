"""
RLS benchmark — now using augmented EKF for joint estimation
============================================================
The separate RLS estimator is gone. 'a' is estimated directly
inside the EKF by augmenting the state to [θ, ω, a].

Access via:  ekf.a   (the gravity parameter estimate)
"""

import pyodrivecan
import asyncio
import numpy as np
import smbus
import time
from PendulumEKF2 import PendulumEKF   # ← augmented version


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════
class CONFIG:
    # --- EKF tuning ---
    A_INIT      = -1     # initial guess for gravity parameter
    Q_TORQUE    = 1.0     # process noise: disturbance torque variance
    Q_A         = 0.0000000000005      # process noise: how fast 'a' is allowed to drift
    #                         increase Q_A if estimate is too slow to converge
    #                         decrease Q_A if estimate is too noisy

    # --- Physics & Control ---
    I_INERTIA   = 0.037322
    CTRL_C      = 6.0
    CTRL_K_RHO  = 1.5

    # --- Trajectory ---
    TRAJ_FREQ   = 0.8       # Hz
    TRAJ_AMP    = 20      # degrees


# ═════════════════════════════════════════════════════════════════════════════
# HARDWARE
# ═════════════════════════════════════════════════════════════════════════════
bus = smbus.SMBus(1)

def read_raw_angle():
    try:
        data = bus.read_i2c_block_data(0x40, 0xFE, 2)
        return data[0] / 255 + data[1] / 64 / 255
    except Exception:
        return 0.0


# ═════════════════════════════════════════════════════════════════════════════
# CONTROL LOOP
# ═════════════════════════════════════════════════════════════════════════════
async def controller(odrive):
    await asyncio.sleep(1)
    odrive.set_controller_mode("torque_control")

    ekf = PendulumEKF(
        I         = CONFIG.I_INERTIA,
        a_init    = CONFIG.A_INIT,
        q_torque  = CONFIG.Q_TORQUE,
        q_a       = CONFIG.Q_A,
    )

    rest_pos        = read_raw_angle()
    val, hturns     = rest_pos, 0
    u               = 0.0
    t0              = time.monotonic()
    t_prev          = asyncio.get_running_loop().time()

    print("\n--- Augmented EKF Controller Active ---")
    print(f"Tracking: {CONFIG.TRAJ_AMP}deg @ {CONFIG.TRAJ_FREQ}Hz")
    print(f"Initial a guess: {CONFIG.A_INIT}  (EKF will correct this)")

    try:
        while True:
            t_now = asyncio.get_running_loop().time()
            dt    = max(t_now - t_prev, 1e-4)
            t_prev = t_now
            t_s   = time.monotonic() - t0

            # ── 1. Encoder unwrap ────────────────────────────────────────────
            prev_val = val
            val      = read_raw_angle()
            if (val - prev_val) < -0.5:
                hturns += 1
            elif (val - prev_val) > 0.5:
                hturns -= 1
            p_meas = (val + hturns - rest_pos) * np.pi

            # ── 2. EKF update — gives θ̂, ω̂, AND â simultaneously ───────────
            p_hat, v = ekf.update(p_meas, u, dt)
            a_est    = ekf.a       # ← gravity parameter from augmented state

            # ── 3. Reference trajectory ──────────────────────────────────────
            w     = CONFIG.TRAJ_FREQ * 2 * np.pi
            A     = np.radians(CONFIG.TRAJ_AMP)
            x_d   = A * np.sin(w * t_s)
            xd_d  = A * w * np.cos(w * t_s)
            xdd_d = -A * (w ** 2) * np.sin(w * t_s)

            # ── 4. Sliding surface ───────────────────────────────────────────
            e   = p_hat - x_d
            ed  = v - xd_d
            rho = ed + CONFIG.CTRL_C * e

            # ── 5. Control law ───────────────────────────────────────────────
            #   Gravity compensation uses the EKF's own estimate of 'a'.
            #   No separate RLS or regressor needed.
            u_ff = -xdd_d * CONFIG.I_INERTIA
            u_pd = CONFIG.CTRL_K_RHO * rho + CONFIG.I_INERTIA * CONFIG.CTRL_C * ed
            u_grav = a_est * np.sin(p_hat)          # adaptive gravity feed-forward
            u  = u_ff + u_pd + u_grav

            odrive.set_torque(u)

            # ── 6. Print ─────────────────────────────────────────────────────
            if int(t_s * 10) % 3 == 0:
                print(
                    f"T:{t_s:.1f}s | "
                    f"a_est:{a_est:.4f} | "
                    f"p:{np.degrees(p_hat):+.2f}deg | "
                    f"err:{np.degrees(e):+.2f}deg | "
                    f"P[a]:{ekf.P[2,2]:.4f}",
                    end='\r'
                )

            await asyncio.sleep(0.001)

    finally:
        odrive.set_torque(0)
        print("\nSafety stop: torque zeroed.")


# ═════════════════════════════════════════════════════════════════════════════
# SETUP
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
