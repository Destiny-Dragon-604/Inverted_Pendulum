import pyodrivecan
import asyncio
import numpy as np
import smbus
import time
from PendulumEKF3 import PendulumEKF


class CONFIG:
    A_INIT     = 0
    Q_TORQUE   = 1.0
    Q_A        = 0.5
    I_INERTIA  = 0.037322
    CTRL_C     = 6.0
    CTRL_K_RHO = 1.5
    TRAJ_FREQ  = 1.2
    TRAJ_AMP   = 5.0


bus = smbus.SMBus(1)

def read_raw_angle():
    try:
        data = bus.read_i2c_block_data(0x40, 0xFE, 2)
        return data[0] / 255 + data[1] / 64 / 255
    except Exception:
        return 0.0


async def controller(odrive):
    await asyncio.sleep(1)
    odrive.set_controller_mode("torque_control")

    ekf = PendulumEKF(
        I        = CONFIG.I_INERTIA,
        a_init   = CONFIG.A_INIT,
        q_torque = CONFIG.Q_TORQUE,
        q_a      = CONFIG.Q_A,
    )

    rest_pos    = read_raw_angle()
    val, hturns = rest_pos, 0
    u           = 0.0
    t0          = time.monotonic()
    t_prev      = asyncio.get_running_loop().time()

    print("\n--- Augmented EKF Controller (DEBUG MODE) ---")
    print(f"{'T':>6} | {'a_est':>10} | {'P[a]':>10} | {'K_a':>10} | {'innov':>8} | {'sin(θ)':>8} | {'K_a*innov':>10}")
    print("-" * 80)

    try:
        while True:
            t_now  = asyncio.get_running_loop().time()
            dt     = max(t_now - t_prev, 1e-4)
            t_prev = t_now
            t_s    = time.monotonic() - t0

            # 1. Encoder
            prev_val = val
            val      = read_raw_angle()
            if (val - prev_val) < -0.5:
                hturns += 1
            elif (val - prev_val) > 0.5:
                hturns -= 1
            p_meas = (val + hturns - rest_pos) * np.pi

            # 2. EKF
            p_hat, v = ekf.update(p_meas, u, dt)
            a_est    = ekf.a

            # 3. Trajectory
            w     = CONFIG.TRAJ_FREQ * 2 * np.pi
            A     = np.radians(CONFIG.TRAJ_AMP)
            x_d   = A * np.sin(w * t_s)
            xd_d  = A * w * np.cos(w * t_s)
            xdd_d = -A * (w ** 2) * np.sin(w * t_s)

            # 4. Sliding surface
            e   = p_hat - x_d
            ed  = v - xd_d
            rho = ed + CONFIG.CTRL_C * e

            # 5. Control
            u_ff   = -xdd_d * CONFIG.I_INERTIA
            u_pd   = CONFIG.CTRL_K_RHO * rho + CONFIG.I_INERTIA * CONFIG.CTRL_C * ed
            u_grav = -a_est * np.sin(p_hat)   # cancel +a·sin(θ) in plant
            u      = u_ff + u_pd + u_grav
            odrive.set_torque(u)

            # 6. Debug print (every ~0.3 s)
            if int(t_s * 10) % 3 == 0:
                Ka_innov = ekf.K_a * ekf.innovation
                print(
                    f"{t_s:>6.1f} | "
                    f"{a_est:>+10.5f} | "
                    f"{ekf.P[2,2]:>10.5f} | "
                    f"{ekf.K_a:>+10.6f} | "
                    f"{ekf.innovation:>+8.5f} | "
                    f"{np.sin(p_hat):>+8.5f} | "
                    f"{Ka_innov:>+10.7f}",
                    end='\r'
                )

            await asyncio.sleep(0.001)

    finally:
        odrive.set_torque(0)
        print("\nSafety stop: torque zeroed.")


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
