import pyodrivecan
import asyncio
import numpy as np
from datetime import datetime, timedelta
import smbus
import time
from pathlib import Path
from PendulumEKF import PendulumEKF

import csv


# ─────────────────────────────────────────────────────────────────────────────
#  RECURSIVE LEAST SQUARES  (scalar parameter version)
#
#  Model:   y[k]  =  φ[k] · a  +  noise
#
#  where
#    φ[k] = sin(θ[k])          (regressor — ACTUAL angle p, not desired x_d)
#    y[k] = u[k-1] - I·θ̈[k]   (physics residual from equation of motion)
#    a    = mgl / I             (unknown scalar to estimate)
#
#  Equation of motion:
#    I·θ̈  =  u  -  a·sin(θ)
#    →  a·sin(θ)  =  u  -  I·θ̈
#    →  y  =  u  -  I·θ̈,   φ  =  sin(θ)
#
#  This formulation is clean because:
#    - φ and y are tied directly to physics, not the control law
#    - θ̈ from the EKF is already filtered (low noise)
#    - u from the previous timestep makes it fully causal
#
#  Update equations  (with forgetting factor λ):
#    K[k]  = P[k-1]·φ[k] / (λ + φ[k]²·P[k-1])
#    a[k]  = a[k-1] + K[k]·(y[k] - φ[k]·a[k-1])
#    P[k]  = (1/λ)·(1 - K[k]·φ[k])·P[k-1]
#
#  P is the scalar covariance / "confidence" estimate.
#  Large P  → estimator moves quickly  (high initial uncertainty).
#  Small P  → estimator is locked in   (converged).
# ─────────────────────────────────────────────────────────────────────────────
class ScalarRLS:
    """
    Single-parameter recursive least squares with forgetting factor.

    Parameters
    ----------
    lam       : forgetting factor (0 < λ ≤ 1).
                λ = 1.0  → standard RLS, no forgetting (use if param is constant).
                λ = 0.98 → effective memory ≈ 50 samples — good default for
                           slowly time-varying physical parameters.
    P0        : initial covariance. Large = high initial uncertainty → moves fast.
    a0        : initial parameter estimate. Warm-start near known value if available.
    a_min/max : hard clamps on estimate (safety guard against wind-up).
    """

    def __init__(
        self,
        lam: float = 0.98,
        P0: float = 1000.0,
        a0: float = 0.0,
        a_min: float = -50.0,
        a_max: float = 50.0,
    ):
        self.lam   = float(lam)
        self.P     = float(P0)
        self.a     = float(a0)
        self.a_min = float(a_min)
        self.a_max = float(a_max)

        # Diagnostics (read-only after update)
        self.K     = 0.0   # last Kalman gain
        self.innov = 0.0   # last innovation  (y - φ·a_prev)

    def update(self, phi: float, y: float) -> float:
        """
        Run one RLS step.

        Parameters
        ----------
        phi : regressor  φ[k]  = sin(p)   (actual pendulum angle)
        y   : output     y[k]  = u - I·θ̈  (physics residual)

        Returns
        -------
        Updated scalar estimate  a[k]
        """
        denom = self.lam + phi * phi * self.P
        if abs(denom) < 1e-12:      # numerical guard (near-zero regressor)
            return self.a

        self.K     = self.P * phi / denom
        self.innov = y - phi * self.a
        self.a     = self.a + self.K * self.innov
        self.a     = max(self.a_min, min(self.a_max, self.a))   # clamp
        self.P     = (1.0 / self.lam) * (1.0 - self.K * phi) * self.P
        self.P     = max(self.P, 1e-6)   # covariance floor

        return self.a


################## CSV LOGGER ##################
class CsvLogger:
    def __init__(
        self,
        log_dir="data",
        prefix="pendulum",
        fieldnames=None,
        flush_every=50,
        flush_interval_s=0.5,
    ):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = Path(log_dir) / f"{prefix}_{ts}.csv"

        self.fieldnames = fieldnames or [
            "t_s",              # time since calibration (s)
            "p_turns",          # pendulum angle (turns)
            "p_rad",            # pendulum angle (rad)
            "v_turns_s",        # pendulum velocity (rad/s)
            "phi_rad_s",        # wheel speed (rad/s)
            "a_est",            # RLS estimate of  a = mgl/I
            "u_torque",         # commanded torque
            "raw_turns",        # raw encoder turns [0..~1)
            "rest_turns",       # calibration rest position
            "hturns",           # wrap counter
            "dt_s",             # loop dt
            "x_d_s",            # desired angle (rad)
            "rho_s",            # sliding surface σ = ė + C·e
            "SV_s",             # RLS covariance P (replaces SVD singular value)
            "accel_s",          # I·θ̈  (EKF acceleration estimate)
            "eq_trim_deg",      # auto trim (deg)
            "phi_filt_rad_s",   # filtered wheel speed used for trim
        ]

        self._f = open(self.path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.fieldnames)
        self._w.writeheader()

        self._buf            = []
        self._flush_every    = int(flush_every)
        self._flush_interval_s = float(flush_interval_s)
        self._last_flush     = time.monotonic()

    def log(self, **row):
        clean = {k: row.get(k, "") for k in self.fieldnames}
        self._buf.append(clean)
        now = time.monotonic()
        if (
            len(self._buf) >= self._flush_every
            or (now - self._last_flush) >= self._flush_interval_s
        ):
            self.flush()

    def flush(self):
        if not self._buf:
            return
        self._w.writerows(self._buf)
        self._buf.clear()
        self._f.flush()
        self._last_flush = time.monotonic()

    def close(self):
        try:
            self.flush()
        finally:
            self._f.close()


################## ENCODER ##################
bus = smbus.SMBus(1)


def read_raw_angle():
    data = bus.read_i2c_block_data(0x40, 0xFE, 2)
    return data[0] / 255 + data[1] / 64 / 255


################# WAIT #################
async def wait_for_enter(prompt: str = "Press ENTER to start..."):
    await asyncio.to_thread(input, prompt)


################# Filters ####################
class DirtyDerivative:
    """Low-pass filtered derivative: dx = a*dx + (1-a)*(x-x_prev)/dt"""

    def __init__(self, fc_hz: float, x0: float = 0.0):
        self.tau         = 1.0 / (2.0 * np.pi * float(fc_hz))
        self.x_prev      = float(x0)
        self.dx          = 0.0
        self.initialized = False

    def update(self, x: float, dt: float) -> float:
        if dt <= 0:
            return self.dx
        if not self.initialized:
            self.x_prev      = float(x)
            self.initialized = True
            return self.dx
        a        = self.tau / (self.tau + dt)
        self.dx  = a * self.dx + (1.0 - a) * (x - self.x_prev) / dt
        self.x_prev = float(x)
        return self.dx


class LowPass1:
    """1st-order low-pass: y += alpha*(x - y), alpha = dt/(tau+dt)."""

    def __init__(self, fc_hz: float, y0: float = 0.0):
        self.tau         = 1.0 / (2.0 * np.pi * float(fc_hz))
        self.y           = float(y0)
        self.initialized = False

    def update(self, x: float, dt: float) -> float:
        if dt <= 0:
            return self.y
        if not self.initialized:
            self.y           = float(x)
            self.initialized = True
            return self.y
        a      = dt / (self.tau + dt)
        self.y = self.y + a * (float(x) - self.y)
        return self.y


################## ODrive helpers ##################
async def get_odrive_velocity_rad_s(odrive) -> float:
    if hasattr(odrive, "get_velocity"):
        v = await odrive.get_velocity()
    elif hasattr(odrive, "velocity"):
        v = odrive.velocity
    elif hasattr(odrive, "vel_estimate"):
        v = odrive.vel_estimate
    else:
        v = 0.0
    return float(v) * 2.0 * np.pi


################## CONTROL LOOP ##################
async def controller(odrive):
    await asyncio.sleep(0.002)

    odrive.set_controller_mode("torque_control")
    stop_at   = datetime.now() + timedelta(seconds=10000)
    next_time = 0
    dt_target = 0.001

    # ── RLS estimator ─────────────────────────────────────────────────────────
    # λ = 0.98  →  effective memory window ≈ 1/(1-λ) = 50 samples (~50 ms)
    # P0 = 1000 →  start with high uncertainty so estimator moves fast early
    # a0 = 1.006 → warm-start near known physical value
    rls = ScalarRLS(lam=0.98, P0=1000.0, a0=1.006, a_min=-20.0, a_max=20.0)

    # ── Gains (unchanged from original) ──────────────────────────────────────
    C        = 6
    k_rho    = 1
    k_s      = 0.0       # sign term off — discontinuity corrupts RLS signal
    I        = 0.037322
    pi       = np.pi
    a_actual = 1.006

    ekf = PendulumEKF(I, a_actual)
    u   = 0.0            # initialise u so first RLS step is well-defined

    # ── AUTO TRIM ─────────────────────────────────────────────────────────────
    vel_lp          = LowPass1(fc_hz=0.5)
    trim_switch     = 0
    eq_trim_rad     = 0.0
    eq_trim_rad1    = 0.0
    eq_trim_deg     = 0.0
    kp_trim         = 0.01
    trim_k_i        = 0.0001
    trim_deadband   = 0.001
    trim_max_deg    = 1.00
    trim_slew_deg_s = 0.005

    # ── Calibration ───────────────────────────────────────────────────────────
    rest_pos = read_raw_angle()

    print("\nSystem initialized. Move pendulum near upright.")
    await wait_for_enter(
        "Press ENTER to start control (zero reference will be set now)... "
    )

    t0 = time.monotonic()

    # ── CSV logging ───────────────────────────────────────────────────────────
    log_frequency = 30
    log_period_s  = 1 / log_frequency
    last_log_s    = 0.0

    logger = CsvLogger(prefix="pendulum_rls")
    print(f"[CSV] Logging to: {logger.path}")

    # ── Encoder unwrap init ───────────────────────────────────────────────────
    val    = read_raw_angle()
    hturns = 0

    odrive.set_torque(0)

    # ── Desired trajectory ────────────────────────────────────────────────────
    freq  = 0.8
    omega = freq * 2 * pi
    A_deg = 15
    A     = (A_deg / 180.0) * pi

    loop   = asyncio.get_running_loop()
    t_prev = loop.time()

    # theta_DD from EKF — initialise so first RLS y is defined
    theta_DD = 0.0

    try:
        while datetime.now() < stop_at:
            next_time += dt_target
            await asyncio.sleep(max(0, next_time - loop.time()))

            # ── Encoder unwrap ────────────────────────────────────────────────
            prev_val = val
            val      = read_raw_angle()
            diff     = val - prev_val
            if diff < -0.5:
                hturns += 1
            elif diff > 0.5:
                hturns -= 1

            position = val + hturns
            p_meas   = (position - rest_pos) * pi
            p        = p_meas - eq_trim_rad1

            # ── Timing ───────────────────────────────────────────────────────
            t_now = loop.time()
            dt    = max(t_now - t_prev, 1e-6)
            t_s   = t_now - t0

            # ── Desired trajectory ────────────────────────────────────────────
            SIN_Wt = np.sin(-omega * t_s)
            x_d    =  A * SIN_Wt
            xd_d   = -A * omega * np.cos(-omega * t_s)
            xdd_d  = -A * omega * omega * SIN_Wt

            # ── State estimates ───────────────────────────────────────────────
            e        = p - x_d
            p_hat, v = ekf.update(p, u, dt)

            # theta_DD from EKF dynamics — used by RLS on the NEXT iteration
            # (causal: current u hasn't been computed yet this step)
            theta_DD_new = ekf.dynamics(
                np.array([[p_hat], [v]]), u
            )[1, 0]

            ed  = v - xd_d
            rho = ed + C * e

            # ── ODrive velocity + AUTO TRIM ───────────────────────────────────
            phi_wheel = await get_odrive_velocity_rad_s(odrive)
            phi_filt  = vel_lp.update(phi_wheel, dt)

            steady = (abs(v) < 0.5) and (abs(rho) < 0.5)
            if steady and abs(phi_filt) > trim_deadband:
                dtrim     = -trim_k_i * phi_filt * dt
                dtrim_max = (trim_slew_deg_s * pi / 180.0) * dt
                dtrim     = max(-dtrim_max, min(dtrim_max, dtrim))
                eq_trim_rad  += dtrim
                eq_trim_rad1  = trim_switch * (eq_trim_rad - kp_trim * phi_filt)
                trim_max_rad  = trim_max_deg * pi / 180.0
                eq_trim_rad1  = max(-trim_max_rad, min(trim_max_rad, eq_trim_rad1))

            eq_trim_deg = eq_trim_rad1 * 180.0 / pi

            # ── RLS update ────────────────────────────────────────────────────
            #
            # Equation of motion:  I·θ̈ = u - a·sin(θ)
            # Rearranged:          a·sin(θ) = u - I·θ̈
            #
            # Regressor:  φ = sin(p)           — actual angle
            # Output:     y = u - I·theta_DD   — uses u and theta_DD from the
            #                                    PREVIOUS iteration (causal)
            #
            phi_rls = np.sin(p)                      # scalar, actual angle
            y_rls   = float(u - I * theta_DD)        # physics residual (prev step)

            a_hat = rls.update(phi_rls, y_rls)

            # ── Control law ──────────────────────────────────────────────────
            u_ad = a_hat * np.sin(p)    # adaptive gravity feed-forward

            u = (
                -xdd_d * I
                + k_rho * rho
                + k_s * np.sign(rho)
                + 1.0 * e
                + I * C * ed
                + u_ad
                - 0.00 * phi_wheel
            )

            odrive.set_torque(u)

            # Store theta_DD for next iteration's RLS output
            theta_DD = theta_DD_new
            accel    = I * theta_DD

            # ── Log (rate-limited) ────────────────────────────────────────────
            if (t_s - last_log_s) >= log_period_s:
                logger.log(
                    t_s          = t_s,
                    p_turns      = p / (2 * pi),
                    p_rad        = p,
                    v_turns_s    = v,
                    phi_rad_s    = phi_wheel,
                    a_est        = a_hat,
                    u_torque     = u,
                    raw_turns    = val,
                    rest_turns   = rest_pos,
                    hturns       = hturns,
                    dt_s         = dt,
                    x_d_s        = x_d,
                    rho_s        = rho,
                    SV_s         = rls.P,
                    accel_s      = accel,
                    eq_trim_deg  = eq_trim_deg,
                    phi_filt_rad_s = phi_filt,
                )
                last_log_s = t_s

            t_prev = t_now

            print(
                f"a_hat={a_hat:.4f}  "
                f"P={rls.P:.4f}  "
                f"innov={rls.innov:+.4f}  "
                f"y_rls={y_rls:+.4f}  "
                f"phi_rls={phi_rls:+.4f}  "
                f"p={p*180/pi:.2f}deg  "
                f"trim={eq_trim_deg:+.2f}deg"
            )

            await asyncio.sleep(0)

    finally:
        try:
            odrive.set_torque(0)
        except Exception:
            pass
        logger.close()
        print("[CSV] Logger closed.")


################## ODRIVE SETUP ##################
odrive = pyodrivecan.ODriveCAN(0)


async def main():
    odrive.clear_errors(identify=False)
    print("Cleared Errors")
    await asyncio.sleep(1)

    odrive.initCanBus()

    print("Put Arm at bottom center to calibrate Zero Position.")
    await asyncio.sleep(1)
    cur_pos = odrive.position
    await asyncio.sleep(1)
    print(f"Encoder Absolute Position Set: {cur_pos}")

    odrive.setAxisState("open_loop_control")

    await asyncio.gather(
        odrive.loop(),
        controller(odrive),
    )


################ RUN #################
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, stopping...")
        try:
            odrive.set_torque(0)
        except Exception:
            pass
        try:
            odrive.estop()
        except Exception:
            pass
            '
