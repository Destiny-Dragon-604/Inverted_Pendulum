import pyodrivecan
import asyncio
import numpy as np
from datetime import datetime, timedelta
import smbus
import time
from pathlib import Path
import csv


# ─────────────────────────────────────────────────────────────────────────────
#  LEVANT EXACT DIFFERENTIATOR  (2nd order)
#
#  Gives position, velocity, AND acceleration from a noisy position signal.
#  Converges in finite time with bounded noise — completely model-free.
#
#  Reference: Levant, A. (2003). Higher-order sliding modes, differentiation
#             and output-feedback control. Int. J. Control, 76(9–10), 924–941.
#
#  States:
#    z0  →  tracks  x          (filtered position)
#    z1  →  tracks  x_dot      (velocity)
#    z2  →  tracks  x_ddot     (acceleration)
#
#  Tuning:
#    L   →  upper bound on |x_dddot|  (Lipschitz constant of acceleration).
#            For a pendulum: start L = 20–50. Increase if velocity/accel
#            lag behind or oscillate. Decrease if noisy.
#
#  Why this instead of double dirty-derivative?
#    - Dirty derivatives chain: noise ×(1/dt)² at high freq → huge accel noise
#    - Levant uses fractional-power terms that suppress noise much more
#      aggressively while preserving finite-time convergence.
# ─────────────────────────────────────────────────────────────────────────────
class LevantDiff:
    """
    3rd-order Levant exact differentiator.
    Estimates x, x_dot, x_ddot from noisy x measurements.

    Parameters
    ----------
    L       : Lipschitz constant of x_ddot. Tune this.
              Start at 20, increase if lagging, decrease if noisy.
    x0      : Initial position estimate (set to first measurement).
    """

    def __init__(self, L: float = 30.0, x0: float = 0.0):
        self.L  = float(L)
        self.z0 = float(x0)   # position estimate
        self.z1 = 0.0         # velocity estimate
        self.z2 = 0.0         # acceleration estimate
        self.initialized = False

    def update(self, x_meas: float, dt: float):
        """
        Parameters
        ----------
        x_meas  : raw position measurement (rad)
        dt      : timestep (s)

        Returns
        -------
        (z0, z1, z2) : (position, velocity, acceleration) estimates
        """
        if dt <= 0:
            return self.z0, self.z1, self.z2

        if not self.initialized:
            self.z0 = float(x_meas)
            self.initialized = True
            return self.z0, self.z1, self.z2

        L  = self.L
        e0 = self.z0 - x_meas

        # Levant gains (standard choices from the paper)
        l0 = 1.1 * L ** (1.0 / 3.0)
        l1 = 1.5 * L ** (0.5)
        l2 = 1.1 * L

        # Correction terms (fractional power preserves finite-time convergence)
        sig0 = l0 * (abs(e0) ** (2.0 / 3.0)) * np.sign(e0)
        sig1 = l1 * (abs(e0) ** (1.0 / 3.0)) * np.sign(e0)
        sig2 = l2 * np.sign(e0)

        # Derivatives of each state
        z0_dot = self.z1 - sig0
        z1_dot = self.z2 - sig1
        z2_dot =         - sig2

        # Euler integration  (RK2/RK4 not needed; step is already small)
        self.z0 += dt * z0_dot
        self.z1 += dt * z1_dot
        self.z2 += dt * z2_dot

        return self.z0, self.z1, self.z2


# ─────────────────────────────────────────────────────────────────────────────
#  RECURSIVE LEAST SQUARES  (scalar parameter)
#
#  Model:   y[k] = φ[k] · a  +  noise
#
#  where
#    φ[k] = sin(θ[k])          (regressor)
#    y[k] = u[k-1] - I·θ̈[k]   (from physics: I·θ̈ = u - a·sin(θ))
#    a    = mgl / I             (unknown — what we're estimating)
#
#  This is now DECOUPLED from the EKF.  θ̈ comes from the Levant
#  differentiator (raw encoder → Levant → accel), so the RLS is fitting
#  real physics, not circular model output.
#
#  Update equations  (forgetting factor λ):
#    K[k]  = P[k-1]·φ / (λ + φ²·P[k-1])
#    a[k]  = a[k-1] + K[k]·(y[k] - φ·a[k-1])
#    P[k]  = (1/λ)·(1 - K[k]·φ)·P[k-1]
# ─────────────────────────────────────────────────────────────────────────────
class ScalarRLS:
    """
    Single-parameter RLS with forgetting factor.

    Parameters
    ----------
    lam         : forgetting factor (0 < λ ≤ 1). λ=0.99 → memory ≈ 100 samples.
    P0          : initial covariance (large = trust data quickly).
    a0          : initial estimate (warm-start if you have a prior).
    a_min/a_max : hard clamps (safety).
    phi_min     : minimum |sin(θ)| to accept an update.  Prevents near-zero
                  regressor from blowing up the covariance update (excitation
                  condition).
    """

    def __init__(
        self,
        lam:   float = 0.990,
        P0:    float = 1000.0,
        a0:    float = 0.0,
        a_min: float = -20.0,
        a_max: float =  20.0,
        phi_min: float = 0.15,
    ):
        self.lam     = float(lam)
        self.P       = float(P0)
        self.a       = float(a0)
        self.a_min   = float(a_min)
        self.a_max   = float(a_max)
        self.phi_min = float(phi_min)

        # diagnostics
        self.K     = 0.0
        self.innov = 0.0

    def update(self, phi: float, y: float) -> float:
        """
        Run one RLS step.

        Only updates if |phi| > phi_min  (persistent excitation gate).

        Returns
        -------
        Current parameter estimate a (updated or held).
        """
        if abs(phi) < self.phi_min:
            return self.a   # not enough excitation — hold estimate

        denom = self.lam + phi * phi * self.P
        if abs(denom) < 1e-12:
            return self.a

        self.K     = self.P * phi / denom
        self.innov = y - phi * self.a

        self.a = self.a + self.K * self.innov
        self.a = max(self.a_min, min(self.a_max, self.a))   # clamp

        self.P = (1.0 / self.lam) * (1.0 - self.K * phi) * self.P
        self.P = max(self.P, 1e-6)   # numerical floor

        return self.a


# ─────────────────────────────────────────────────────────────────────────────
#  EKF  — used ONLY for state estimation (θ, ω) in the control law.
#  Its internal 'a' is updated from the RLS once confidence is high.
#  It no longer feeds back into the RLS (decoupled).
# ─────────────────────────────────────────────────────────────────────────────
class PendulumEKF:
    """
    Error-state EKF for state estimation only.
    Parameter 'a' is set externally by the RLS once it converges.
    """

    def __init__(self, I: float, a: float):
        self.I   = float(I)
        self.a   = float(a)
        self.fc  = 0.0
        self.fv  = 0.0
        self.v_eps = 0.1

        self.x = np.zeros((2, 1))   # [theta, omega]
        self.P = np.eye(2) * 0.1

        self.q_torque = 100.0
        self.R = np.array([[1e-4]])
        self.initialized = False

    @staticmethod
    def wrap_angle(a: float) -> float:
        return (a + np.pi) % (2 * np.pi) - np.pi

    def dynamics(self, x: np.ndarray, u: float) -> np.ndarray:
        theta, omega = x[0, 0], x[1, 0]
        friction = self.fc * np.tanh(omega / self.v_eps) + self.fv * omega
        theta_dd = (u - self.a * np.sin(theta) - friction) / self.I
        return np.array([[omega], [theta_dd]])

    def _jacobian(self) -> np.ndarray:
        theta, omega = self.x[0, 0], self.x[1, 0]
        sech2 = 1.0 - np.tanh(omega / self.v_eps) ** 2
        return np.array([
            [0.0, 1.0],
            [-self.a * np.cos(theta) / self.I,
             (-self.fc * sech2 / self.v_eps - self.fv) / self.I],
        ])

    def _discretize(self, A: np.ndarray, dt: float) -> np.ndarray:
        Adt = A * dt
        return np.eye(2) + Adt + 0.5 * (Adt @ Adt)

    def _process_noise(self, dt: float) -> np.ndarray:
        G = np.array([[0.0], [1.0 / self.I]])
        return G @ np.array([[self.q_torque]]) @ G.T * dt

    def update(self, theta_meas: float, torque_u: float, dt: float):
        if dt <= 0:
            return self.x[0, 0], self.x[1, 0]

        if not self.initialized:
            self.x[0, 0] = theta_meas
            self.initialized = True
            return self.x[0, 0], self.x[1, 0]

        # Propagate
        self.x += dt * self.dynamics(self.x, torque_u)
        self.x[0, 0] = self.wrap_angle(self.x[0, 0])

        A = self._jacobian()
        F = self._discretize(A, dt)
        Q = self._process_noise(dt)
        self.P = F @ self.P @ F.T + Q

        # Measurement update
        H = np.array([[1.0, 0.0]])
        innov = self.wrap_angle(theta_meas - self.x[0, 0])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ np.array([[innov]])
        self.x[0, 0] = self.wrap_angle(self.x[0, 0])

        I2 = np.eye(2)
        self.P = (I2 - K @ H) @ self.P @ (I2 - K @ H).T + K @ self.R @ K.T

        return self.x[0, 0], self.x[1, 0]

    def set_a(self, a: float):
        """Update model parameter from RLS. Inflates P slightly."""
        self.a = float(a)
        self.P += np.diag([5e-3, 2e-2])


# ─────────────────────────────────────────────────────────────────────────────
#  CSV LOGGER
# ─────────────────────────────────────────────────────────────────────────────
class CsvLogger:
    FIELDS = [
        "t_s",           # time since start (s)
        "p_rad",         # pendulum angle, unwrapped + trim (rad)
        "p_meas_rad",    # raw unwrapped angle before trim (rad)
        "v_ekf",         # EKF velocity estimate (rad/s)
        "v_levant",      # Levant velocity estimate (rad/s)
        "acc_levant",    # Levant acceleration estimate (rad/s²)
        "a_est",         # RLS estimate of  mgl/I
        "rls_P",         # RLS covariance (confidence proxy)
        "rls_innov",     # RLS innovation
        "phi_rls",       # RLS regressor sin(θ)
        "y_rls",         # RLS output signal
        "u_torque",      # commanded torque (Nm)
        "x_d",           # desired angle (rad)
        "e",             # tracking error (rad)
        "rho",           # sliding surface
        "eq_trim_deg",   # auto-trim offset (deg)
        "phi_wheel",     # wheel speed (rad/s)
        "dt_s",          # loop timestep
    ]

    def __init__(self, log_dir="data", prefix="pendulum", flush_every=50,
                 flush_interval_s=0.5):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = Path(log_dir) / f"{prefix}_{ts}.csv"
        self._f = open(self.path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._w.writeheader()
        self._buf = []
        self._flush_every   = int(flush_every)
        self._flush_interval = float(flush_interval_s)
        self._last_flush    = time.monotonic()

    def log(self, **row):
        self._buf.append({k: row.get(k, "") for k in self.FIELDS})
        now = time.monotonic()
        if len(self._buf) >= self._flush_every or \
                (now - self._last_flush) >= self._flush_interval:
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


# ─────────────────────────────────────────────────────────────────────────────
#  FILTERS
# ─────────────────────────────────────────────────────────────────────────────
class LowPass1:
    """1st-order low-pass: y += alpha*(x - y), alpha = dt/(tau+dt)."""

    def __init__(self, fc_hz: float, y0: float = 0.0):
        self.tau = 1.0 / (2.0 * np.pi * float(fc_hz))
        self.y   = float(y0)
        self.initialized = False

    def update(self, x: float, dt: float) -> float:
        if dt <= 0:
            return self.y
        if not self.initialized:
            self.y = float(x)
            self.initialized = True
            return self.y
        a    = dt / (self.tau + dt)
        self.y += a * (float(x) - self.y)
        return self.y


# ─────────────────────────────────────────────────────────────────────────────
#  HARDWARE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
bus = smbus.SMBus(1)


def read_raw_angle() -> float:
    data = bus.read_i2c_block_data(0x40, 0xFE, 2)
    return data[0] / 255 + data[1] / 64 / 255


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


async def wait_for_enter(prompt: str = "Press ENTER to continue..."):
    await asyncio.to_thread(input, prompt)


# ─────────────────────────────────────────────────────────────────────────────
#  CONTROL LOOP
# ─────────────────────────────────────────────────────────────────────────────
async def controller(odrive):
    await asyncio.sleep(0.002)
    odrive.set_controller_mode("torque_control")

    # ── Physical constants ────────────────────────────────────────────────────
    I   = 0.037322   # moment of inertia (kg·m²)
    pi  = np.pi

    # ── Control gains ─────────────────────────────────────────────────────────
    C     = 6.0      # sliding surface gain  (rho = e_dot + C*e)
    k_rho = 1.0      # proportional on sliding surface
    k_s   = 0.0      # switching gain (set > 0 for robustness, but chatters)

    # ── Desired trajectory ────────────────────────────────────────────────────
    freq  = 0.8
    omega = freq * 2.0 * pi
    A_deg = 15.0
    A     = (A_deg / 180.0) * pi

    # ── EKF — state estimation only ───────────────────────────────────────────
    # a_init is a rough prior.  RLS will correct it once it converges.
    a_init = 0.5
    ekf = PendulumEKF(I=I, a=a_init)

    # ── Levant differentiator ─────────────────────────────────────────────────
    # L = upper bound on |theta_ddd| for your pendulum.
    # For a typical reaction-wheel pendulum: L=30 is a safe starting point.
    # If velocity/accel look laggy → increase L.
    # If they look noisy → decrease L.
    levant = LevantDiff(L=3.0)

    # ── RLS estimator (fully decoupled from EKF) ──────────────────────────────
    # a0 = 0.0 means "no prior" — let the data speak.
    # phi_min = 0.15 rad: only update when |sin(θ)| > 0.15 (PE condition).
    rls = ScalarRLS(lam=0.990, P0=1000.0, a0=0.0, a_min=-20.0, a_max=20.0,
                    phi_min=0.15)

    # Threshold below which we trust the RLS enough to feed back into EKF
    RLS_CONFIDENCE_THRESH = 0.5   # P < this → feed a_hat into EKF
    ekf_updated_from_rls  = False

    # ── Auto-trim ─────────────────────────────────────────────────────────────
    vel_lp          = LowPass1(fc_hz=0.5)
    trim_switch     = 0             # set to 1 to enable trim
    eq_trim_rad     = 0.0
    eq_trim_rad1    = 0.0
    kp_trim         = 0.01
    trim_k_i        = 0.0001
    trim_deadband   = 0.001
    trim_max_deg    = 1.00
    trim_slew_deg_s = 0.005

    # ── Calibration ───────────────────────────────────────────────────────────
    rest_pos = read_raw_angle()
    print("\nSystem initialized. Move pendulum near upright.")
    await wait_for_enter(
        "Press ENTER to start control (zero reference set now)... "
    )

    t0   = time.monotonic()
    loop = asyncio.get_running_loop()

    # ── Encoder unwrap state ──────────────────────────────────────────────────
    val    = read_raw_angle()
    hturns = 0

    # ── Logging ───────────────────────────────────────────────────────────────
    log_frequency = 30
    log_period_s  = 1.0 / log_frequency
    last_log_s    = 0.0
    logger = CsvLogger(prefix="pendulum_levant_rls")
    print(f"[CSV] Logging to: {logger.path}")

    # ── Timing ────────────────────────────────────────────────────────────────
    dt_target = 0.001
    next_time = loop.time()
    t_prev    = loop.time()

    odrive.set_torque(0)
    u = 0.0   # u[k-1] for causal RLS output

    stop_at = datetime.now() + timedelta(seconds=10000)

    try:
        while datetime.now() < stop_at:
            next_time += dt_target
            await asyncio.sleep(max(0.0, next_time - loop.time()))

            # ── Timing ───────────────────────────────────────────────────────
            t_now = loop.time()
            dt    = t_now - t_prev
            if dt <= 0:
                dt = 1e-6
            t_s    = t_now - t0
            t_prev = t_now

            # ── Encoder unwrap ────────────────────────────────────────────────
            prev_val = val
            val      = read_raw_angle()
            diff     = val - prev_val
            if diff < -0.5:
                hturns += 1
            elif diff > 0.5:
                hturns -= 1

            position = val + hturns
            p_meas   = (position - rest_pos) * pi   # raw unwrapped (rad)
            p        = p_meas - eq_trim_rad1         # trim-corrected angle

            # ── Levant differentiator ─────────────────────────────────────────
            # Inputs raw angle (not trim-corrected — trim only shifts zero,
            # the differentiated velocity/accel are trim-invariant).
            _, v_levant, acc_levant = levant.update(p_meas, dt)

            # ── EKF state estimate (θ, ω) ─────────────────────────────────────
            p_ekf, v_ekf = ekf.update(p, u, dt)

            # ── Desired trajectory ────────────────────────────────────────────
            SIN_Wt = np.sin(-omega * t_s)
            x_d    =  A * SIN_Wt
            xd_d   = -A * omega * np.cos(-omega * t_s)
            xdd_d  = -A * omega * omega * SIN_Wt

            # ── Tracking errors ───────────────────────────────────────────────
            e  = p - x_d
            ed = v_ekf - xd_d
            rho = ed + C * e

            # ── Wheel speed + auto-trim ───────────────────────────────────────
            phi_wheel = await get_odrive_velocity_rad_s(odrive)
            phi_filt  = vel_lp.update(phi_wheel, dt)

            steady = (abs(v_ekf) < 0.5) and (abs(rho) < 0.5)
            if steady and abs(phi_filt) > trim_deadband:
                dtrim     = -trim_k_i * phi_filt * dt
                dtrim_max = (trim_slew_deg_s * pi / 180.0) * dt
                dtrim     = max(-dtrim_max, min(dtrim_max, dtrim))
                eq_trim_rad  += dtrim
                eq_trim_rad1  = trim_switch * (eq_trim_rad - kp_trim * phi_filt)
                trim_max_rad  = trim_max_deg * pi / 180.0
                eq_trim_rad1  = max(-trim_max_rad, min(trim_max_rad, eq_trim_rad1))

            eq_trim_deg = eq_trim_rad1 * 180.0 / pi

            # ── RLS parameter estimation ──────────────────────────────────────
            #
            # Physics:   I * theta_DD = u - a * sin(θ)
            # Rearrange: a * sin(θ)   = u - I * theta_DD
            #
            # Regressor:  phi_rls = sin(θ)            [from encoder]
            # Output:     y_rls   = u_prev - I * acc  [u from previous step]
            #
            # acc_levant comes from the Levant differentiator acting on the
            # raw encoder — NO EKF involved. This breaks the circular
            # dependency entirely.
            #
            # Gate: only update when:
            #   1. |sin(θ)| > phi_min     (excitation condition, inside RLS)
            #   2. |acc_levant| < acc_max  (reject spike transients)
            #   3. |v_levant|   < vel_max  (only near upright, model valid)
            #
            ACC_MAX = 50.0   # rad/s² — reject Levant transients at startup
            VEL_MAX =  3.0   # rad/s  — model linearisation valid near upright

            phi_rls = np.sin(p)
            y_rls   = u - I * acc_levant    # u is from PREVIOUS loop iteration

            if abs(acc_levant) < ACC_MAX and abs(v_levant) < VEL_MAX:
                a_hat = rls.update(phi_rls, y_rls)
            else:
                a_hat = rls.a   # hold — Levant hasn't converged yet

            # ── Feed RLS result into EKF once confident ───────────────────────
            # Only do this once (or you'll keep inflating EKF covariance).
            if rls.P < RLS_CONFIDENCE_THRESH and not ekf_updated_from_rls:
                ekf.set_a(a_hat)
                ekf_updated_from_rls = True
                print(f"[RLS→EKF] a updated: {a_hat:.4f}  (P={rls.P:.4f})")

            # ── Control law ───────────────────────────────────────────────────
            # u_ad = a_hat * sin(θ)  (adaptive feedforward — enable once
            # RLS has converged, i.e. ekf_updated_from_rls is True).
            u_ad = a_hat * phi_rls if ekf_updated_from_rls else 0.0

            u = (
                - xdd_d * I
                + k_rho * rho
                + k_s   * np.sign(rho)
                + 1.0   * e
                + I * C * ed
                + u_ad               # adaptive gravity compensation
                - 0.0 * phi_wheel    # back-EMF term (off)
            )

            odrive.set_torque(u)

            # ── Logging ───────────────────────────────────────────────────────
            if (t_s - last_log_s) >= log_period_s:
                logger.log(
                    t_s=t_s,
                    p_rad=p,
                    p_meas_rad=p_meas,
                    v_ekf=v_ekf,
                    v_levant=v_levant,
                    acc_levant=acc_levant,
                    a_est=a_hat,
                    rls_P=rls.P,
                    rls_innov=rls.innov,
                    phi_rls=phi_rls,
                    y_rls=y_rls,
                    u_torque=u,
                    x_d=x_d,
                    e=e,
                    rho=rho,
                    eq_trim_deg=eq_trim_deg,
                    phi_wheel=phi_wheel,
                    dt_s=dt,
                )
                last_log_s = t_s

            print(
                f"t={t_s:6.2f}s | "
                f"a_hat={a_hat:+.4f}  P={rls.P:.4f} | "
                f"acc_L={acc_levant:+6.2f}  v_L={v_levant:+5.2f} | "
                f"θ={p*180/pi:+6.2f}°  ρ={rho:+.3f} | "
                f"u={u:+.4f}"
            )

            await asyncio.sleep(0)

    finally:
        try:
            odrive.set_torque(0)
        except Exception:
            pass
        logger.close()
        print("[CSV] Logger closed.")


# ─────────────────────────────────────────────────────────────────────────────
#  ODRIVE SETUP & ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
odrive = pyodrivecan.ODriveCAN(0)


async def main():
    odrive.clear_errors(identify=False)
    print("Cleared errors.")
    await asyncio.sleep(1)

    odrive.initCanBus()

    print("Put arm at bottom centre to calibrate zero position.")
    await asyncio.sleep(1)
    cur_pos = odrive.position
    await asyncio.sleep(1)
    print(f"Encoder absolute position set: {cur_pos}")

    odrive.setAxisState("open_loop_control")

    await asyncio.gather(
        odrive.loop(),
        controller(odrive),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("KeyboardInterrupt — stopping.")
        try:
            odrive.set_torque(0)
        except Exception:
            pass
        try:
            odrive.estop()
        except Exception:
            pass
