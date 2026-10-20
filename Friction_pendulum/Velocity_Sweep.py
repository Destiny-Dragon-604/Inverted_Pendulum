"""
Velocity_Sweep.py (friction sweep + trapezoid profile for ODrive)

Default output folder:
  <this_script_dir>/data

Modes:
  --simulate : simulation (PI velocity loop on a simulated plant)
  --odrive   : ODrive trapezoid velocity profile 0 -> wmax -> 0 (continuous)

Key features:
- Python 3.8/3.9 compatible (no `|` union types, no `list[str]`)
- Output to ./data by default
- Sequential-safe filenames (3 files share the same _000N suffix)
- Trapezoid profile with independent ramp-up / hold / ramp-down times:
    --t-up, --t-hold, --t-down
- Best-effort motor stop on normal exit, exceptions, Ctrl+C, SIGTERM

Run examples:
  SIM:
    python3 Velocity_Sweep.py --simulate --wmax 1.5 --t-up 5 --t-hold 1 --t-down 5
  ODrive trapezoid:
    python3 Velocity_Sweep.py --odrive --wmax 1.5 --t-up 5 --t-hold 1 --t-down 5
  Both:
    python3 Velocity_Sweep.py --simulate --odrive --wmax 1.5 --t-up 5 --t-hold 1 --t-down 5
"""

import argparse
import asyncio
import atexit
import csv
import signal
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple

import numpy as np


# ---------------------------- Velocity cutoff (ADDED) ----------------------------

VEL_CUTOFF_RAD_S = 1.0

def cutoff_vel(omega_rad_s: Union[np.ndarray, float], cutoff: float = VEL_CUTOFF_RAD_S):
    """
    Hard cutoff: if |omega| < cutoff -> 0.
    Works for float or ndarray.
    """
    w = np.asarray(omega_rad_s, dtype=float)
    w2 = np.where(np.abs(w) < float(cutoff), 0.0, w)
    return float(w2) if np.isscalar(omega_rad_s) else w2


# ---------------------------- Friction model ----------------------------

def tau_fric(
    omega_rad_s: Union[np.ndarray, float],
    fc: float,
    fv: float,
    omega_scale: float = 1000.0,
) -> np.ndarray:
    """
    Smooth coulomb + viscous:
      tau = fc * tanh(omega*omega_scale) + fv*omega
    """
    w = cutoff_vel(omega_rad_s)  # <-- ADDED cutoff
    w = np.array(w, dtype=float)
    return np.tanh(w * omega_scale) * fc + fv * w


# ---------------------------- Regression helper (ADDED) ----------------------------

def fit_fc_fv_from_odrive(
    omega_meas_rad_s: np.ndarray,
    tau_cmd_Nm: np.ndarray,
    omega_scale: float = 1000.0,
) -> Tuple[float, float]:
    """
    Fit: tau = tanh(omega*omega_scale)*fc - fv*omega

    Linear least squares:
      tau = fc*x1 + fv*x2
      x1 = tanh(omega*omega_scale)
      x2 = -omega
    """
    omega = np.asarray(omega_meas_rad_s, dtype=float).reshape(-1)
    tau = np.asarray(tau_cmd_Nm, dtype=float).reshape(-1)

    m = np.isfinite(omega) & np.isfinite(tau)
    omega = omega[m]
    tau = tau[m]

    # --- ADDED: drop near-zero velocities from fit ---
    keep = np.abs(omega) >= VEL_CUTOFF_RAD_S
    omega = omega[keep]
    tau = tau[keep]

    if omega.size < 10:
        raise ValueError(f"Not enough valid samples to fit (got {omega.size}).")

    x1 = np.tanh(omega * omega_scale)
    x2 = -omega
    A = np.column_stack([x1, x2])  # parameters: [fc, fv]

    theta, *_ = np.linalg.lstsq(A, tau, rcond=None)
    fc_hat = float(theta[0])
    fv_hat = float(theta[1])
    return fc_hat, fv_hat


# ---------------------------- filename helpers ----------------------------

def unique_triplet(raw_path: Path, sum_path: Path, curve_path: Path) -> Tuple[Path, Path, Path]:
    """
    Ensures the 3 files for one run share the SAME suffix if needed.
    (raw, summary, curve will all get _000N together)
    """
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if (not raw_path.exists()) and (not sum_path.exists()) and (not curve_path.exists()):
        return raw_path, sum_path, curve_path

    i = 1
    while True:
        r = raw_path.parent / f"{raw_path.stem}_{i:04d}{raw_path.suffix}"
        s = sum_path.parent / f"{sum_path.stem}_{i:04d}{sum_path.suffix}"
        c = curve_path.parent / f"{curve_path.stem}_{i:04d}{curve_path.suffix}"
        if (not r.exists()) and (not s.exists()) and (not c.exists()):
            return r, s, c
        i += 1


# ---------------------------- CSV logger (buffered + periodic flush) ----------------------------

class CsvLogger:
    """
    Buffered CSV logger with periodic flush.
    """
    def __init__(
        self,
        path: Path,
        fieldnames: List[str],
        flush_every: int = 200,
        flush_interval_s: float = 0.5,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fieldnames = fieldnames

        self._f = open(self.path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.fieldnames)
        self._w.writeheader()

        self._buf: List[Dict[str, Any]] = []
        self._flush_every = int(flush_every)
        self._flush_interval_s = float(flush_interval_s)
        self._last_flush = time.monotonic()
        self._closed = False

    def log(self, **row: Any) -> None:
        if self._closed:
            return
        clean = {k: row.get(k, "") for k in self.fieldnames}
        self._buf.append(clean)

        now = time.monotonic()
        if len(self._buf) >= self._flush_every or (now - self._last_flush) >= self._flush_interval_s:
            self.flush()

    def flush(self) -> None:
        if self._closed or not self._buf:
            return
        self._w.writerows(self._buf)
        self._buf.clear()
        self._f.flush()
        self._last_flush = time.monotonic()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
        finally:
            self._f.close()
            self._closed = True


# ---------------------------- Profiles ----------------------------

def trapezoid_profile(t: float, t_up: float, t_hold: float, t_down: float, w_max: float) -> float:
    """
    0 -> w_max over t_up
    hold w_max for t_hold
    w_max -> 0 over t_down
    then hold 0.
    """
    t_up = max(0.0, float(t_up))
    t_hold = max(0.0, float(t_hold))
    t_down = max(0.0, float(t_down))

    if t <= 0.0:
        return 0.0

    if t_up > 0.0 and t < t_up:
        return w_max * (t / t_up)

    t1 = t_up + t_hold
    if t < t1:
        return w_max

    t2 = t1 + t_down
    if t_down > 0.0 and t < t2:
        return w_max * (1.0 - (t - t1) / t_down)

    return 0.0


# ---------------------------- SIM mode ----------------------------

@dataclass
class SimParams:
    J: float = 0.02
    fc: float = 5.0
    fv: float = 5.0
    omega_scale: float = 1000.0

    w_max: float = 1.5
    t_up: float = 5.0
    t_hold: float = 0.5
    t_down: float = 5.0

    sample_hz: float = 200.0
    kp: float = 2.0
    ki: float = 10.0
    tau_limit: float = 20.0


def run_simulation(out_dir: Path, p: SimParams) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_base = out_dir / f"friction_sweep_raw_sim_{ts}.csv"
    sum_base = out_dir / f"friction_sweep_summary_sim_{ts}.csv"
    curve_base = out_dir / f"friction_model_curve_sim_{ts}.csv"
    raw_path, sum_path, curve_path = unique_triplet(raw_base, sum_base, curve_base)

    # model curve
    omega_curve = np.linspace(-p.w_max, p.w_max, 600)
    tau_curve = tau_fric(omega_curve, p.fc, p.fv, p.omega_scale)
    with open(curve_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["omega_rad_s", "tau_model_Nm"])
        w.writerows(zip(omega_curve.tolist(), tau_curve.tolist()))

    dt = 1.0 / float(p.sample_hz)
    t_total = float(p.t_up + p.t_hold + p.t_down) + 0.5
    N = int(t_total / dt)

    raw_logger = CsvLogger(
        path=raw_path,
        fieldnames=["t_s", "omega_ref_rad_s", "omega_rad_s", "tau_cmd_Nm", "tau_fric_Nm", "accel_rad_s2"],
        flush_every=200,
        flush_interval_s=0.5,
    )

    omega = 0.0
    integ = 0.0
    omegas: List[float] = []
    taus_est: List[float] = []

    try:
        for k in range(N):
            t_s = k * dt
            omega_ref = trapezoid_profile(t_s, p.t_up, p.t_hold, p.t_down, p.w_max)
            omega_ref = float(cutoff_vel(omega_ref))  # <-- ADDED cutoff on command

            e = omega_ref - omega
            integ += e * dt
            tau_cmd = p.kp * e + p.ki * integ
            tau_cmd = float(np.clip(tau_cmd, -p.tau_limit, p.tau_limit))

            tauF = float(tau_fric(omega, p.fc, p.fv, p.omega_scale))
            domega = (tau_cmd - tauF) / p.J
            omega = omega + domega * dt
            omega = float(cutoff_vel(omega))          # <-- ADDED cutoff on state (optional but consistent)

            raw_logger.log(
                t_s=t_s,
                omega_ref_rad_s=omega_ref,
                omega_rad_s=omega,
                tau_cmd_Nm=tau_cmd,
                tau_fric_Nm=tauF,
                accel_rad_s2=domega,
            )

            omegas.append(float(omega))
            taus_est.append(float(tauF))
    finally:
        raw_logger.close()

    # summary: binned mean tau vs omega
    omega_arr = np.array(omegas, dtype=float)
    tau_arr = np.array(taus_est, dtype=float)

    nbins = 120
    bins = np.linspace(float(np.min(omega_arr)), float(np.max(omega_arr)), nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    tau_mean = np.full(nbins, np.nan, dtype=float)

    for i in range(nbins):
        m = (omega_arr >= bins[i]) & (omega_arr < bins[i + 1])
        if np.any(m):
            tau_mean[i] = float(np.mean(tau_arr[m]))

    with open(sum_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["omega_rad_s", "tau_fric_est_Nm"])
        for c, tm in zip(centers.tolist(), tau_mean.tolist()):
            if np.isfinite(tm):
                w.writerow([c, tm])

    print(f"[SIM] Wrote:\n  {curve_path}\n  {raw_logger.path}\n  {sum_path}")


# ---------------------------- ODrive helpers ----------------------------

async def maybe_await(x: Any) -> Any:
    return await x if asyncio.iscoroutine(x) else x


def _try_call(fn, *args) -> bool:
    try:
        fn(*args)
        return True
    except TypeError:
        return False
    except Exception:
        return False


async def set_velocity_cmd_turns_s(odrive: Any, v_turns_s: float) -> None:
    """
    Send velocity command in turns/s (ODrive native).
    Tries common wrapper method names + signatures.
    """
    candidates = [
        "set_velocity", "set_vel", "set_input_vel",
        "setVelocity", "setVel", "setInputVel",
        "input_vel", "inputVel",
    ]
    for name in candidates:
        if hasattr(odrive, name):
            fn = getattr(odrive, name)

            if not callable(fn):
                try:
                    setattr(odrive, name, v_turns_s)
                    return
                except Exception:
                    pass

            try:
                fn(v_turns_s)
                return
            except TypeError:
                pass

            try:
                fn(0, v_turns_s)  # axis-indexed signature
                return
            except TypeError:
                pass

    raise AttributeError("No velocity command method found on odrive object.")


async def set_controller_velocity_mode(odrive: Any) -> None:
    for name in ["set_controller_mode", "setControllerMode", "set_control_mode"]:
        if hasattr(odrive, name):
            fn = getattr(odrive, name)
            if _try_call(fn, "velocity_control") or _try_call(fn, "VELOCITY_CONTROL") or _try_call(fn, 2):
                return


async def set_axis_state_closed_loop(odrive: Any) -> None:
    for name in ["setAxisState", "set_axis_state", "setState", "set_state"]:
        if hasattr(odrive, name):
            fn = getattr(odrive, name)
            if _try_call(fn, "closed_loop_control") or _try_call(fn, "CLOSED_LOOP_CONTROL") or _try_call(fn, 8):
                return


async def set_axis_state_idle(odrive: Any) -> None:
    for name in ["setAxisState", "set_axis_state", "setState", "set_state"]:
        if hasattr(odrive, name):
            fn = getattr(odrive, name)
            if _try_call(fn, "idle") or _try_call(fn, "IDLE") or _try_call(fn, 1):
                return


async def get_velocity_turns_s(odrive: Any) -> float:
    for attr in ["get_velocity", "velocity", "vel", "getVel", "get_vel", "get_input_vel"]:
        if hasattr(odrive, attr):
            v = getattr(odrive, attr)
            v = await maybe_await(v() if callable(v) else v)
            return float(v)
    raise AttributeError("Cannot read velocity (turns/s).")


async def get_iq_measured_A(odrive: Any) -> Optional[float]:
    for attr in ["get_iq_measured", "iq_measured", "get_current", "current", "getIqMeasured", "get_iq"]:
        if hasattr(odrive, attr):
            x = getattr(odrive, attr)
            x = await maybe_await(x() if callable(x) else x)
            try:
                return float(x)
            except Exception:
                return None
    return None


async def emergency_stop_async(odrive: Any) -> None:
    """
    Best-effort stop sequence.
    """
    try:
        await set_velocity_cmd_turns_s(odrive, 0.0)
    except Exception:
        pass

    for name in ["set_torque", "setTorque", "set_input_torque", "setInputTorque"]:
        if hasattr(odrive, name):
            try:
                getattr(odrive, name)(0.0)
            except Exception:
                pass

    try:
        await set_axis_state_idle(odrive)
    except Exception:
        pass

    if hasattr(odrive, "estop"):
        try:
            odrive.estop()
        except Exception:
            pass


# ---------------------------- ODrive trapezoid run ----------------------------

@dataclass
class ODriveParams:
    w_max_rad_s: float = 1.5
    t_up: float = 12.0
    t_hold: float = 1
    t_down: float = 12.0
    sample_hz: float = 200.0
    torque_constant_Nm_per_A: Optional[float] = None


async def odrive_trapezoid_profile(out_dir: Path, odrive: Any, p: ODriveParams) -> None:
    """
    Continuous trapezoid command 0 -> +wmax -> 0.
    Logs command, measured omega, iq, and tau_cmd (iq*Kt).
    Fits fc, fv after the run using:
        tau = tanh(omega*1000)*fc - fv*omega
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_base  = out_dir / f"friction_sweep_raw_odrive_profile_{ts}.csv"
    sum_base  = out_dir / f"friction_sweep_summary_odrive_profile_{ts}.csv"
    curve_base = out_dir / f"friction_model_curve_odrive_profile_{ts}.csv"
    raw_path, sum_path, curve_path = unique_triplet(raw_base, sum_base, curve_base)

    raw_logger = CsvLogger(
        path=raw_path,
        fieldnames=["t_s", "omega_cmd_rad_s", "omega_meas_rad_s", "iq_measured_A", "tau_cmd_Nm"],
        flush_every=200,
        flush_interval_s=0.5,
    )

    dt = 1.0 / float(p.sample_hz)
    t_total = float(p.t_up + p.t_hold + p.t_down) + 0.5
    print(f"[ODRIVE] t_total={t_total:.2f}s (t_up={p.t_up}, t_hold={p.t_hold}, t_down={p.t_down})")

    if p.torque_constant_Nm_per_A is None:
        print("[ODRIVE] WARNING: --Kt not provided. Regression needs tau_cmd = iq*Kt, so fc/fv will NOT be fit.")

    t0 = time.monotonic()

    # For fitting (only samples where both omega and tau_cmd exist)
    omega_fit: List[float] = []
    tau_fit: List[float] = []

    print("[ODRIVE] Starting trapezoid profile...")

    try:
        while True:
            t_s = time.monotonic() - t0
            if t_s > t_total:
                break

            omega_cmd = trapezoid_profile(t_s, p.t_up, p.t_hold, p.t_down, p.w_max_rad_s)
            omega_cmd = float(cutoff_vel(omega_cmd))  # <-- ADDED cutoff on command
            await set_velocity_cmd_turns_s(odrive, omega_cmd / (2.0 * np.pi))

            v_meas_turns_s = await get_velocity_turns_s(odrive)
            omega_meas = float(v_meas_turns_s) * 2.0 * np.pi
            omega_meas = float(cutoff_vel(omega_meas))  # <-- ADDED cutoff on measured omega

            iq = await get_iq_measured_A(odrive)

            tau_cmd: Optional[float] = None
            if (iq is not None) and (p.torque_constant_Nm_per_A is not None):
                tau_cmd = float(iq) * float(p.torque_constant_Nm_per_A)

                # --- ADDED: only collect fit samples when |omega| >= cutoff ---
                if abs(omega_meas) >= VEL_CUTOFF_RAD_S:
                    omega_fit.append(omega_meas)
                    tau_fit.append(tau_cmd)

            raw_logger.log(
                t_s=t_s,
                omega_cmd_rad_s=omega_cmd,
                omega_meas_rad_s=omega_meas,
                iq_measured_A=iq if iq is not None else "",
                tau_cmd_Nm=tau_cmd if tau_cmd is not None else "",
            )

            await asyncio.sleep(dt)

    finally:
        raw_logger.close()
        await emergency_stop_async(odrive)

    # ---------------- regression + outputs ----------------
    fc_hat: Optional[float] = None
    fv_hat: Optional[float] = None

    if p.torque_constant_Nm_per_A is not None and len(tau_fit) >= 10:
        try:
            fc_hat, fv_hat = fit_fc_fv_from_odrive(
                omega_meas_rad_s=np.array(omega_fit, dtype=float),
                tau_cmd_Nm=np.array(tau_fit, dtype=float),
                omega_scale=1000.0,
            )
            print(f"[FIT] fc = {fc_hat:.6g}  (Nm)")
            print(f"[FIT] fv = {fv_hat:.6g}  (Nms/rad)")
        except Exception as e:
            print(f"[FIT] Regression failed: {e}")

    # summary file: write params + binned curve (tau vs omega)
    with open(sum_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["meta_key", "meta_value"])
        w.writerow(["Kt_Nm_per_A", "" if p.torque_constant_Nm_per_A is None else p.torque_constant_Nm_per_A])
        w.writerow(["fc_fit_Nm", "" if fc_hat is None else fc_hat])
        w.writerow(["fv_fit_Nms_per_rad", "" if fv_hat is None else fv_hat])
        w.writerow([])

        if len(tau_fit) >= 10:
            omega_arr = np.array(omega_fit, dtype=float)
            tau_arr = np.array(tau_fit, dtype=float)

            nbins = 120
            bins = np.linspace(float(np.min(omega_arr)), float(np.max(omega_arr)), nbins + 1)
            centers = 0.5 * (bins[:-1] + bins[1:])
            tau_mean = np.full(nbins, np.nan, dtype=float)

            for i in range(nbins):
                m = (omega_arr >= bins[i]) & (omega_arr < bins[i + 1])
                if np.any(m):
                    tau_mean[i] = float(np.mean(tau_arr[m]))

            w.writerow(["omega_meas_rad_s", "tau_cmd_mean_Nm"])
            for c, tm in zip(centers.tolist(), tau_mean.tolist()):
                if np.isfinite(tm):
                    w.writerow([c, tm])

    # model curve file (from fitted params if available)
    with open(curve_path, "w", newline="") as f:
        w = csv.writer(f)
        if fc_hat is None or fv_hat is None:
            w.writerow(["note"])
            w.writerow(["fit not available (missing --Kt or insufficient valid samples)"])
        else:
            omega_curve = np.linspace(-p.w_max_rad_s, p.w_max_rad_s, 600)
            tau_curve = np.tanh(omega_curve * 1000.0) * fc_hat - fv_hat * omega_curve
            w.writerow(["omega_rad_s", "tau_model_fit_Nm"])
            w.writerows(zip(omega_curve.tolist(), tau_curve.tolist()))

    print(f"[ODRIVE] Wrote:\n  {curve_path}\n  {raw_path}\n  {sum_path}")



# ---------------------------- CLI ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None, help="output directory (default: ./data next to script)")

    ap.add_argument("--simulate", action="store_true", help="run simulation")
    ap.add_argument("--odrive", action="store_true", help="run ODrive trapezoid profile")

    # targets + timing
    ap.add_argument("--wmax", type=float, default=1.5, help="max speed (rad/s)")
    ap.add_argument("--t-up", type=float, default=5.0, help="ramp up time (s)")
    ap.add_argument("--t-hold", type=float, default=0.5, help="hold at wmax time (s)")
    ap.add_argument("--t-down", type=float, default=5.0, help="ramp down time (s)")

    # sim-only params
    ap.add_argument("--fc", type=float, default=5.0)
    ap.add_argument("--fv", type=float, default=5.0)
    ap.add_argument("--omega-scale", type=float, default=1000.0)
    ap.add_argument("--J", type=float, default=0.02)
    ap.add_argument("--kp", type=float, default=2.0)
    ap.add_argument("--ki", type=float, default=10.0)
    ap.add_argument("--tau-limit", type=float, default=20.0)
    ap.add_argument("--sim-hz", type=float, default=200.0)

    # odrive params
    ap.add_argument("--odrive-hz", type=float, default=200.0)
    ap.add_argument("--Kt", type=float, default=None, help="torque constant Nm/A (optional)")
    import sys
    print("sys.argv =", sys.argv)
    args = ap.parse_args()
    print(f"ARGS: t_up={args.t_up}, t_hold={args.t_hold}, t_down={args.t_down}, wmax={args.wmax}")

    if args.out is None:
        out_dir = Path(__file__).resolve().parent / "data"
    else:
        out_dir = Path(args.out)

    # default safe behavior
    if not args.simulate and not args.odrive:
        args.odrive = True

    if args.simulate:
        sp = SimParams(
            J=args.J,
            fc=args.fc,
            fv=args.fv,
            omega_scale=args.omega_scale,
            w_max=args.wmax,
            t_up=args.t_up,
            t_hold=args.t_hold,
            t_down=args.t_down,
            sample_hz=args.sim_hz,
            kp=args.kp,
            ki=args.ki,
            tau_limit=args.tau_limit,
        )
        run_simulation(out_dir, sp)

    if args.odrive:
        import pyodrivecan
        odrive = pyodrivecan.ODriveCAN(0)

        # --- sync stop fallback for signals/atexit ---
        def emergency_stop_sync():
            try:
                for name in ["set_velocity", "set_vel", "set_input_vel", "setVelocity", "setVel", "setInputVel"]:
                    if hasattr(odrive, name):
                        fn = getattr(odrive, name)
                        try:
                            fn(0.0)
                            break
                        except Exception:
                            pass

                for name in ["set_torque", "setTorque", "set_input_torque", "setInputTorque"]:
                    if hasattr(odrive, name):
                        try:
                            getattr(odrive, name)(0.0)
                        except Exception:
                            pass

                for name in ["setAxisState", "set_axis_state", "setState", "set_state"]:
                    if hasattr(odrive, name):
                        fn = getattr(odrive, name)
                        if _try_call(fn, "idle") or _try_call(fn, "IDLE") or _try_call(fn, 1):
                            break

                if hasattr(odrive, "estop"):
                    try:
                        odrive.estop()
                    except Exception:
                        pass
            except Exception:
                pass

        atexit.register(emergency_stop_sync)

        def _sig_handler(signum, frame):
            emergency_stop_sync()
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        async def runner() -> None:
            if hasattr(odrive, "clear_errors"):
                try:
                    odrive.clear_errors(identify=False)
                except TypeError:
                    odrive.clear_errors()

            if hasattr(odrive, "initCanBus"):
                odrive.initCanBus()

            await set_controller_velocity_mode(odrive)
            await set_axis_state_closed_loop(odrive)

            op = ODriveParams(
                w_max_rad_s=args.wmax,
                t_up=args.t_up,
                t_hold=args.t_hold,
                t_down=args.t_down,
                sample_hz=args.odrive_hz,
                torque_constant_Nm_per_A=args.Kt,
            )

            try:
                if hasattr(odrive, "loop"):
                    await asyncio.gather(
                        odrive.loop(),
                        odrive_trapezoid_profile(out_dir, odrive, op),
                    )
                else:
                    await odrive_trapezoid_profile(out_dir, odrive, op)
            finally:
                await emergency_stop_async(odrive)

        try:
            asyncio.run(runner())
        except KeyboardInterrupt:
            print("KeyboardInterrupt: stopping...")
            emergency_stop_sync()


if __name__ == "__main__":
    main()
