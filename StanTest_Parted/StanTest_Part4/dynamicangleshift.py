import pyodrivecan
import asyncio
import numpy as np
from datetime import datetime, timedelta
import smbus
import time
from pathlib import Path
from SVM_helper_functions2 import MinSVD
import csv
import sys

# ---------- Terminal single-key (no Enter) support (Unix-like) ----------
# Works on Linux / most terminal environments.
try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None


class RawTerminal:
    """Context manager to put terminal into cbreak mode so we can read single keys."""
    def __init__(self):
        self.enabled = (termios is not None) and sys.stdin.isatty()
        self.fd = None
        self.old = None

    def __enter__(self):
        if not self.enabled:
            return self
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return
        try:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
        except Exception:
            pass


async def keypress_task(key_queue: asyncio.Queue):
    """
    Reads single characters and pushes them into an asyncio queue.
    Uses raw terminal mode if available; otherwise falls back to line-based input.
    """
    if (termios is not None) and sys.stdin.isatty():
        # raw single-key mode
        with RawTerminal():
            while True:
                ch = await asyncio.to_thread(sys.stdin.read, 1)
                if not ch:
                    await asyncio.sleep(0.01)
                    continue
                await key_queue.put(ch)
    else:
        # fallback: requires Enter
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                await asyncio.sleep(0.01)
                continue
            for ch in line.strip():
                await key_queue.put(ch)


################## CSV LOGGER ##################
class CsvLogger:
    def __init__(
        self,
        log_dir="logs",
        prefix="pendulum",
        fieldnames=None,
        flush_every=50,
        flush_interval_s=0.5,
    ):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = Path(log_dir) / f"{prefix}_{ts}.csv"

        self.fieldnames = fieldnames or [
            "t_s",
            "p_turns",
            "p_rad",
            "v_turns_s",
            "phi_rad_s",
            "a_est",
            "u_torque",
            "raw_turns",
            "rest_turns",
            "hturns",
            "dt_s",
            "x_d_s",
            "rho_s",
            "SV_s",
            "eq_trim_deg",   # <-- added
        ]

        self._f = open(self.path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.fieldnames)
        self._w.writeheader()

        self._buf = []
        self._flush_every = int(flush_every)
        self._flush_interval_s = float(flush_interval_s)
        self._last_flush = time.monotonic()

    def log(self, **row):
        clean = {k: row.get(k, "") for k in self.fieldnames}
        self._buf.append(clean)

        now = time.monotonic()
        if len(self._buf) >= self._flush_every or (now - self._last_flush) >= self._flush_interval_s:
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


# torque ramp up
def ramp_up_torque(t, ramp_time, max_torque):
    if t < ramp_time:
        return max_torque * t / ramp_time
    return max_torque


################## ENCODER ##################
bus = smbus.SMBus(1)

def read_raw_angle():
    data = bus.read_i2c_block_data(0x40, 0xFE, 2)
    return data[0] / 255 + data[1] / 64 / 255


################# WAIIIITTT #################
async def wait_for_enter(prompt: str = "Press ENTER to start..."):
    await asyncio.to_thread(input, prompt)


################# Filter ####################
class DirtyDerivative:
    """
    Low-pass filtered numerical derivative:
        dx = a*dx + (1-a)*(x - x_prev)/dt
    where a = tau/(tau + dt), tau = 1/(2*pi*fc)
    """
    def __init__(self, fc_hz: float, x0: float = 0.0):
        self.fc_hz = float(fc_hz)
        self.tau = 1.0 / (2.0 * np.pi * self.fc_hz)
        self.x_prev = float(x0)
        self.dx = 0.0
        self.initialized = False

    def update(self, x: float, dt: float) -> float:
        if dt <= 0:
            return self.dx

        if not self.initialized:
            self.x_prev = float(x)
            self.dx = 0.0
            self.initialized = True
            return self.dx

        a = self.tau / (self.tau + dt)
        dx_raw = (x - self.x_prev) / dt
        self.dx = a * self.dx + (1.0 - a) * dx_raw
        self.x_prev = float(x)
        return self.dx


################## CONTROL LOOP ##################
async def controller(odrive):
    await asyncio.sleep(0)

    odrive.set_controller_mode("torque_control")
    stop_at = datetime.now() + timedelta(seconds=10000)

    p_dot_filt = DirtyDerivative(fc_hz=20.0)
    v_dot_filt = DirtyDerivative(fc_hz=20.0)

    SVD = MinSVD(p_max=5, eta=.15)

    # Gains
    C = 4
    k_rho = 4
    k_a = .001
    gamma = 50.1
    pi = np.pi
    I = 0.037322

    # ---- Calibration moment ----
    rest_pos = read_raw_angle()

    print("\nSystem initialized. Move pendulum near upright (doesn't need to be perfect).")
    await wait_for_enter("Press ENTER to start control (zero reference will be set now)... ")

    # ---- keyboard command queue ----
    key_q: asyncio.Queue = asyncio.Queue()
    key_task = asyncio.create_task(keypress_task(key_q))
    print("Key controls:  u = +0.01° trim,  d = -0.01° trim  (single key, no Enter)")

    # ---- equilibrium trim (deg/rad) ----
    eq_trim_deg = 0.0
    eq_trim_step_deg = 0.01
    eq_trim_rad = 0.0  # kept in sync with deg

    t0 = time.monotonic()

    # ---- CSV log rate control (decimation) ----
    log_frequency = 30
    log_period_s = 1 / log_frequency
    last_log_s = 0.0

    logger = CsvLogger(prefix="pendulum_control")
    print(f"[CSV] Logging to: {logger.path}")

    # Initialize state
    val = read_raw_angle()
    hturns = 0

    odrive.set_torque(10)

    a_last = 0

    loop = asyncio.get_running_loop()
    t_prev = loop.time()

    # ---- terminal/status print rate limiting ----
    status_hz = 5.0
    status_period = 1.0 / status_hz
    last_status_s = 0.0

    try:
        while datetime.now() < stop_at:
            # ---- handle keypresses (non-blocking) ----
            # consume all pending keys each iteration
            while True:
                try:
                    ch = key_q.get_nowait()
                except asyncio.QueueEmpty:
                    break

                if ch == "u":
                    eq_trim_deg += eq_trim_step_deg
                    eq_trim_rad = eq_trim_deg * pi / 180.0
                    print(f"\n[EQ TRIM] {eq_trim_deg:+.2f} deg")
                elif ch == "d":
                    eq_trim_deg -= eq_trim_step_deg
                    eq_trim_rad = eq_trim_deg * pi / 180.0
                    print(f"\n[EQ TRIM] {eq_trim_deg:+.2f} deg")
                # optional: quick reset
                elif ch == "r":
                    eq_trim_deg = 0.0
                    eq_trim_rad = 0.0
                    print(f"\n[EQ TRIM] reset to {eq_trim_deg:+.2f} deg")

            # ---- Encoder unwrap ----
            prev_val = val
            val = read_raw_angle()
            diff = val - prev_val

            if diff < -0.5:
                hturns += 1
            elif diff > 0.5:
                hturns -= 1

            position = val + hturns

            # Raw angle about "rest_pos"
            p_meas = (position - rest_pos) * pi  # radians (your original scaling)

            # APPLY equilibrium trim: shift what "center" means
            # Positive trim means we subtract more -> controller thinks pendulum is more negative.
            p = p_meas - eq_trim_rad

            # ---- Timing ----
            t_now = loop.time()
            dt = t_now - t_prev
            if dt <= 0:
                dt = 1e-6

            # ---- Define Error ----
            freq = .8
            omega = freq * 2 * pi
            A_deg = 0
            A = (A_deg / 180) * pi
            t_s = time.monotonic() - t0
            t = t_s

            SIN_Wt = np.sin(-omega * t)
            offset = (.1) * pi / 180
            x_d = A * SIN_Wt + offset
            xd_d = -A * omega * np.cos(-omega * t)
            xdd_d = -A * omega * omega * SIN_Wt

            # ---- State estimates ----
            e = p - x_d
            v = p_dot_filt.update(p, dt)
            v_drive = 0  # await odrive.get_velocity()
            phi = v_drive * 2 * pi
            ed = v - xd_d
            rho = ed + C * e

            theta_DD = v_dot_filt.update(v, dt)

            SIN = np.sin(p)
            SUM = SVD.sum_components_scalar()
            a = a_last + dt * (gamma * SIN * rho + 1 * gamma * k_a * SUM)

            # ---- Control law ----
            u = (
                + k_rho * rho
                + 0.0 * np.sign(rho)
                + 1 * e
                + I * C * ed
                + 1 * a * SIN
                - 0.05 * phi
            )

            odrive.set_torque(u)

            # ---- Log (rate-limited) ----
            if (t_s - last_log_s) >= log_period_s:
                logger.log(
                    t_s=t_s,
                    p_turns=p / (2 * pi),
                    p_rad=p,
                    v_turns_s=v,
                    phi_rad_s=phi,
                    a_est=a,
                    u_torque=u,
                    raw_turns=val,
                    rest_turns=rest_pos,
                    hturns=hturns,
                    dt_s=dt,
                    x_d_s=x_d,
                    rho_s=rho,
                    SV_s=SVD.SV,
                    eq_trim_deg=eq_trim_deg,
                )
                last_log_s = t_s

            # ---- Update state ----
            _SV = SVD.update(phi=SIN, epsilon=(I * theta_DD - u + a * SIN))
            a_last = a
            t_prev = t_now

            # ---- Status print (RATE LIMITED, single updating line) ----
            if (t_s - last_status_s) >= status_period:
                p_deg = p * 180 / pi
                # carriage return + pad to overwrite old content
                msg = (
                    f"\rtrim={eq_trim_deg:+.2f} deg | "
                    f"p={p_deg:+7.2f} deg | "
                    f"a={a:+8.2f} | "
                    f"SV={SVD.SV:+7.2f} | "
                    f"SUM={SUM:+7.2f}   "
                )
                sys.stdout.write(msg)
                sys.stdout.flush()
                last_status_s = t_s

            await asyncio.sleep(0)

    finally:
        try:
            key_task.cancel()
        except Exception:
            pass

        # make sure we end the status line cleanly
        try:
            sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception:
            pass

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
