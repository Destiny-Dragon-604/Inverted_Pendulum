import pyodrivecan
import asyncio
import numpy as np
from datetime import datetime, timedelta
import smbus
import time
from pathlib import Path
import csv


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
            "t_s",          # time since calibration (s)
            "p_turns",      # pendulum angle (turns)
            "p_rad",        # pendulum angle (rad)
            "v_turns_s",    # pendulum velocity (turns/s)
            "phi_rad_s",    # wheel speed (rad/s)
            "a_est",        # your estimator state
            "u_torque",     # commanded torque
            "raw_turns",    # raw encoder turns [0..~1)
            "rest_turns",   # calibration rest position
            "hturns",       # wrap counter
            "dt_s",         # loop dt
            "x_d_s",        # desired state store
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

        a = self.tau / (self.tau + dt)  # recompute using actual dt
        dx_raw = (x - self.x_prev) / dt
        self.dx = a * self.dx + (1.0 - a) * dx_raw
        self.x_prev = float(x)
        return self.dx



################## CONTROL LOOP ##################
async def controller(odrive):
    await asyncio.sleep(0)

    odrive.set_controller_mode("torque_control")
    stop_at = datetime.now() + timedelta(seconds=10000)
    
    p_dot_filt = DirtyDerivative(fc_hz=20.0)   # try 20 Hz first


    # Gains
    C = 3
    gamma = 100.1
    pi = np.pi
    I = 0.037322

    # ---- Calibration moment ----
    rest_pos = read_raw_angle()
    
    print("\nSystem initialized. Move pendulum near upright (doesn't need to be perfect).")
    await wait_for_enter("Press ENTER to start control (zero reference will be set now)... ")
    
    t0 = time.monotonic()

    # ---- CSV log rate control (decimation) ----
    log_frequency = 30    # Hz
    log_period_s = 1/log_frequency   #  logging (increase to log less)
    last_log_s = 0.0

    logger = CsvLogger(prefix="pendulum_control")
    print(f"[CSV] Logging to: {logger.path}")

    # Initialize state
    val = read_raw_angle()
    hturns = 0

    odrive.set_torque(10)

    p_last = 0.0
    a_last = 20 # should be 26.74

    loop = asyncio.get_running_loop()
    t_prev = loop.time()

    try:
        while datetime.now() < stop_at:
            # ---- Encoder unwrap ----
            prev_val = val
            val = read_raw_angle()
            diff = val - prev_val

            if diff < -0.5:
                hturns += 1
            elif diff > 0.5:
                hturns -= 1

            position = val + hturns
            p = position - rest_pos
            p = p * pi  # radians

            # ---- Timing ----
            t_now = loop.time()
            dt = t_now - t_prev
            if dt <= 0:
                dt = 1e-6

            # ---- Define Error ----
            freq = 1
            omega = freq*2*pi
            A_deg = 4
            A = (A_deg / 180) * pi
            t_s = time.monotonic() - t0
            t = t_s

            SIN_Wt = np.sin(-omega * t)
            x_d = A * SIN_Wt
            xd_d = -A * omega * np.cos(-omega * t )
            xdd_d = -A * omega * omega * SIN_Wt

            # ---- State estimates ----
            e = p - x_d
            v = p_dot_filt.update(p,dt)
            v_drive = 0 #await odrive.get_velocity()
            phi = v_drive * 2 * pi
            ed = v - xd_d

            SIN = np.sin(p)
            a = a_last + dt * gamma * SIN * e

            # ---- Control law ----
            u = (1 / I) * (
                xdd_d
                + 4 * np.sign(e + C * ed)
                + 12 * e
                + 1 * ed
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
                )
                last_log_s = t_s

            # ---- Update state ----
            p_last = p
            a_last = a
            t_prev = t_now

            print(
                f"estimation = {a:.6f}, "
                f"p (deg) = {(p * 180 / pi):.6f}"
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
