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
    # Returns a value ~[0, 1) in "turns" based on your current scaling
    data = bus.read_i2c_block_data(0x40, 0xFE, 2)
    return data[0] / 255 + data[1] / 64 / 255


################## CONTROL LOOP ##################
async def controller(odrive):
    await asyncio.sleep(0)

    # Controller configuration
    odrive.set_controller_mode("torque_control")
    stop_at = datetime.now() + timedelta(seconds=10000)

    # Gains
    C = 3
    gamma = 10.1
    pi = np.pi
    I = .037322

    # ---- Calibration moment ----
    rest_pos = read_raw_angle()
    t0 = time.monotonic()
    log_period_s = 0.02   # 50 Hz logging (increase to log less)
    last_log_s = 0.0


    logger = CsvLogger(prefix="pendulum_control")
    print(f"[CSV] Logging to: {logger.path}")

    # Initialize state
    val = read_raw_angle()
    hturns = 0

    odrive.set_torque(10)   # initial torque kick (keep if intentional)

    p_last = 0.0
    a_last = 26.74

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
            p = position - rest_pos  # pendulum angle relative to calibration (turns)
            p = p*pi #convert to radians

            # ---- Timing ----
            t_now = loop.time()
            dt = t_now - t_prev
            if dt <= 0:
                dt = 1e-6
                
            # ---- Define Error ----
            freq = 1
            omega = freq*2*pi #frequency Hz
            A_deg = 7
            A = (A_deg/180)*pi #amplitude in rad
            t = t_now - t0
            SIN_Wt = np.sin(omega*t)
            x_d = A*SIN_Wt
            xd_d = A*omega*np.cos(omega*t)
            xdd_d = -A*omega*omega*SIN_Wt

            # ---- State estimates ----
            e = p - x_d
            v = (p - p_last) / dt                 # pendulum velocity (turns/s)
            v_drive = await odrive.get_velocity() # whatever units your ODrive returns
            phi = v_drive * 2 * pi                # wheel speed (rad/s) (as you had it)
            ed = v - xd_d

            SIN = np.sin(p)
            a = a_last - dt * gamma * SIN * e

            # ---- Control law ----
            u = 1/I*( xdd_d + 0*np.sign(e + C*ed) - 12*e + 1*ed + 1*a*SIN - 0.05*phi + 0*np.exp(-t/10)/20 )
            odrive.set_torque(u)

            # ---- Log (rate-limited) ----
            t_s = time.monotonic() - t0
            
			if (t_s - last_log_s) >= log_period_s:
				logger.log(
					t_s=t_s,
					p_turns=p/(2*pi),
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

            print(f"estimation = {a:.6f}, phi = {phi:.6f} rad/s, p (deg) = {(p*180/pi):.6f}")
            await asyncio.sleep(0)

    finally:
        # Best-effort safe shutdown + flush CSV
        try:
            odrive.set_torque(0)
        except Exception:
            pass
        logger.close()
        print("[CSV] Logger closed.")


################## ODRIVE SETUP ##################
# Set up Node_ID 10 (your comment says ACTIVE NODE ID = 10)
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

    # odrive.setAxisState("closed_loop_control")
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
