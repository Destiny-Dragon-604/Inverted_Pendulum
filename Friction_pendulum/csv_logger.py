import pyodrivecan
import asyncio
from datetime import datetime, timedelta
import smbus
import time
import numpy as np
import csv
from pathlib import Path


################## CSV LOGGER ##################
class CsvLogger:
    """
    Standalone CSV logger (buffered + periodic flush).

    Saves to: <same_dir_as_this_file>/data/<prefix>_YYYYmmdd_HHMMSS.csv
    """
    def __init__(
        self,
        prefix="pendulum",
        fieldnames=None,
        flush_every=50,
        flush_interval_s=0.5,
    ):
        base_dir = Path(__file__).resolve().parent
        data_dir = base_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = data_dir / f"{prefix}_{ts}.csv"

        self.fieldnames = fieldnames or [
            "t_s",          # time since start (s)
            "dt_s",         # loop dt (s)
            "v_cmd_rad_s",  # commanded velocity (rad/s)
            "pos_turns",    # odrive.position (turns) (if available)
            "raw_abs_turns" # AS5600-like absolute [0..1) from I2C
        ]

        self._f = open(self.path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.fieldnames)
        self._w.writeheader()

        self._buf = []
        self._flush_every = int(flush_every)
        self._flush_interval_s = float(flush_interval_s)
        self._last_flush = time.monotonic()
        self._closed = False

    def log(self, **row):
        if self._closed:
            return
        clean = {k: row.get(k, "") for k in self.fieldnames}
        self._buf.append(clean)

        now = time.monotonic()
        if len(self._buf) >= self._flush_every or (now - self._last_flush) >= self._flush_interval_s:
            self.flush()

    def flush(self):
        if self._closed or not self._buf:
            return
        self._w.writerows(self._buf)
        self._buf.clear()
        self._f.flush()
        self._last_flush = time.monotonic()

    def close(self):
        if self._closed:
            return
        try:
            self.flush()
        finally:
            self._f.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


################## ENCODER ###############
bus = smbus.SMBus(1)

def read_raw_angle():
    data = bus.read_i2c_block_data(0x40, 0xFE, 2)
    return data[0] / 255 + data[1] / 64 / 255


################## RAMP PROFILE (MATLAB -> PYTHON) ###############
def ramp_profile(t: float, max_value: float, t_ramp: float = 5.0) -> float:
    """
    MATLAB-equivalent ramp(t, max_value)
    Piecewise profile with t_ramp-second segments.
    Total pattern length: 5*t_ramp seconds, then 0 afterwards.
    """
    if t < 0:
        return 0.0

    if t < t_ramp:
        return max_value * t / t_ramp
    elif t <= 2 * t_ramp:
        return t_ramp * max_value - max_value * (t - t_ramp) / t_ramp
    elif t < 3 * t_ramp:
        return 0.0
    elif t < 4 * t_ramp:
        return -max_value * (t - 3 * t_ramp) / t_ramp
    elif t <= 5 * t_ramp:
        return -t_ramp * max_value + max_value * (t - 4 * t_ramp) / t_ramp
    else:
        return 0.0


################## CONTROL TASK ################
async def controller(odrive, logger: CsvLogger):
    # Must be in CLOSED LOOP for velocity control
    odrive.set_controller_mode("velocity_control")

    stop_at = datetime.now() + timedelta(seconds=10000)

    # ===== Profile settings =====
    max_value = 1.7       # rad/s
    t_ramp = 15.0         # seconds
    T_cycle = 6 * t_ramp  # repeat period
    repeat = True         # set False to run one pass then hold 0

    # ===== CAN command rate limit =====
    CMD_HZ = 200.0
    DT = 1.0 / CMD_HZ
    next_send = time.monotonic()

    last_sent_v = None
    t0 = time.monotonic()
    last_log_t = t0

    while datetime.now() < stop_at:
        now = time.monotonic()

        if now >= next_send:
            next_send += DT

            t = now - t0
            if repeat:
                t_cycle = t % T_cycle
            else:
                t_cycle = min(t, T_cycle)

            v_cmd = ramp_profile(t_cycle, max_value=max_value, t_ramp=t_ramp)

            # only transmit when command actually changes a bit
            if (last_sent_v is None) or (abs(v_cmd - last_sent_v) > 1e-4):
                odrive.set_velocity(v_cmd)
                last_sent_v = v_cmd

            # ---- LOGGING ----
            dt_log = now - last_log_t
            last_log_t = now

            # raw absolute encoder (safe-guard if I2C glitches)
            try:
                raw_abs = read_raw_angle()
            except Exception:
                raw_abs = ""

            # odrive position (safe-guard if property access fails)
            try:
                pos_turns = odrive.position
            except Exception:
                pos_turns = ""

            logger.log(
                t_s=t,
                dt_s=dt_log,
                v_cmd_rad_s=v_cmd,
                pos_turns=pos_turns,
                raw_abs_turns=raw_abs,
            )

        await asyncio.sleep(0.001)


################## MAIN ################
odrive = pyodrivecan.ODriveCAN(0)

async def main():
    odrive.clear_errors(identify=False)
    print("Cleared Errors")
    await asyncio.sleep(0.2)

    odrive.initCanBus()

    print("Put Arm at bottom center to calibrate Zero Position.")
    await asyncio.sleep(1)

    cur_pos = odrive.position
    print(f"Encoder Absolute Position Set: {cur_pos}")

    # velocity control requires closed loop
    odrive.setAxisState("closed_loop_control")

    logger = CsvLogger(prefix="velocity_profile")

    try:
        print(f"Logging to: {logger.path}")
        await asyncio.gather(
            odrive.loop(),
            controller(odrive, logger),
        )
    finally:
        logger.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, stopping...")
        try:
            odrive.estop()
        except Exception as e:
            print("Failed to estop:", e)
