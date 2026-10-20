import pyodrivecan
import asyncio
from datetime import datetime, timedelta
import smbus
import time
import numpy as np

################## ENCODER ###############
bus = smbus.SMBus(1)

def read_raw_angle():
    data = bus.read_i2c_block_data(0x40, 0xFE, 2)
    return data[0] / 255 + data[1] / 64 / 255


################## RAMP PROFILE (MATLAB -> PYTHON) ###############
def ramp_profile(t: float, max_value: float, t_ramp: float = 5.0) -> float:
    """
    MATLAB-equivalent ramp(t, max_value)
    Piecewise profile with 5-second segments (t_ramp).
    Total pattern length: 5*t_ramp = 25 seconds, then 0 afterwards.
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
async def controller(odrive):
    # Must be in CLOSED LOOP for velocity control
    odrive.set_controller_mode("velocity_control")

    stop_at = datetime.now() + timedelta(seconds=10000)

    # ===== Profile settings =====
    max_value = 1.5      # rad/s (your "max_value")
    t_ramp = 15.0         # seconds (same as MATLAB)
    T_cycle = 6*t_ramp       # you used linspace(0,30)
    repeat = True        # set False to run one 0-30s pass then hold 0

    # ===== CAN command rate limit =====
    CMD_HZ = 200.0
    DT = 1.0 / CMD_HZ
    next_send = time.monotonic()

    last_sent_v = None
    t0 = time.monotonic()

    while datetime.now() < stop_at:
        now = time.monotonic()

        if now >= next_send:
            next_send += DT

            t = now - t0
            if repeat:
                t = t % T_cycle
            else:
                if t > T_cycle:
                    t = T_cycle  # clamp

            v_cmd = ramp_profile(t, max_value=max_value, t_ramp=t_ramp)

            # only transmit when command actually changes a bit
            # (prevents spamming tiny float changes)
            if (last_sent_v is None) or (abs(v_cmd - last_sent_v) > 1e-4):
                odrive.set_velocity(v_cmd)
                last_sent_v = v_cmd

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

    await asyncio.gather(
        odrive.loop(),
        controller(odrive),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, stopping...")
        try:
            odrive.estop()
        except Exception as e:
            print("Failed to estop:", e)
