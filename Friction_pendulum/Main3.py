import pyodrivecan
import asyncio
from datetime import datetime, timedelta
import smbus
import time

################## ENCODER ###############
bus = smbus.SMBus(1)

def read_raw_angle():
    # AS5600-ish style read (your original)
    data = bus.read_i2c_block_data(0x40, 0xFE, 2)
    return data[0] / 255 + data[1] / 64 / 255


################## CONTROL TASK ################
async def controller(odrive):
    # Put ODrive in velocity mode (must be in CLOSED LOOP)
    odrive.set_controller_mode("velocity_control")

    stop_at = datetime.now() + timedelta(seconds=10000)

    # Rate limit CAN commands (Hz)
    CMD_HZ = 200.0                 # 200 Hz is usually plenty
    DT = 1.0 / CMD_HZ

    last_sent_v = None
    next_send = time.monotonic()

    # Example: constant velocity command
    v_cmd = 1.0

    while datetime.now() < stop_at:
        now = time.monotonic()

        # only send at CMD_HZ
        if now >= next_send:
            next_send += DT

            # only transmit if value changed (prevents needless CAN spam)
            if last_sent_v != v_cmd:
                odrive.set_velocity(v_cmd)
                last_sent_v = v_cmd

        # yield to event loop (important)
        await asyncio.sleep(0.001)


################## MAIN ################
odrive = pyodrivecan.ODriveCAN(0)

async def main():
    odrive.clear_errors(identify=False)
    print("Cleared Errors")
    await asyncio.sleep(0.2)

    # init CAN
    odrive.initCanBus()

    print("Put Arm at bottom center to calibrate Zero Position.")
    await asyncio.sleep(1)

    cur_pos = odrive.position
    print(f"Encoder Absolute Position Set: {cur_pos}")

    # You CANNOT do velocity control in open_loop_control
    # Use closed_loop_control
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
