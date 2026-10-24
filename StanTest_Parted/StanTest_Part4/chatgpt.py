import pyodrivecan
import asyncio
import math
import smbus
from datetime import datetime, timedelta

############################
# USER PHYSICAL CONSTANTS
############################

MASS = 0.7        # kg
LENGTH = 0.178     # meters
GRAVITY = 9.81

# Torque needed just to *hold* upright:
TAU_GRAV = MASS * GRAVITY * LENGTH   # ≈ 0.539 Nm

# Your motor torque constant:
# If unknown, assume Kt ≈ 0.08 Nm/A (typical hobby BLDC)
Kt = 0.08

# Estimated amps required to fight gravity:
AMPS_REQUIRED = TAU_GRAV / Kt

print(f"[INFO] Static holding torque ≈ {TAU_GRAV:.3f} Nm")
print(f"[INFO] Estimated amps needed ≈ {AMPS_REQUIRED:.1f} A")

############################
# CONTROL GAINS
############################

# You said torque isn't enough, so allow higher current:
MAX_AMPS = max(5.0, AMPS_REQUIRED * 1.5)  # 150% of what gravity needs
K_AMPS = MAX_AMPS / (math.pi/3)           # reacts strongly within ±60°

LOOP_DT = 0.001  # 1 ms loop

############################
# ENCODER SETUP
############################

bus = smbus.SMBus(1)

def read_raw_angle():
    data = bus.read_i2c_block_data(0x40, 0xFE, 2)
    return data[0] / 255 + data[1] / 64 / 255

def normalize(pos, rest):
    return pos - rest

############################
# RAW TORQUE CONTROLLER
############################

async def controller(odrive):
    await asyncio.sleep(1)

    print("[INFO] Switching to TORQUE_CONTROL (raw amps mode)")
    odrive.set_controller_mode("torque_control")
    await asyncio.sleep(0.3)

    rest_pos = read_raw_angle()
    prev_val = rest_pos
    hturns = 0

    stop_at = datetime.now() + timedelta(hours=1)

    print(f"[INFO] Using MAX_AMPS={MAX_AMPS:.1f}, K_AMPS={K_AMPS:.2f}")
    print("[INFO] Remember: firmware current limit must be raised via odrivetool!")
    print()

    while datetime.now() < stop_at:
        # --- Encoder unwrap ---
        raw = read_raw_angle()
        diff = raw - prev_val
        prev_val = raw

        if diff < -0.5:   hturns += 1
        elif diff > 0.5:  hturns -= 1

        position = normalize(raw + hturns, rest_pos)
        angle_rad = position * math.pi

        # ----- RAW CURRENT REQUEST -----
        amps_cmd = -K_AMPS * angle_rad

        # ----- SATURATION CHECK -----
        sat = False
        if amps_cmd > MAX_AMPS:
            amps_cmd = MAX_AMPS
            sat = True
        elif amps_cmd < -MAX_AMPS:
            amps_cmd = -MAX_AMPS
            sat = True

        # ---- SEND AS TORQUE ----
        odrive.set_torque(amps_cmd)

        # ---- DIAGNOSTICS ----
        if sat:
            print(f"[SAT] angle={angle_rad:.3f} rad | amps={amps_cmd:.2f}A | LIMIT HIT!")
        else:
            print(f"angle={angle_rad:.3f} rad | amps={amps_cmd:.2f}A")

        await asyncio.sleep(LOOP_DT)


############################
# MAIN APP
############################

async def main():
    odrive = pyodrivecan.ODriveCAN(0)

    odrive.clear_errors(identify=False)
    await asyncio.sleep(1)
    odrive.initCanBus()

    print("[INFO] Put pendulum straight down to record zero.")
    await asyncio.sleep(2)

    await asyncio.gather(
        odrive.loop(),
        controller(odrive)
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[STOP] Emergency stop!")
        pyodrivecan.ODriveCAN(0).estop()

