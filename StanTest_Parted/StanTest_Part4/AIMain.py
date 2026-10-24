#!/usr/bin/env python3
# Combined Reaction-Wheel Pendulum Controller (ODrive + I2C Encoder)
# - Reads absolute angle from I2C encoder
# - Unwraps to continuous turns and normalizes upright to 0 rad
# - Runs PD + wheel-damping controller in ODrive torque mode

import asyncio
import math
from datetime import datetime
import smbus
import time as pytime

# =========================
# ====== USER CONFIG ======
# =========================

# --- Encoder (I2C) ---
ENC_I2C_ADDR = 0x40      # <-- confirm for your sensor
ENC_REG_RAW  = 0xFE      # <-- register that returns 2 bytes of angle
ENC_BUS_NUM  = 1         # I2C bus number (RPi usually 1)

# If your odrive.velocity is in turns/s, set to "turns_s"; else "rad_s"
VELOCITY_UNITS = "rad_s"   # "rad_s" or "turns_s"

# ODrive torque command units
# If your firmware is set to torque control in Nm, use "Nm".
# If it wants current, use "A" and provide KT_NM_PER_A below.
TORQUE_UNITS = "Nm"        # "Nm" or "A"
KT_NM_PER_A  = 0.060       # motor torque constant (Nm/A), only used if TORQUE_UNITS == "A"

# Axis / Bus selection
ODRIVE_BUS_ID = 0          # CAN interface index for pyodrivecan.ODriveCAN(…)
GEAR_RATIO    = 1        # motor:pendulum angle ratio, 1.0 if direct

# Limits & safety
TORQUE_LIMIT_NM = 10.0      # Saturation limit in Nm (or in A if TORQUE_UNITS=="A", after conversion below)
FALLBACK_ANGLE = 1000      # rad (~31°); estop if exceeded
RUN_SECONDS = 600.0        # how long to try balancing



# Loop timing
CONTROL_DT = 0.001  # seconds (≈1 kHz)

# =========================
# ===== IMPLEMENTATION ====
# =========================

# Global bus instance (avoid recreating per read)
bus = smbus.SMBus(ENC_BUS_NUM)

def read_raw_angle_turns():
    """
    Read raw angle from encoder and return turns in [0, 1).

    Adjust this to match your encoder's byte format.
    Current example follows the user's prior code (8-bit + 6-bit fraction feel).
    If you actually have a 12-bit sensor, use the commented 12-bit path.
    """
    data = bus.read_i2c_block_data(ENC_I2C_ADDR, ENC_REG_RAW, 2)
    hi, lo = data[0], data[1]

    # ---- OPTION A: user's previous split (keep if correct for your sensor) ----
    turns = (hi / 255.0) + (lo / (64.0 * 255.0))  # ~[0, ~1)
    turns = turns % 1.0

    # ---- OPTION B: common 12-bit angle across two bytes (uncomment if needed) ----
    # angle_counts = ((hi << 4) | (lo >> 4)) & 0x0FFF   # 0..4095
    # turns = angle_counts / 4096.0

    return turns

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def to_drive_units(tau_nm):
    """
    Convert Nm to the units expected by ODrive.
    If TORQUE_UNITS == "Nm": pass-through.
    If TORQUE_UNITS == "A": convert via Kt (Nm/A) -> current [A].
    """
    if TORQUE_UNITS == "Nm":
        return tau_nm
    elif TORQUE_UNITS == "A":
        return tau_nm / KT_NM_PER_A
    else:
        raise ValueError("TORQUE_UNITS must be 'Nm' or 'A'")

def from_velocity_units(odrive_vel_raw):
    """
    Convert odrive.velocity to rad/s if needed.
    """
    if VELOCITY_UNITS == "rad_s":
        return float(odrive_vel_raw)
    elif VELOCITY_UNITS == "turns_s":
        return float(odrive_vel_raw) * 2.0 * math.pi
    else:
        raise ValueError("VELOCITY_UNITS must be 'rad_s' or 'turns_s'")

async def controller(odrive):
    """
    Main control loop:
      - unwrap encoder turns, normalize upright to 0
      - estimate theta_dot
      - compute tau_cmd = -Kp*theta - Kd*theta_dot - Kw*wheel_rate
      - send torque/current to ODrive
    """
    # Control gains (initial guesses)
    Kp = -120      # Nm/rad (or A/rad if TORQUE_UNITS=="A")
    Kd = -20       # Nm·s/rad
    Kw = 10     # Nm/(rad/s) wheel-rate damping
    LPF_ALPHA = 0  # 0..1, low-pass for theta_dot (smaller = more smoothing)
    
    await asyncio.sleep(1.0)
    odrive.set_controller_mode("torque_control")

    # Establish zero reference near upright
    print("Hold pendulum near upright to set zero…")
    await asyncio.sleep(1.0)
    rest_turns = read_raw_angle_turns()
    prev_turns = rest_turns
    wrap_count = 0

    # Rate estimation state
    theta_prev = 0.0
    theta_dot = 0.0

    # Torque saturation in *drive* units
    torque_limit_drive = to_drive_units(TORQUE_LIMIT_NM)

    # Timing
    loop = asyncio.get_event_loop()
    t_prev = loop.time()
    stop_at = pytime.time() + RUN_SECONDS

    print("Starting control loop.")
    while pytime.time() < stop_at:
        # ----- Encoder read + unwrap to continuous turns -----
        raw_turns = read_raw_angle_turns()
        dturns = raw_turns - prev_turns
        if dturns < -0.5:
            wrap_count += 1
        elif dturns > 0.5:
            wrap_count -= 1
        cont_turns = raw_turns + wrap_count
        prev_turns = raw_turns

        # Normalize so upright zero = rest_turns
        norm_turns = cont_turns - rest_turns

        # turns -> radians at the pendulum joint
        theta = (2.0 * math.pi / GEAR_RATIO) * norm_turns

        # ----- dt -----
        t_now = loop.time()
        dt = max(1e-4, t_now - t_prev)
        t_prev = t_now

        # ----- theta_dot finite-difference + LPF -----
        raw_theta_dot = (theta - theta_prev) / dt
        theta_dot = (1.0 - LPF_ALPHA) * theta_dot + LPF_ALPHA * raw_theta_dot
        theta_prev = theta

        # ----- wheel rate from ODrive -----
        try:
            wheel_rate = from_velocity_units(odrive.velocity)  # rad/s
        except Exception:
            wheel_rate = 0.0

        # ----- Control law -----
        Kp = Kp-abs(theta)
        Kd = Kd-abs(theta)
        tau_cmd_nm = -(Kp * theta + Kd * theta_dot + Kw * wheel_rate)

        # Saturate (in Nm), then convert to drive units
        tau_cmd_nm = clamp(tau_cmd_nm, -TORQUE_LIMIT_NM, TORQUE_LIMIT_NM)
        drive_cmd = to_drive_units(tau_cmd_nm)
        drive_cmd = clamp(drive_cmd, -torque_limit_drive, torque_limit_drive)

        # Safety: bail if we’re too far from upright
        if abs(theta) > FALLBACK_ANGLE:
            print(f"[SAFETY] Fall detected | θ={theta:+.3f} rad > {FALLBACK_ANGLE:.2f} rad. E-stop.")
            try:
                odrive.estop()
            finally:
                break

        # Send command
        odrive.set_torque(drive_cmd)

        # Telemetry
        print(
            f"θ={theta:+.3f} rad | θ̇={theta_dot:+.3f} rad/s | ϕ̇={wheel_rate:+.2f} rad/s | "
            f"τ={tau_cmd_nm:+.3f} Nm ({drive_cmd:+.3f} {TORQUE_UNITS}) | dt={dt*1e3:.1f} ms"
        )

        # ~1 kHz loop
        await asyncio.sleep(CONTROL_DT)

async def main():
    import pyodrivecan  # import locally so the file can still be linted without it
    odrive = pyodrivecan.ODriveCAN(ODRIVE_BUS_ID)

    # Clear errors and init bus
    odrive.clear_errors(identify=False)
    print("Cleared ODrive errors.")
    await asyncio.sleep(0.2)
    odrive.initCanBus()

    # Optional: see if position is available (might not be in torque mode)
    try:
        print(f"ODrive position (may be unused in torque mode): {odrive.position}")
    except Exception:
        pass

    # Run drive background loop + controller concurrently
    await asyncio.gather(
        odrive.loop(),
        controller(odrive)
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("KeyboardInterrupt: attempting estop…")
        try:
            # If we can still reach the drive object, stop it
            import pyodrivecan
            pyodrivecan.ODriveCAN(ODRIVE_BUS_ID).estop()
        except Exception:
            pass
