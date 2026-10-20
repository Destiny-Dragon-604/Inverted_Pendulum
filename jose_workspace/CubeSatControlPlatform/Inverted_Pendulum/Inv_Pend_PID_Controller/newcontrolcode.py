import pyodrivecan
import asyncio
import numpy as np
import smbus
import time

# Encoder setup
bus = smbus.SMBus(1)
AS5048A_ADDR = 0x40
AS5048A_ANGLE_REG = 0xFE

def read_raw_angle():
    data = bus.read_i2c_block_data(AS5048A_ADDR, AS5048A_ANGLE_REG, 2)
    return data[0] * 256 + data[1]

def raw_to_degrees(raw_angle):
    degrees = ((raw_angle / 16383.0) * 360) / 8
    return round((degrees + 90) % 360, 2)

# Initialize variables for calibration and control
last_angle_raw = read_raw_angle()
total_rotations = 0
rest_position = raw_to_degrees(last_angle_raw)
error_tolerance = 0.02
stable_duration = 3
stable_count = 0
reading_interval = 0.01
stable_count_threshold = stable_duration / reading_interval

def read_angle():
    global last_angle_raw, total_rotations, rest_position, stable_count

    current_angle_raw = read_raw_angle()
    current_angle = raw_to_degrees(current_angle_raw)
    change_in_angle_raw = current_angle_raw - last_angle_raw
    last_angle_raw = current_angle_raw

    if change_in_angle_raw > 8191:
        total_rotations += 1
    elif change_in_angle_raw < -8191:
        total_rotations -= 1

    total_angle = round((total_rotations * 180 + current_angle - rest_position) % 360, 2)

    if abs(total_angle) < error_tolerance:
        stable_count += 1
        if stable_count >= stable_count_threshold:
            rest_position = current_angle
            stable_count = 0
            print(f"Recalibrated rest position to: {rest_position:.2f} degrees")
        total_angle = 0
    else:
        stable_count = 0

    return total_angle

def controller_lqr(p_angle, p_vel, m_vel):
    # LQR gains
    Kp = 40
    Kd = 7
    Km = 0.3

    # Control law
    control_signal = Kp * p_angle + Kd * p_vel + Km * m_vel
    control_signal = max(min(control_signal, 12), -12)  # Apply voltage limits
    return control_signal

async def controller(odrive):
    try:
        last_angle = read_angle()
        last_time = time.time()

        while odrive.running:
            current_angle = read_angle()
            current_time = time.time()
            theta = current_angle
            theta_dot = (current_angle - last_angle) / (current_time - last_time)

            last_angle = current_angle
            last_time = current_time

            motor_velocity = odrive.get_motor_velocity()  # Assuming this is the correct synchronous method

            # Decide control action
            if abs(theta) < 0.5:  # Stabilize near upright
                control_signal = controller_lqr(theta, theta_dot, motor_velocity)
            else:  # Swing-up strategy
                control_signal = -np.sign(theta_dot) * 0.4  # Adjust scaling as needed

            print(f"Angle: {theta:.2f}, Angular Velocity: {theta_dot:.2f}, Control Signal: {control_signal:.2f}")

            # Send torque control command via CAN
            odrive.send_torque_command(control_signal)  # Assuming this is the correct synchronous method

            await asyncio.sleep(0.025)  # Run control loop every ~25ms

    except Exception as e:
        print(f"An error occurred during control: {e}")
    finally:
        print("Resetting ODrive and clearing errors...")
        odrive.clear_errors()  # Assuming this is the correct synchronous method
        odrive.running = False

async def main():
    odrive = pyodrivecan.ODriveCAN(0)

    try:
        odrive.init_can()  # Assuming this is the correct synchronous initialization method
        odrive.running = True  # Set the running state
        await controller(odrive)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        odrive.clear_errors()  # Assuming this is the correct synchronous method
        print("Cleanup complete.")

if __name__ == "__main__":
    asyncio.run(main())

