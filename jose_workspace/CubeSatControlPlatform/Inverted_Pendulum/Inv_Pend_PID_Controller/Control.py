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
rest_position = raw_to_degrees(last_angle_raw)  # Initialize to the first reading
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

class ReactionWheelPendulumController:
    def __init__(self, Kp_balance, Kd_balance, tolerance=5.0):
        self.Kp_balance = Kp_balance
        self.Kd_balance = Kd_balance
        self.target_angle = 0  # Target upright angle in degrees
        self.tolerance = tolerance  # Tolerance for 0 to 359 wrap-around

    def balance_control(self, theta, theta_dot):
        # Normalize theta to handle the wrap-around
        if theta > 360 - self.tolerance:  # If angle is close to 359
            theta = 0  # Treat it as 0 degrees

        error = theta - self.target_angle  # Error from target 0 degrees
        control = -self.Kp_balance * error - self.Kd_balance * theta_dot
        control = np.clip(control, -10000, 10000)  # Adjust limits as needed
        return control

    def compute_control(self, theta, theta_dot):
        return self.balance_control(theta, theta_dot)

async def controller(odrive, pendulum_controller):
    try:
        last_angle = read_angle()  # Initialize angle reading
        last_time = time.time()

        while odrive.running:
            current_angle = read_angle()  # Get current angle
            current_time = time.time()
            theta = current_angle
            theta_dot = (current_angle - last_angle) / (current_time - last_time)

            last_angle = current_angle
            last_time = current_time

            control_signal = pendulum_controller.compute_control(theta, theta_dot)
            print(f"Angle: {theta:.2f}, Angular Velocity: {theta_dot:.2f}, Control Signal: {control_signal:.2f}")
            odrive.set_velocity(control_signal)
            await asyncio.sleep(0.01)

    except Exception as e:
        print(f"An error occurred during control: {e}")
    finally:
        print("Resetting ODrive and clearing errors...")
        odrive.clear_errors()
        odrive.set_velocity(0)
        odrive.running = False

async def main():
    trial_id = 4
    odrive = pyodrivecan.ODriveCAN(0)
    pendulum_controller = ReactionWheelPendulumController(Kp_balance=10.0, Kd_balance=2.0)

    try:
        odrive.initCanBus()
        odrive.running = True
        await controller(odrive, pendulum_controller)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        odrive.set_velocity(0)
        odrive.clear_errors()
        print("Cleanup complete.")

if __name__ == "__main__":
    asyncio.run(main())

