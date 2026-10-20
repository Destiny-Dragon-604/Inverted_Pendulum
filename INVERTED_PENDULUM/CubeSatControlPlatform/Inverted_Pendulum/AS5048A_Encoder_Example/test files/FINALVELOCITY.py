import smbus
import time
import pyodrivecan
import asyncio
import math

# Constants for the pendulum setup
m = 0.5  # Mass of the pendulum in kg
g = 9.81  # Acceleration due to gravity in m/s^2
l = 0.5  # Length of the pendulum in meters

# Create an SMBus instance for AS5048A encoder communication
bus = smbus.SMBus(1)

# AS5048A default address and angle register
AS5048A_ADDR = 0x40
AS5048A_ANGLE_REG = 0xFE

# Global variables
angle = 0  # Store the current angle
TARGET_ANGLE = 0  # The target angle (vertical position)
MAX_TORQUE = 8  # Maximum torque limit (positive)
MIN_TORQUE = -8  # Minimum torque limit (negative)
Kp = 0.5  # Proportional gain for the torque controller
Ki = 0.01  # Integral gain for the torque controller
Kd = 0.1  # Derivative gain for the torque controller
error_tolerance = 0.5  # Tolerance for stopping the motor near the target angle
deadband = 1.0  # Deadband where no torque is applied near the target
integral_error = 0  # Accumulator for integral term
previous_error = 0  # To track error for derivative term
integral_max = 2  # Cap on the integral term to avoid windup

# Function to read raw angle from the AS5048A encoder
def read_raw_angle():
    data = bus.read_i2c_block_data(AS5048A_ADDR, AS5048A_ANGLE_REG, 2)
    return data[0] * 256 + data[1]

# Function to convert the raw angle into degrees
def raw_to_degrees(raw_angle):
    degrees = ((raw_angle / 16383.0) * 360) / 8
    return round((degrees + 90) % 360, 2)

# Function to calculate the current angle
def read_angle():
    global last_angle_raw, total_rotations, rest_position, angle
    
    current_angle_raw = read_raw_angle()
    current_angle = raw_to_degrees(current_angle_raw)
    
    change_in_angle_raw = current_angle_raw - last_angle_raw
    last_angle_raw = current_angle_raw
    
    if change_in_angle_raw > 8191:  # Large positive jump (wrap around)
        total_rotations += 1
    elif change_in_angle_raw < -8191:  # Large negative jump (wrap around)
        total_rotations -= 1

    total_angle = round((total_rotations * 180 + current_angle - rest_position) % 360, 2)
    angle = total_angle
    return total_angle

# Torque controller to stabilize the pendulum using mglθ = torque
async def torque_controller(odrive):
    global angle, integral_error, previous_error
    try:
        while True:
            theta = math.radians(angle)  # Convert angle to radians
            torque_gravity = m * g * l * math.sin(theta)  # mglθ

            # Error is the difference between desired torque (0) and torque due to gravity
            error = -torque_gravity
            derivative_error = (error - previous_error) / 0.01  # Calculate the derivative (change in error)
            previous_error = error  # Update previous error for the next loop

            # Accumulate the error for the integral term, and clamp it
            integral_error += error * 0.01  # The 0.01 comes from the loop's time interval (10ms)
            integral_error = max(-integral_max, min(integral_max, integral_error))  # Cap the integral

            # Apply deadband logic: If error is small, don't apply torque
            if abs(error) < deadband:
                torque = 0
            else:
                # PID control to calculate the torque based on error
                torque = Kp * error + Ki * integral_error + Kd * derivative_error

                # Limit torque to within allowed bounds
                torque = max(MIN_TORQUE, min(MAX_TORQUE, torque))

            print(f"Current Angle: {angle}, Error: {error}, Integral Error: {integral_error}, Derivative Error: {derivative_error}, Torque: {torque}")

            if abs(error) < error_tolerance:
                torque = 0
                if odrive.canBus is not None:
                    odrive.set_torque(torque)
                print("Pendulum balanced, stopping motor.")
                break
            
            if odrive.canBus is not None:
                odrive.set_torque(torque)

            read_angle()
            await asyncio.sleep(0.01)

    finally:
        if odrive.canBus is not None:
            odrive.set_torque(0)
        odrive.running = False

# Main function to initialize the ODrive, and start the torque controller
async def main():
    odrive = pyodrivecan.ODriveCAN(0)

    try:
        odrive.initCanBus()
        if odrive.canBus is None:
            raise Exception("CAN bus initialization failed.")
        
        await torque_controller(odrive)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if odrive.canBus is not None:
            try:
                odrive.set_torque(0)
                odrive.canBus.shutdown()
            except Exception as shutdown_error:
                print(f"Error during CAN bus shutdown: {shutdown_error}")
        print("CAN BUS Flushed and Shutdown.")

if __name__ == "__main__":
    print("Initializing rest position...")
    time.sleep(2)
    rest_position = raw_to_degrees(read_raw_angle())
    asyncio.run(main())
