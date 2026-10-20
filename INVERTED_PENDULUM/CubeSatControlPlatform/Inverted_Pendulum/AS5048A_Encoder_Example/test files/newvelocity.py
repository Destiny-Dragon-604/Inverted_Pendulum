import smbus
import time
import pyodrivecan
import asyncio
import math

# Create an SMBus instance for AS5048A encoder communication
bus = smbus.SMBus(1)

# AS5048A default address and angle register
AS5048A_ADDR = 0x40
AS5048A_ANGLE_REG = 0xFE

# Global variables
angle = 0  # Store the current angle
TARGET_ANGLE = 180  # The target angle to reach
MAX_TORQUE = 8  # Maximum torque limit (positive)
MIN_TORQUE = -8  # Minimum torque limit (negative)
Kp = 0.5  # Further Reduced Proportional gain for the torque controller
Ki = 0.01  # Reduced Integral gain for the torque controller
Kd = 0.1  # Derivative gain for the torque controller
error_tolerance = 0.5  # Tolerance for stopping the motor near the target angle
deadband = 1.0  # Deadband where no torque is applied near the target
integral_error = 0  # Accumulator for integral term
previous_error = 0  # To track error for derivative term
integral_max = 2  # More aggressive cap on the integral term to avoid windup
initial_spin_velocity = 25  # Initial velocity for the fast spin
angle_threshold = 90  # Angle after which the controller kicks in
kick_loops = 4  # Number of times to perform the kick
kick_delay = 0.5  # Delay between each velocity switch in seconds

# Function to read raw angle from the AS5048A encoder
def read_raw_angle():
    data = bus.read_i2c_block_data(AS5048A_ADDR, AS5048A_ANGLE_REG, 2)
    return data[0] * 256 + data[1]

# Function to convert the raw angle into degrees
def raw_to_degrees(raw_angle):
    degrees = ((raw_angle / 16383.0) * 360) / 8
    return round((degrees + 90) % 360, 2)

# Initialize the encoder's rest position
last_angle_raw = read_raw_angle()
total_rotations = 0
rest_position = raw_to_degrees(last_angle_raw)

# Function to calculate the current angle
def read_angle():
    global last_angle_raw, total_rotations, rest_position, angle
    
    current_angle_raw = read_raw_angle()
    current_angle = raw_to_degrees(current_angle_raw)
    
    # Handle wrap-around of the encoder value
    change_in_angle_raw = current_angle_raw - last_angle_raw
    last_angle_raw = current_angle_raw
    
    if change_in_angle_raw > 8191:  # Large positive jump (wrap around)
        total_rotations += 1
    elif change_in_angle_raw < -8191:  # Large negative jump (wrap around)
        total_rotations -= 1

    total_angle = round((total_rotations * 180 + current_angle - rest_position) % 360, 2)
    
    angle = total_angle
    return total_angle

# Initial fast spin followed by reverse, repeated for 4 cycles
async def initial_spin(odrive):
    # Perform the kick (spin forward then reverse) for 4 cycles
    for i in range(kick_loops):
        # Spin forward fast
        print(f"Kick {i + 1}: Spinning forward with velocity {initial_spin_velocity}.")
        if odrive.canBus is not None:
            odrive.set_velocity(initial_spin_velocity)
        
        await asyncio.sleep(kick_delay)  # Apply the forward spin for 0.5 seconds

        # Now reverse the spin to transfer momentum
        reverse_spin_velocity = -initial_spin_velocity
        print(f"Kick {i + 1}: Reversing spin with velocity {reverse_spin_velocity}.")
        if odrive.canBus is not None:
            odrive.set_velocity(reverse_spin_velocity)
        
        await asyncio.sleep(kick_delay)  # Apply the reverse spin for 0.5 seconds

    print("Completed initial momentum transfer after 4 kicks.")

# Torque controller to stabilize the pendulum after the initial spin
async def torque_controller(odrive, target_angle):
    global angle, integral_error, previous_error
    try:
        while True:
            error = target_angle - angle
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
                torque = max(MIN_TORQUE, min(MAX_TORQUE, torque))  # Ensure torque stays within [-5, 5]

            # Debug: Print current angle, error, integral error, derivative error, and torque
            print(f"Current Angle: {angle}, Error: {error}, Integral Error: {integral_error}, Derivative Error: {derivative_error}, Torque: {torque}")

            # Stop the motor if close to the target angle
            if abs(error) < error_tolerance:
                torque = 0
                if odrive.canBus is not None:
                    odrive.set_torque(torque)
                print("Target reached, stopping motor.")
                break
            
            # Set torque on ODrive
            if odrive.canBus is not None:
                odrive.set_torque(torque)
            
            # Read the current angle and update at regular intervals
            read_angle()
            await asyncio.sleep(0.01)
            
    finally:
        if odrive.canBus is not None:
            odrive.set_torque(0)
        odrive.running = False

# Main function to initialize the ODrive, run the initial spin, and start the torque controller
async def main():
    odrive = pyodrivecan.ODriveCAN(0)

    try:
        # Ensure the CAN Bus is properly initialized
        odrive.initCanBus()
        if odrive.canBus is None:
            raise Exception("CAN bus initialization failed.")
        
        # Step 1: Apply the initial fast spin (kick) 4 times to gain momentum
        await initial_spin(odrive)
        
        # Step 2: Wait for the pendulum to swing beyond the angle threshold before engaging the controller
        while True:
            read_angle()  # Continuously read the angle
            if angle > angle_threshold:
                print(f"Angle threshold of {angle_threshold} degrees crossed, activating torque controller.")
                break
            await asyncio.sleep(0.01)
        
        # Step 3: Run the torque controller to stabilize at the target angle
        await torque_controller(odrive, TARGET_ANGLE)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Shutdown the CAN bus and stop the motor
        if odrive.canBus is not None:
            try:
                odrive.set_torque(0)
                odrive.canBus.shutdown()
            except Exception as shutdown_error:
                print(f"Error during CAN bus shutdown: {shutdown_error}")
        print("CAN BUS Flushed and Shutdown.")

# Initialization of rest position and running the control loop
if __name__ == "__main__":
    print("Initializing rest position...")
    time.sleep(2)  # Allow time for the encoder to stabilize
    rest_position = raw_to_degrees(read_raw_angle())  # Set the initial rest position
    asyncio.run(main())  # Run the main loop asynchronously
