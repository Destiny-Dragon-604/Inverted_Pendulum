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
TARGET_ANGLE = 180
  # The target angle to reach
MAX_VELOCITY = 30000  # Maximum velocity limit (positive)
MIN_VELOCITY = -30000 # Minimum velocity limit (negative)
Kp = 5 # Proportional gain for the controller
error_tolerance = 0.1  # Tolerance for stopping the motor near the target angle (smaller for precision)

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

# Controller to reach the target angle
async def controller(odrive, target_angle):
    global angle
    try:
        while True:
            error = target_angle - angle

            # Debug: Print current angle and error
            print(f"Current Angle: {angle}, Error: {error}")
            
            # Proportional control to set velocity based on error
            velocity = Kp * error
            
            # Limit velocity to within allowed bounds
            if velocity > MAX_VELOCITY:
                velocity = MAX_VELOCITY
            elif velocity < MIN_VELOCITY:
                velocity = MIN_VELOCITY

            # Stop the motor if close to the target angle
            if abs(error) < error_tolerance:
                velocity = 0
                if odrive.canBus is not None:
                    odrive.set_velocity(velocity)
                print("Target reached, stopping motor.")
                break
            
            # Set velocity on ODrive
            try:
                if odrive.canBus is not None:
                    odrive.set_velocity(velocity)
                else:
                    raise Exception("CAN bus is not initialized")
            except OSError as e:
                if e.errno == 105:  # Buffer full error  
                    if odrive.canBus is not None:
                        odrive.set_velocity(velocity)
            
            # Read the current angle and update at regular intervals
            read_angle()
            await asyncio.sleep(0.01)
            
    finally:
        if odrive.canBus is not None:
            odrive.set_velocity(0)
        odrive.running = False
        


# Main function to initialize the ODrive and run the control loop
async def main():
    trial_id = 4
    odrive = pyodrivecan.ODriveCAN(0)

    try:
        # Ensure the CAN Bus is properly initialized
        odrive.initCanBus()
        if odrive.canBus is None:
            raise Exception("CAN bus initialization failed.")
        
        odrive.running = True  # Set the motor to running
        
        # Run the controller to reach the target angle
        await controller(odrive, TARGET_ANGLE)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Shutdown the CAN bus and stop the motor
        if odrive.canBus is not None:
            try:
                odrive.set_velocity(0)
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
