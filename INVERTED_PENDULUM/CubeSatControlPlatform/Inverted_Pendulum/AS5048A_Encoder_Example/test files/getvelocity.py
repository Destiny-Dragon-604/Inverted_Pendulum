import pyodrivecan
import asyncio

# Example function to retrieve the velocity from the ODrive axis
async def get_velocity(odrive):
    try:
        # Assuming `get_velocity` is part of an axis object like axis0
        velocity = await odrive.axis0.controller.get_velocity()  # Correctly await the coroutine
        print(f"Current velocity: {velocity} counts/s")
        return velocity
    except Exception as e:
        print(f"Error reading velocity: {e}")
        return None

async def main():
    odrive = pyodrivecan.ODriveCAN(0)  # Initialize ODriveCAN object with node_id = 0

    try:
        odrive.initCanBus()  # Initialize the CAN bus communication
        odrive.running = True
        
        # Get the current velocity of the motor
        velocity = await get_velocity(odrive)
        if velocity is not None:
            print(f"Motor velocity: {velocity} counts/s")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if odrive.canBus:
            odrive.canBus.shutdown()  # Ensure CAN bus is properly shut down
        print("CAN Bus shutdown.")

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
