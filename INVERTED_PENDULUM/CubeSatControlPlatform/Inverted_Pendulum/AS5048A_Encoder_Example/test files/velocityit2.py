import pyodrivecan
import asyncio
import math


async def controller(odrive, trial_id, tuning_factor=0.1):
    try:
        print(f"Using trial_id: {trial_id}")

        # Loop 10 times
        for i in range(4):
            # Set velocity to -5000 counts per second
            odrive.set_velocity(1000)
            await asyncio.sleep(1.5)  # Wait for the dynamically calculated delay

            # Set velocity to 5000 counts per second
            odrive.set_velocity(-1000)
            await asyncio.sleep(1.5)  # Wait for the dynamically calculated delay
        

        # Set velocity to 0 counts per second after the loop ends
        odrive.set_velocity(0)
    finally:
        # Set running flag to False to stop further operations
        odrive.running = False


async def main():
    # Define the trial ID
    trial_id = 4  # Example, replace with dynamic fetching if needed

    # Create ODriveCAN object with node_id = 0
    odrive = pyodrivecan.ODriveCAN(0)

    try:
        # Initialize the ODrive object
        odrive.initCanBus()

        # Set running flag to True to start continuous movement
        odrive.running = True

        # Run the controller function with a trial_id
        await controller(odrive, trial_id)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Implement any necessary cleanup here
        print("Cleanup (if necessary)")
        # Ensure the motor stops if an error occurs
        odrive.set_velocity(0)

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
