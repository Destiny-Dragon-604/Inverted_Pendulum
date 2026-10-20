import pyodrivecan
import asyncio
import math

async def controller(odrive, trial_id, tuning_factor=0.0003):
    try:
        print(f"Using trial_id: {trial_id}")

        # Loop 10 times
        for i in range(4):
            # Calculate the adjusted delay using a logarithmic function
            delay = max(0.1, 0.5 - tuning_factor * math.log(i + 1))

            # Set velocity to -2000 counts per second
            odrive.set_velocity(-2000)
            await asyncio.sleep(delay)  # Wait for the dynamically calculated delay

            # Set velocity to 5000 counts per second
            odrive.set_velocity(5000)
            await asyncio.sleep(delay)  # Wait for the dynamically calculated delay
       
        for i in range(1):
            # Fixed delay
            delay = 0.648

            # Set velocity to -5000 counts per second
            odrive.set_velocity(-5000)
            await asyncio.sleep(delay)  # Wait for the dynamically calculated delay

            # Set velocity to 5000 counts per second
            odrive.set_velocity(5000)
            await asyncio.sleep(delay)  # Wait for the dynamically calculated delay

        # Set velocity to 0 counts per second after the loop ends
        odrive.set_velocity(0)

    except Exception as e:
        print(f"An error occurred during control: {e}")

    finally:
        # Reset the ODrive to a safe state
        print("Resetting ODrive and clearing errors...")
        odrive.clear_errors()
        odrive.set_velocity(0)
        print("ODrive reset successfully.")
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
        odrive.clear_errors()  # Clear any errors to reset the state
        print("Cleanup complete.")

if __name__ == "__main__":  # Corrected this line
    # Run the main function
    asyncio.run(main())

