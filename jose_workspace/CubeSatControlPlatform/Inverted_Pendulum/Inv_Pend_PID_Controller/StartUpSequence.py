import pyodrivecan
import asyncio
import math

class StartUpSequence:
    def __init__(self, node_id, tuning_factor=0.0003):
        self.node_id = node_id
        self.odrive = pyodrivecan.ODriveCAN(node_id)
        self.tuning_factor = tuning_factor

    async def controller(self, trial_id):
        """
        Controls the startup sequence to accelerate the flywheel.
        """
        try:
            print(f"Using trial_id: {trial_id}")

            # Loop to accelerate flywheel in stages
            for i in range(4):
                # Calculate the adjusted delay using a logarithmic function
                delay = max(0.1, 0.5 - self.tuning_factor * math.log(i + 1))

                # Set velocity to -2000 counts per second
                print(f"Setting velocity to -2000 counts per second with delay {delay:.2f}s")
                self.odrive.set_velocity(-2000)
                await asyncio.sleep(delay)  # Wait for the dynamically calculated delay

                # Set velocity to 5000 counts per second
                print(f"Setting velocity to 5000 counts per second with delay {delay:.2f}s")
                self.odrive.set_velocity(5000)
                await asyncio.sleep(delay)  # Wait for the dynamically calculated delay

            # Additional loop for specific delay
            for i in range(1):
                # Fixed delay
                delay = 0.648
                print(f"Setting velocity to -5000 counts per second with fixed delay {delay:.2f}s")
                self.odrive.set_velocity(-5000)
                await asyncio.sleep(delay)

                print(f"Setting velocity to 5000 counts per second with fixed delay {delay:.2f}s")
                self.odrive.set_velocity(5000)
                await asyncio.sleep(delay)

            # Set velocity to 0 counts per second after the loop ends
            print("Setting velocity to 0 counts per second")
            self.odrive.set_velocity(0)

        except Exception as e:
            print(f"An error occurred during control: {e}")

        finally:
            # Reset the ODrive to a safe state
            print("Resetting ODrive and clearing errors...")
            self.odrive.clear_errors()
            self.odrive.set_velocity(0)
            print("ODrive reset successfully.")
            # Set running flag to False to stop further operations
            self.odrive.running = False

    async def run(self, trial_id):
        """
        Runs the startup sequence.
        """
        try:
            # Initialize the ODrive object
            print("Initializing CAN Bus...")
            self.odrive.initCanBus()

            # Clear any existing errors
            print("Clearing ODrive errors...")
            self.odrive.clear_errors()

            # Set running flag to True to start continuous movement
            print("Starting ODrive startup sequence...")
            self.odrive.running = True

            # Run the controller function with a trial_id
            await self.controller(trial_id)

        except Exception as e:
            print(f"An error occurred: {e}")

        finally:
            # Ensure the motor stops if an error occurs
            print("Cleanup (if necessary)")
            self.odrive.set_velocity(0)
            self.odrive.clear_errors()  # Clear any errors to reset the state
            print("Cleanup complete.")

# Main function to run the startup sequence
def main():
    # Define the trial ID and node ID
    trial_id = "trial_001"  # Example trial ID
    node_id = 1   # Example node ID for ODrive

    # Create an instance of StartUpSequence
    startup_sequence = StartUpSequence(node_id)

    # Run the startup sequence using asyncio
    asyncio.run(startup_sequence.run(trial_id))

if __name__ == "__main__":
    main()

