import asyncio
from StartUpSequence import StartUpSequence 

async def main():
    # Define the ODrive node ID
    node_id = 1  # Replace with the actual node ID of your ODrive

    # Define a trial ID
    trial_id = "trial_001"  # Replace with your specific trial ID

    # Create an instance of StartUpSequence with the desired parameters
    startup_sequence = StartUpSequence(node_id=node_id)

    # Run the startup sequence
    await startup_sequence.run(trial_id)

if __name__ == "__main__":
    # Start the asyncio event loop and run the main coroutine
    asyncio.run(main())

