
import time
import threading
import can
import struct
from TorqueReactionTestThreads import TorqueReactionTestThreads
from TorqueReactionTestDatabase import TorqueReactionTestDatabase

# Global flag for running state
running = threading.Event()
running.set()

# Set up database for logging
torque_reaction_test_database = TorqueReactionTestDatabase("TorqueReactionTestDatabase.db")
trial_id = torque_reaction_test_database.add_trial()
print(f"Added Trial with ID: {trial_id}")

# CAN setup
node_id = 0
bus = can.interface.Bus("can0", interface="socketcan")

# Clear CAN buffer
while not (bus.recv(timeout=0) is None): pass

# Put motor into closed-loop torque control mode
bus.send(can.Message(arbitration_id=(node_id << 5 | 0x07), data=struct.pack('<I', 8), is_extended_id=False))

# Wait for the motor to enter the correct state
start_time = time.time()
timeout = 10  # seconds

for msg in bus:
    if time.time() - start_time > timeout:
        print("Timeout waiting for motor state confirmation.")
        break

    if msg.arbitration_id == (node_id << 5 | 0x01):
        error, state, result, traj_done = struct.unpack('<IBBB', bytes(msg.data[:7]))
        if state == 8:
            print("ODrive entered torque control mode.")
            break

# Pause before starting
print('Pause for 5 s...')
time.sleep(5)

# Torque to apply (Nm)
torque_setpoint = 0.35
initial_time = time.time()

# Threads for sending torque and logging feedback
threads = TorqueReactionTestThreads()

set_motor_torque_thread = threading.Thread(
    target=threads.set_torque_thread,
    args=(node_id, bus, torque_setpoint, initial_time, running)
)

get_torque_estimate_thread = threading.Thread(
    target=threads.get_system_torque_thread,
    args=(node_id, bus, initial_time, running)
)

print("\nTorque control active.")
set_motor_torque_thread.start()
get_torque_estimate_thread.start()

# Graceful shutdown on Ctrl+C
try:
    while True:
        time.sleep(0.001)

except KeyboardInterrupt:
    running.clear()
    print("\nKeyboardInterrupt: Stopping threads...")

finally:
    # Wait for threads to finish
    set_motor_torque_thread.join()
    get_torque_estimate_thread.join()

    # Set motor torque to 0
    bus.send(can.Message(arbitration_id=(node_id << 5 | 0x0E), data=struct.pack('<f', 0.0), is_extended_id=False))
    print(f"Successfully set ODrive {node_id} torque to 0 Nm")

    # Log collected data
    for i in range(len(threads.time_array)):
        torque_reaction_test_database.add_data(
            trial_id,
            (threads.time_array[i], threads.torque_setpoint_array[i], threads.torque_estimate_array[i])
        )
        print("Data saved to database.")

    # Always shut down the bus
    bus.shutdown()
    print("\nProgram terminated gracefully.")
