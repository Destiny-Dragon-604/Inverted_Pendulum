# Perform inverted pendulum PID control

import time
import threading
import can 
import struct
from simple_pid import PID
from AS5048AEncoder import AS5048AEncoder  # Import the encoder class
from InvertedPendulumPID import InvertedPendulumPID
from InvPendDatabase import InvPendDatabase

# Define a shared variable or event that threads can check
running = threading.Event()
running.set()  # Set it to true initially

# Create an instance of the database
invPendPIDDatabase = InvPendDatabase("InvPendIMUatabase.db")

# Add a new trial to the database
trial_id = invPendPIDDatabase.add_trial()
print(f"Added Trial with ID: {trial_id}")

# CAN initialization
node_id = 0
bus = can.interface.Bus(interface="socketcan", channel="can0")  # Updated for deprecation warning

while not (bus.recv(timeout=0) is None): 
    pass

bus.send(can.Message(arbitration_id=(node_id << 5 | 0x07), data=struct.pack('<I', 8), is_extended_id=False))

start_time = time.time()
timeout = 10  # seconds

for msg in bus:
    if time.time() - start_time > timeout:
        print("Timeout waiting for the expected CAN message.")
        break

    if msg.arbitration_id == (node_id << 5 | 0x01):
        error, state, result, traj_done = struct.unpack('<IBBB', bytes(msg.data[:7]))
        if state == 8:
            break

# Initialize instance of AS5048AEncoder
encoder = AS5048AEncoder()

# Initialize instance of InvertedPendulumPID
p_constant = -0.01
i_constant = 0
d_constant = 0

# Torque upper and lower limits (50% of O-Drive)
pid_upper_limit = 0.63
pid_lower_limit = -0.63
target_angle = 0

pid = InvertedPendulumPID(p_constant, i_constant, d_constant, target_angle, pid_lower_limit, pid_upper_limit)

# Pause to remove lock
print("\nRemove lock mechanism. Time Remaining:\n")
time_remaining = 5

while time_remaining != 0:
    print(f"{time_remaining} s")
    time_remaining -= 1
    time.sleep(1)

# Global variables
odrive_error_detected = False
initialTime = time.time()

# Threads
read_angle_thread = threading.Thread(target=pid.read_angle_thread, args=(running,))
set_motor_torque_thread = threading.Thread(target=pid.set_torque_thread, args=(node_id, bus, running))
add_data_to_database = threading.Thread(target=pid.add_data_to_database, args=('InvPendIMUatabase.db', invPendPIDDatabase, initialTime, trial_id, running))

# Initiate threads
print("\nPID Active")
read_angle_thread.start()
set_motor_torque_thread.start()
add_data_to_database.start()

# Shutdown can bus upon ctrl+c
try:
    while True:
        time.sleep(0.001)

except KeyboardInterrupt:
    # Signal threads to stop
    running.clear()

    print("\nThreads cleared!")

finally:
    # Wait for the threads to stop
    read_angle_thread.join()
    set_motor_torque_thread.join()
    add_data_to_database.join()

    bus.send(can.Message(arbitration_id=(node_id << 5 | 0x0E), data=struct.pack('<f', 0.0), is_extended_id=False))
    print(f"Successfully set ODrive {node_id} to 0 [Nm]")

    # Shutdown the bus in the finally block to ensure it's always executed
    bus.shutdown()
    print("\nProgram terminated gracefully.")

