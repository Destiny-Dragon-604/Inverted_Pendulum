# InvertedPendulumPID class file (InvertedPendulumPID.py)

from simple_pid import PID
import struct
import time
import can
from AS5048AEncoder import AS5048AEncoder  # Import the encoder class
import sqlite3

class InvertedPendulumPID:
    def __init__(self, p, i, d, target_angle, lower_limit, upper_limit):
        # Setting PID constants
        self.p_parameter = p
        self.i_parameter = i
        self.d_parameter = d

        # Setup PID controller
        self.pid = PID(p, i, d, setpoint=target_angle)
        self.pid.output_limits = (lower_limit, upper_limit)  # RPS bounds on motor

        # Initialize the encoder
        self.encoder = AS5048AEncoder()

    # Thread to set motor velocity, CHANGE TO TORQUE CONTROL
    def set_vel_thread(self, node_id, bus, running):
        while running.is_set():
            angle = self.encoder.read_angle()  # Use the encoder's angle reading
            velocity = self.pid(angle)
            bus.send(can.Message(arbitration_id=(node_id << 5 | 0x0d), data=struct.pack('<ff', float(velocity), 0.0), is_extended_id=False))

    # Function to set torque for a specific O-Drive
    def set_torque_thread(self, node_id, bus, running):
        while running.is_set():
            angle = self.encoder.read_angle()  # Use the encoder's angle reading
            torque = self.pid(angle)
            bus.send(can.Message(
                arbitration_id=(node_id << 5 | 0x0E),  # 0x0E: Set_Input_Torque
                data=struct.pack('<f', torque),
                is_extended_id=False
            ))
            time.sleep(0.001)

    # Function to set torque to 0 for a specific O-Drive
    def set_torque_0(self, node_id, bus, running):
        while running.is_set():
            bus.send(can.Message(
                arbitration_id=(node_id << 5 | 0x0E),  # 0x0E: Set_Input_Torque
                data=struct.pack('<f', 0),
                is_extended_id=False
            ))
            time.sleep(0.001)

    # Thread to read in orientation angle from Encoder
    def read_angle_thread(self, running):
        while running.is_set():
            angle = self.encoder.read_angle()
            print(f"Read angle: {angle:.2f} degrees")
            time.sleep(0.001)

    # Prints arm angle and motor velocity
    def get_pos_vel_thread(self, node_id, bus, running):
        while running.is_set():
            for msg in bus:
                if msg.arbitration_id == (node_id << 5 | 0x09):
                    pos, vel = struct.unpack('<ff', bytes(msg.data))
                    print(f"Roll: {self.encoder.read_angle():.2f} degrees, vel: {vel:.3f} [turns/s]")

    def add_data_to_database(self, db_path, db, initial_time, trial_id, running):
        while running.is_set():
            # Inside this loop, a new connection is created on each iteration
            with sqlite3.connect(db_path) as conn:
                encoder_angle = self.encoder.read_angle()
                imuData = [time.time() - initial_time, 0, 0, 0, 0, 0, 0, encoder_angle, 0, 0]
                sql = ''' INSERT INTO imu_data(trial_id, time, raw_accel_x, raw_accel_y, raw_accel_z, raw_gyro_x, raw_gyro_y, raw_gyro_z, angle_x, angle_y, angle_z)
                  VALUES(?, ?, ?, ?, ?, ?, ?, ?, ? ,?, ?) '''
                cursor = conn.cursor()
                cursor.execute(sql, (trial_id, *imuData))
                conn.commit()
                time.sleep(0.001)

