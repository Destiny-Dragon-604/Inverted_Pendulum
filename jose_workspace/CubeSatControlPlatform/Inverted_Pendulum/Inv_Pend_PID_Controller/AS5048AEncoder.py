import smbus
import time

class AS5048AEncoder:
    def __init__(self, bus_number=1, address=0x40):
        # Create an SMBus instance
        self.bus = smbus.SMBus(bus_number)
        self.address = address
        self.angle_register = 0xFE
        
        # Initialize the last angle, total rotations, and rest position
        self.last_angle_raw = self.read_raw_angle()
        self.total_rotations = 0
        self.rest_position = self.raw_to_degrees(self.last_angle_raw)
        self.error_tolerance = 0.02  # Error tolerance for recalibration
        self.stable_duration = 3  # Seconds to consider the encoder at rest
        self.stable_count = 0  # Counter for stable readings
        self.reading_interval = 0.01  # Time interval between readings
        self.stable_count_threshold = self.stable_duration / self.reading_interval  # Stable reading threshold

    def read_raw_angle(self):
        data = self.bus.read_i2c_block_data(self.address, self.angle_register, 2)
        return data[0] * 256 + data[1]

    def raw_to_degrees(self, raw_angle):
        # Adjust raw angle to make +x (right direction) as 0 degrees
        degrees = ((raw_angle / 16383.0) * 360) / 8
        return round((degrees + 90) % 360, 2)

    def read_angle(self):
        # Read the current raw angle
        current_angle_raw = self.read_raw_angle()
        
        # Convert to degrees
        current_angle = self.raw_to_degrees(current_angle_raw)
        
        # Calculate change in angle
        change_in_angle_raw = current_angle_raw - self.last_angle_raw
        
        # Update the last angle
        self.last_angle_raw = current_angle_raw
        
        # Check for wrap around (if the encoder passes through 0)
        if change_in_angle_raw > 8191:  # more than half of 16383
            self.total_rotations += 1
        elif change_in_angle_raw < -8191:
            self.total_rotations -= 1
        
        # Calculate the total angle considering the rotations
        total_angle = round((self.total_rotations * 180 + current_angle - self.rest_position) % 360, 2)
        
        # Check if the angle is within the error tolerance
        if abs(total_angle) < self.error_tolerance:
            self.stable_count += 1
            if self.stable_count >= self.stable_count_threshold:
                self.rest_position = current_angle  # Recalibrate to new rest position
                self.stable_count = 0  # Reset stable count after recalibration
                print(f"Recalibrated rest position to: {self.rest_position:.2f} degrees")
            total_angle = 0  # Zero out the angle if within tolerance
        else:
            self.stable_count = 0  # Reset stable count if movement is detected
        
        return total_angle

    def close(self):
        self.bus.close()

