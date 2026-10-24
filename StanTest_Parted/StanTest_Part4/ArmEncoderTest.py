import smbus
import time

# Create an SMBus instance
bus = smbus.SMBus(1)

# AS5048A default address
AS5048A_ADDR = 0x40

# AS5048A Register
AS5048A_ANGLE_REG = 0xFE

class Wrap: # This dumbass sensor has #N_g poles, just use 1:1 gear ratios and comment this out next time 
    def __init__(self):
        self.prev_val = None
        self.turns = 0
        self.angle = 0.0
    def update(self,val):
        if self.prev_val is not None: 
            diff = val-self.prev_val
            print(f'{diff}',flush=True)
            if diff<-0.5: # Forward wrapping
                self.turns += 1
            elif diff >0.5: # Backwards jump 
                self.turns -= 1
            print(selfturn)
        self.prev_val = val
        
        self.angle = self.turns + val

        
        return self.angle


# Function to read raw angle from the encoder
def read_raw_angle():
    data = bus.read_i2c_block_data(AS5048A_ADDR, AS5048A_ANGLE_REG, 2)
    data = data[0] / 255 + data[1]/64/255
    data = data
    return data
    
def wrapped_raw():
    det = Wrap()
    data = det.update(read_raw_angle())
    return data

# Function to convert raw angle to degrees
def Degrees():
    degrees = wrapped_raw() * 360
    return degrees 

def read_angle(rest_position,N_g):
    current_angle = Degrees() 

    # Calculate the total angle considering the rotations
    total_angle = round((current_angle - rest_position), 2)
    
    return total_angle
    
############### Main script ################

print("Initializing rest position...")

# Initialize the last angle, total rotations, and rest position
N_g = 2 # gear ratio
reading_interval = 0.03  # Time interval between readings
time.sleep(2)  # Give some time to ensure the encoder is at rest
rest_position = Degrees()  # Set the initial rest position
print("Rest position initialized at {:.2f} degrees.".format(rest_position))

while True:
    rawData = read_raw_angle()
    Deg = Degrees()
    angle = read_angle(rest_position,N_g)
    print(f"Raw: {round(rawData,4)} ; \t DegFunc {round(Deg,4)}; \t Angle: {round(angle,4)}")
    time.sleep(reading_interval)

