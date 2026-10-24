import smbus
import time




def read_raw_angle(): # Function to read raw angle from the encoder
    data = bus.read_i2c_block_data(AS5048A_ADDR, AS5048A_ANGLE_REG, 2)
    data = data[0] / 255 + data[1]/64/255
    data = data
    return data

def normalize(curr_position,rest_position): #Normalize to rest position
    current_angle = read_raw_angle()
    normd = curr_position - rest_position
    return normd

###### Initilize #####
bus = smbus.SMBus(1) # Create an SMBus instance
AS5048A_ADDR = 0x40 # AS5048A default address
AS5048A_ANGLE_REG = 0xFE # AS5048A Register
reading_interval = 0.1  # Time interval between readings
rest_pos = read_raw_angle()
val = 0
hturns = 0

####### Run #########
while True:
    prev_val = val
    val = read_raw_angle()
    diff = val - prev_val
    if diff < -.5:  # This is to wrap the position, it is based on the fact that you have a 1:2 gear ratio. I would change this to 1:1
        hturns += 1
    elif diff> 0.5:
        hturns -= 1
    position = val + hturns
    position = normalize(position,rest_pos)
    print(f'{position} \t {bus.read_i2c_block_data(AS5048A_ADDR, AS5048A_ANGLE_REG, 2)}')
    time.sleep(reading_interval)
