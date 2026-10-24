import pyodrivecan
import asyncio
import math
import numpy as np
from datetime import datetime, timedelta
import smbus
import time
import os
# import uvloop # TODO: Implement uvloop for better performance



################## ENCODER ###############
bus = smbus.SMBus(1)
def read_raw_angle(): # TODO: Change to async for better performance
    data = bus.read_i2c_block_data(0x40, 0xFE, 2)
    return data[0] / 255 + data[1]/64/255

################## ODRIVE ################

async def controller(odrive):
    await asyncio.sleep(0)
    #Run for set time delay example runs for 15 seconds.
    odrive.set_controller_mode("torque_control")
    stop_at = datetime.now() + timedelta(seconds=10000)
    
    #### Gains ######
    K1, K2, K3 = 6,3,4
    K1v, K2v, K3v = 6,3,4
    # #### Initilize #####
    rest_pos = read_raw_angle()
    val = 0
    hturns = 0
    odrive.set_torque(10)
    sum_e = 0
    err_last = 0
    loop = asyncio.get_running_loop()
    dt = loop.time()
    I = 0
    Iv = 0
    while datetime.now() < stop_at:
		# ### Encoder ######
        prev_val = val
        val = read_raw_angle()
        diff = val - prev_val
        if diff < -.5:  # This is to wrap the position due to magnetic encoder limits
            hturns += 1
        elif diff> 0.5:
            hturns -= 1
        position = val + hturns
        e = position - rest_pos
        # stuff
        
        sum_e += e*dt
        de = e-err_last
        dt -= loop.time()
        err_last = 0
        ev = odrive.velocity  
        if ev is None:
           ev = 0.0
        I = K2*de/dt + I
        Iv = K2v*ev/dt + Iv
      
        u2 = -(ev*K1v + Iv +K3v*ev/dt)
        u1 = -(e*K1 + I + K3*de/dt) # TODO: Recheck the equation
        odrive.set_torque(u1)
        
        dt = loop.time()
        await asyncio.sleep(0)  # yield, but don't delay


#Set up Node_ID 10 ACTIV NODE ID = 10
odrive = pyodrivecan.ODriveCAN(0)

# Run multiple busses.
async def main():
    odrive.clear_errors(identify=False)
    print("Cleared Errors")
    await asyncio.sleep(1)

    #Initalize odrive
    odrive.initCanBus()

    
    print("Put Arm at bottom center to calibrate Zero Position.")
    await asyncio.sleep(1)
    cur_pos = odrive.position
    await asyncio.sleep(1)
    print(f"Encoder Absolute Position Set: {cur_pos}")

    #odrive.setAxisState("closed_loop_control")
    odrive.setAxisState("open_loop_control")

    #add each odrive to the async loop so they will run.
    await asyncio.gather(
        odrive.loop(),
        controller(odrive) 
    )



################ RUN STUFF #################
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, stopping...")
        odrive.estop()
        
