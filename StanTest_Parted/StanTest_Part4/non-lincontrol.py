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
    C = 3
    gamma = .0000001 
    pi = np.pi
    
    # #### Initilize #####
    rest_pos = read_raw_angle()
    val = 0
    hturns = 0
    odrive.set_torque(10)
    sum_e = 0
    p_last = 0 
    a_last = 10
    loop = asyncio.get_running_loop()
    dt = loop.time()
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
        p = position - rest_pos
        dt = loop.time() - dt
        v = (p-p_last)/dt
        #a = a_last - dt*gamma*np.sin(p*pi)*p 
        # Calculate next wheel input
        u = 10*np.sign(p+C*v) # - 0*p - 0*v + 0*a*np.sin(p*pi) # TODO: Recheck the equation
        odrive.set_torque(u)
        dt = loop.time()-dt 
        p_last = p 
        #a_last = a 
        #print(f"Coeff Estimation: {a}, Input Force: {u}, posiition {p}")
        await asyncio.sleep(0)


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
        
