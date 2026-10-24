import pyodrivecan
import asyncio
import math
import numpy as np
from datetime import datetime, timedelta
import smbus
import time
import os



################## ODRIVE ################

async def controller(odrive):
    await asyncio.sleep(0)
    while datetime.now() < stop_at:
        ev = odrive.velocity  
        if ev is None:
           ev = 0
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
        
