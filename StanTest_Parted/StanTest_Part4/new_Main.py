import pyodrivecan
import asyncio
import math
import numpy as np
from datetime import datetime, timedelta
import smbus


################ ENCODER ################
bus = smbus.SMBus(1)

def read_raw_angle():
    """Reads normalized angle in [0, 1)."""
    data = bus.read_i2c_block_data(0x40, 0xFE, 2)
    raw = ((data[0] << 6) | (data[1] >> 2)) & 0x0FFF
    return raw / 4096.0



################ KALMAN FILTER ################

class KalmanFilter1D:
    def __init__(self, q_theta=1e-3, q_omega=5e-2, r_meas=1e-2):
        # states: theta, omega
        self.x = np.zeros((2, 1))
        self.P = np.eye(2)
        self.Q = np.array([[q_theta, 0],
                           [0, q_omega]])
        self.R = np.array([[r_meas]])
        self.H = np.array([[1, 0]])
        self.initialized = False

    def initialize(self, theta0):
        self.x[0, 0] = theta0
        self.x[1, 0] = 0
        self.initialized = True

    def predict(self, dt):
        if not self.initialized:
            return
        F = np.array([[1, dt],
                      [0, 1]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        if not self.initialized:
            self.initialize(z)
            return

        z_vec = np.array([[z]])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ (z_vec - self.H @ self.x)
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P

    @property
    def theta(self):
        return float(self.x[0, 0])

    @property
    def omega(self):
        return float(self.x[1, 0])



############### CONTROLLER (SAFE MODE) ###############

async def controller(odrive):

    odrive.set_controller_mode("torque_control")
    stop_at = datetime.now() + timedelta(seconds=10000)

    # Gains (SAFE)
    Kp = 2.0
    Ki = 0.01
    Kd = 3.0     # Disabled until stable
    max_torque = 0.5

    rest_pos = read_raw_angle()

    # Unwrap state
    last_angle = rest_pos
    hturns = 0

    # Kalman filter
    kf = KalmanFilter1D(
        q_theta=1e-3,
        q_omega=5e-2,
        r_meas=1e-2
    )
    kf.initialize(rest_pos)

    loop = asyncio.get_running_loop()
    last_t = loop.time()

    # Hysteresis unwrap thresholds
    unwrap_low = -0.6
    unwrap_high = 0.6

    while datetime.now() < stop_at:

        now_t = loop.time()
        dt = now_t - last_t
        if dt <= 0:
            dt = 1e-4
        last_t = now_t

        # --- Read Encoder ---
        angle = read_raw_angle()
        diff = angle - last_angle

        # Hysteresis unwrap (more robust)
        if diff < unwrap_low:
            hturns += 1
        elif diff > unwrap_high:
            hturns -= 1

        raw_pos = angle + hturns
        last_angle = angle

        # --- KF update ---
        kf.predict(dt)
        kf.update(raw_pos)

        position = kf.theta
        velocity = kf.omega

        # --- Control Law (safe) ---
        error = position - rest_pos

        u = Kp * error + Ki * 0 + Kd * 0
        u = max(min(u, max_torque), -max_torque)

        odrive.set_torque(u)

        # Debug print (OPTIONAL)
        # print(f"pos={position:.3f} vel={velocity:.3f} u={u:.3f}")

        await asyncio.sleep(0)



################ MAIN ################

odrive = pyodrivecan.ODriveCAN(0)

async def main():
    odrive.clear_errors(identify=False)
    await asyncio.sleep(0.5)

    odrive.initCanBus()
    await asyncio.sleep(0.5)

    print("Zero encoder position...")
    print(f"ODrive Start Pos: {odrive.position}")

    odrive.setAxisState("open_loop_control")

    await asyncio.gather(
        odrive.loop(),
        controller(odrive)
    )



if __name__ == "__main__":
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except:
        pass

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        odrive.estop()
        print("STOPPED")

