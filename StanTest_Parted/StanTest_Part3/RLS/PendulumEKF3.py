import numpy as np


class PendulumEKF:
    def __init__(self, I,
                 a_init: float    = 0.943,
                 q_torque: float  = 1.0,
                 q_a: float       = 0.0000000000005,
                 R_encoder: float = 1e-4):

        self.I     = float(I)
        self.v_eps = 0.1
        self.fc    = 0.0
        self.fv    = 0.0

        self.x = np.array([[0.0],
                           [0.0],
                           [float(a_init)]])

        self.P = np.diag([0.1, 0.1, 1.0])

        self.q_torque = float(q_torque)
        self.q_a      = float(q_a)
        self.R        = np.array([[float(R_encoder)]])
        self.H        = np.array([[1.0, 0.0, 0.0]])

        # ── Debug exposures ───────────────────────────────────────────────────
        self.K_a        = 0.0   # Kalman gain for 'a' — watch this
        self.innovation = 0.0   # raw measurement residual

        self.initialized = False

    @staticmethod
    def wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def dynamics(self, x, u):
        theta = x[0, 0]
        omega = x[1, 0]
        a     = x[2, 0]

        friction  = self.fc * np.tanh(omega / self.v_eps) + self.fv * omega
        omega_dot = (u + a * np.sin(theta) - friction) / self.I  # + for inverted pendulum

        return np.array([[omega],
                         [omega_dot],
                         [0.0]])

    def error_jacobian(self):
        theta = self.x[0, 0]
        omega = self.x[1, 0]
        a     = self.x[2, 0]

        sech2 = 1.0 - np.tanh(omega / self.v_eps) ** 2

        df_dtheta = a * np.cos(theta) / self.I    # flipped: +a·sin(θ) in dynamics
        df_domega = (-self.fc * sech2 / self.v_eps - self.fv) / self.I
        df_da     = np.sin(theta) / self.I         # flipped: +sin(θ)/I

        return np.array([
            [0.0,       1.0,       0.0],
            [df_dtheta, df_domega, df_da],
            [0.0,       0.0,       0.0],
        ])

    def discretize(self, A, dt):
        Adt = A * dt
        return np.eye(3) + Adt + 0.5 * (Adt @ Adt)

    def process_noise(self, dt):
        G_torque = np.array([[0.0], [1.0 / self.I], [0.0]])
        Q_torque = G_torque @ G_torque.T * self.q_torque * dt

        G_a = np.array([[0.0], [0.0], [1.0]])
        Q_a = G_a @ G_a.T * self.q_a * dt

        return Q_torque + Q_a

    def update(self, theta_meas: float, torque_u: float, dt: float):
        if dt <= 0:
            return self.x[0, 0], self.x[1, 0]

        if not self.initialized:
            self.x[0, 0] = float(theta_meas)
            self.initialized = True
            return self.x[0, 0], self.x[1, 0]

        # 1. Propagate state
        xdot   = self.dynamics(self.x, torque_u)
        self.x = self.x + dt * xdot
        self.x[0, 0] = self.wrap_angle(self.x[0, 0])

        # 2. Propagate covariance
        A = self.error_jacobian()
        F = self.discretize(A, dt)
        Q = self.process_noise(dt)
        self.P = F @ self.P @ F.T + Q

        # 3. Measurement update
        self.innovation = self.wrap_angle(float(theta_meas) - self.x[0, 0])
        y = np.array([[self.innovation]])

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)   # (3,1)

        # ── Expose Kalman gain for 'a' ────────────────────────────────────────
        self.K_a = float(K[2, 0])

        dx      = K @ y
        self.x += dx
        self.x[0, 0] = self.wrap_angle(self.x[0, 0])

        # Joseph-stabilised covariance
        IKH    = np.eye(3) - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T

        return self.x[0, 0], self.x[1, 0]

    @property
    def a(self) -> float:
        return float(self.x[2, 0])
