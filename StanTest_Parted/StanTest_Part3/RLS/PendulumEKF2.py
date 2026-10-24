"""
PendulumEKF_augmented.py
========================
Extends the original PendulumEKF to jointly estimate the state [θ, ω]
AND the gravity parameter 'a' (= mgl/I in the normalized pendulum model).

State vector:  x = [theta, omega, a]ᵀ   (3 × 1)

Dynamics:
    θ̇  =  ω
    ω̇  =  (u  −  a·sin(θ)  −  fc·tanh(ω/v_ε)  −  fv·ω) / I
    ȧ  =  0        (a treated as a slowly-drifting constant)

Jacobian F = ∂f/∂x  (used for P propagation):
    ∂θ̇/∂θ = 0,   ∂θ̇/∂ω = 1,            ∂θ̇/∂a = 0
    ∂ω̇/∂θ = -a·cos(θ)/I,
    ∂ω̇/∂ω = (-fc·sech²(ω/v_ε)/v_ε  -  fv) / I,
    ∂ω̇/∂a = -sin(θ) / I          ← the key new term
    ∂ȧ/∂θ = 0,   ∂ȧ/∂ω = 0,            ∂ȧ/∂a = 0

Measurement: y = θ    (encoder, scalar)
    H = [1, 0, 0]

Process noise:
    Torque disturbance enters through ω̇  →  Q column = [0, 1/I, 0]ᵀ
    Parameter drift enters through ȧ    →  separate small variance q_a

Public interface (drop-in for the original):
    __init__(I)
    update(theta_meas, torque_u, dt) → (theta_hat, omega_hat)
    .a   → current estimate of the gravity parameter
"""

import numpy as np


class PendulumEKF:
    """
    Error-State EKF with augmented state [θ, ω, a].

    Parameters
    ----------
    I         : float  — moment of inertia (kg·m²)
    a_init    : float  — initial guess for gravity parameter  (default 0.943,
                          can be negative if your sign convention requires it)
    q_torque  : float  — disturbance torque variance          (tuning knob)
    q_a       : float  — parameter drift variance             (tuning knob,
                          larger = allows 'a' to move faster)
    R_encoder : float  — encoder measurement noise variance
    """

    def __init__(self, I,
                 a_init: float   = 0.943,
                 q_torque: float = 100.0,
                 q_a: float      = 0.01,
                 R_encoder: float = 1e-4):

        self.I       = float(I)
        self.v_eps   = 0.1          # tanh smoothing for friction
        self.fc      = 0.0          # Coulomb friction (fixed at 0 per your request)
        self.fv      = 0.0          # Viscous friction (fixed at 0 per your request)

        # ── State: [theta, omega, a] ──────────────────────────────────────────
        self.x = np.array([[0.0],          # theta
                           [0.0],          # omega
                           [float(a_init)]])  # a

        # ── Error covariance ──────────────────────────────────────────────────
        #   [theta var,  omega var,  a var]
        self.P = np.diag([0.1, 0.1, 1.0])

        # ── Noise parameters ──────────────────────────────────────────────────
        self.q_torque = float(q_torque)
        self.q_a      = float(q_a)
        self.R        = np.array([[float(R_encoder)]])

        # ── Measurement matrix ────────────────────────────────────────────────
        self.H = np.array([[1.0, 0.0, 0.0]])

        self.initialized = False

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    # ─────────────────────────────────────────────────────────────────────────
    # Nonlinear dynamics f(x, u)
    # ─────────────────────────────────────────────────────────────────────────
    def dynamics(self, x, u):
        theta = x[0, 0]
        omega = x[1, 0]
        a     = x[2, 0]

        friction   = self.fc * np.tanh(omega / self.v_eps) + self.fv * omega
        omega_dot  = (u - a * np.sin(theta) - friction) / self.I

        return np.array([[omega],
                         [omega_dot],
                         [0.0]])          # ȧ = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Jacobian ∂f/∂x  (3 × 3)
    # ─────────────────────────────────────────────────────────────────────────
    def error_jacobian(self):
        theta = self.x[0, 0]
        omega = self.x[1, 0]
        a     = self.x[2, 0]

        sech2 = 1.0 - np.tanh(omega / self.v_eps) ** 2

        df_dtheta = -a * np.cos(theta) / self.I
        df_domega = (-self.fc * sech2 / self.v_eps - self.fv) / self.I
        df_da     = -np.sin(theta) / self.I          # ← key new term

        return np.array([
            [0.0,       1.0,       0.0],
            [df_dtheta, df_domega, df_da],
            [0.0,       0.0,       0.0],
        ])

    # ─────────────────────────────────────────────────────────────────────────
    def discretize(self, A, dt):
        Adt = A * dt
        return np.eye(3) + Adt + 0.5 * (Adt @ Adt)

    # ─────────────────────────────────────────────────────────────────────────
    # Process noise Q  (3 × 3)
    # ─────────────────────────────────────────────────────────────────────────
    def process_noise(self, dt):
        # Torque disturbance channel: [0, 1/I, 0]ᵀ
        G_torque = np.array([[0.0], [1.0 / self.I], [0.0]])
        Q_torque = G_torque @ G_torque.T * self.q_torque * dt

        # Parameter drift channel: [0, 0, 1]ᵀ
        G_a  = np.array([[0.0], [0.0], [1.0]])
        Q_a  = G_a @ G_a.T * self.q_a * dt

        return Q_torque + Q_a

    # ─────────────────────────────────────────────────────────────────────────
    # Main update  — same external signature as original
    # ─────────────────────────────────────────────────────────────────────────
    def update(self, theta_meas: float, torque_u: float, dt: float):
        """
        Returns
        -------
        theta_hat : float
        omega_hat : float

        Access the gravity estimate via   ekf.a
        """
        if dt <= 0:
            return self.x[0, 0], self.x[1, 0]

        # ── Initialisation ────────────────────────────────────────────────────
        if not self.initialized:
            self.x[0, 0] = float(theta_meas)
            self.initialized = True
            return self.x[0, 0], self.x[1, 0]

        # ── 1. Propagate nominal state ────────────────────────────────────────
        xdot  = self.dynamics(self.x, torque_u)
        self.x = self.x + dt * xdot
        self.x[0, 0] = self.wrap_angle(self.x[0, 0])

        # ── 2. Propagate error covariance ─────────────────────────────────────
        A = self.error_jacobian()
        F = self.discretize(A, dt)
        Q = self.process_noise(dt)
        self.P = F @ self.P @ F.T + Q

        # ── 3. Measurement update ─────────────────────────────────────────────
        innovation = self.wrap_angle(float(theta_meas) - self.x[0, 0])
        y = np.array([[innovation]])

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)       # shape (3, 1)

        dx      = K @ y                                 # correction vector
        self.x += dx
        self.x[0, 0] = self.wrap_angle(self.x[0, 0])

        # Joseph-stabilised covariance
        I3      = np.eye(3)
        IKH     = I3 - K @ self.H
        self.P  = IKH @ self.P @ IKH.T + K @ self.R @ K.T

        return self.x[0, 0], self.x[1, 0]

    # ─────────────────────────────────────────────────────────────────────────
    @property
    def a(self) -> float:
        """Current gravity parameter estimate."""
        return float(self.x[2, 0])
