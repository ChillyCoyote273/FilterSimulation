import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter
from scipy.optimize import curve_fit
from tqdm import tqdm


TIMES = [0, 1/3, 2/3, 1]
WEIGHTS = [1/8, 3/8, 3/8, 1/8]
TABLE = [
    np.array([]),
    np.array([1/3]),
    np.array([-1/3, 1]),
    np.array([1, -1, 1])
]

def runge_kutta(power: float, pos: float, vel: float, dt: float, k_v: float, k_a: float) -> tuple[float, float]:
    def f(f_vel: float) -> float:
        return power / k_a - f_vel * k_v / k_a
    
    ks = np.zeros(len(WEIGHTS))
    for i in range(len(WEIGHTS)):
        ks[i] = f(vel + dt * (ks[:i].dot(TABLE[i])))
    
    next_vel = vel + dt * ks.dot(WEIGHTS)
    next_pos = pos + dt * (vel + next_vel) / 2

    return next_pos, next_vel

model_var = 1.5
def generate_data(length: int = 600, sample_period: float = 0.05,
                  k_v: float = 0.001, k_a: float = 0.00001, k_g: float = 0.1,
                  q: np.ndarray = np.array([[1.5625e-6, 6.25e-5], [6.25e-5, 2.5e-3]]), r: float = 0.01,
                  seed: int = np.random.randint(1e9)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q *= model_var
    np.random.seed(seed)

    pos = 0
    vel = 0
    
    powers = np.zeros(length)
    positions = np.zeros(length)
    velocities = np.zeros(length)

    times = np.array([sample_period * i for i in range(length)])

    power = 0
    for i in range(length):
        if i == 0:
            powers[i] = 0
        else:
            alpha = 0.9
            powers[i] = power * (1 - alpha) + powers[i - 1] * alpha
        power += np.random.normal(-power * sample_period / 3, sample_period * 3)
    powers = np.clip(powers, -1, 1)

    for i in tqdm(range(len(powers))):
        for _ in range(1000):
            acceleration = (powers[i] - k_g) / k_a - vel * k_v / k_a
            vel += acceleration * sample_period / 2000
            pos += vel * sample_period / 1000
            vel += acceleration * sample_period / 2000
        err = np.random.multivariate_normal(np.zeros(2), q)
        pos += err[0]
        vel += err[1]

        positions[i] = pos + np.random.normal(0, np.abs(vel) * r)
        velocities[i] = vel
    
    return times, powers, positions, velocities
