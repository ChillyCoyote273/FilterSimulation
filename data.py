import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter
from scipy.optimize import curve_fit


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


def generate_data(length: int = 600, sample_period: float = 0.05,
                  k_v: float = 0.001, k_a: float = 0.00001,
                  q: np.ndarray = np.array([[0.025, 0.0125], [0.0125, 0.025]]),
                  r: float = 0.01, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)

    pos = 0
    vel = 0
    
    powers = np.zeros(length)
    positions = np.zeros(length)

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
    # powers = np.ones(length)

    for i, power in enumerate(powers):
        for _ in range(1000):
            acceleration = power / k_a - vel * k_v / k_a
            vel += acceleration * sample_period / 2000
            pos += vel * sample_period / 1000
            vel += acceleration * sample_period / 2000
        err = np.random.multivariate_normal(np.zeros(2), q)
        pos += err[0]
        vel += err[1]

        positions[i] = pos + np.random.normal(0, np.abs(vel) * r)
    
    return times, powers, positions


if __name__ == "__main__":
    t, pow, pos = generate_data(600, k_v = .01, k_a = .01, seed = np.random.randint(1e9))

    velocities = (pos[1:] - pos[:-1]) / 0.05
    avg_velocities = (velocities[1:] + velocities[:-1]) / 2
    accelerations = (velocities[1:] - velocities[:-1]) / 0.05

    def f(state: np.ndarray[float], k_v: float, k_a: float) -> float:
        return state[0] / k_a - state[1] * k_v / k_a
    xs = np.array([pow[1:-1], avg_velocities])
    ys = accelerations
    (k_v, k_a), pcov = curve_fit(f, xs, ys)
    print(f'k_v: {k_v}\nk_a: {k_a}')
    # kalman_filter = KalmanFilter(k_v, k_a, np.array([[0.025, 0.0125], [0.0125, 0.025]]), 0.01, 0.05)
    # kalman_states = kalman_filter.run(pow, pos)
    # kalman_positions = kalman_states[0]
    # kalman_velocities = kalman_states[1]

    plt.plot(t, np.zeros(len(pow)))
    plt.plot(t[:-1], velocities / 100)
    # plt.plot(t, kalman_velocities)
    plt.plot(t, pos / 200)
    plt.plot(t, pow)
    plt.show()
