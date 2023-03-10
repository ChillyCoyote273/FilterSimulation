import numpy as np
import matplotlib.pyplot as plt
from data import generate_data
from scipy.optimize import curve_fit
from kalman_filter import KalmanFilter


def main() -> None:
    t, pow, pos, vels = generate_data(600, k_v = .01, k_a = .01)

    velocities = (pos[1:] - pos[:-1]) / 0.05
    avg_velocities = (velocities[1:] + velocities[:-1]) / 2
    accelerations = (velocities[1:] - velocities[:-1]) / 0.05

    def f(state: np.ndarray[float], k_v: float, k_a: float) -> float:
        return state[0] / k_a - state[1] * k_v / k_a
    xs = np.array([pow[1:-1], avg_velocities])
    ys = accelerations
    (k_v, k_a), pcov = curve_fit(f, xs, ys)
    kalman_filter = KalmanFilter(k_v, k_a, np.array([[0.025, 0.0125], [0.0125, 0.025]]), 0.01, 0.05)
    kalman_states, covs = kalman_filter.run(pow, pos)
    kalman_positions = kalman_states[:, 0]
    kalman_velocities = kalman_states[:, 1]
    print(kalman_states[255:265, :])
    print(pos[255:265])
    print(covs[255:265, :, :])

    plt.plot(t, np.zeros(len(pow)))
    plt.plot(t[:-1], velocities)
    plt.plot(t, kalman_velocities)
    plt.plot(t, vels)
    plt.show()


if __name__ == "__main__":
    main()
