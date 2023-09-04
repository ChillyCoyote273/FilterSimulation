import numpy as np
import matplotlib.pyplot as plt
from data import generate_data
from scipy.optimize import curve_fit
from kalman_filter import KalmanFilter
from tqdm import tqdm


def filter_data(us: np.ndarray[float], ys: np.ndarray[float],
                k_v: float = .01, k_a: float = .001, k_g: float = 0.1,
                q: np.ndarray[float] = np.array([[1.5625e-6, 6.25e-5], [6.25e-5, 2.5e-3]]),
                r: float = .01, dt: float = 0.05) -> np.ndarray[float]:
    kalman_filter = KalmanFilter(k_v, k_a, k_g, q, r, dt)
    kalman_states, _ = kalman_filter.run(us, ys)
    return kalman_states


def estimate_model(powers: np.ndarray[float], velocities: np.ndarray[float],
                   accelerations: np.ndarray[float]) -> tuple[float, float]:
    def f(x: np.ndarray[float], k_v: float, k_a: float, k_g: float) -> float:
        return (x[0] - k_g) / k_a - x[1] * k_v / k_a

    xs = np.array([
        powers,
        velocities
    ])
    ys = accelerations
    (k_v, k_a, k_g), cov = curve_fit(f, xs, ys)
    return k_v, k_a, k_g


def estimate_sensor_covariance(positions: np.ndarray[float], velocities: np.ndarray[float],
                               measurements: np.ndarray[float]) -> float:
    errors = positions - measurements
    errors = errors ** 2
    errors /= np.abs(velocities)
    return sum(errors) / (len(errors) - 1)


def estimate_model_covariance(states: np.ndarray[float], powers: np.ndarray[float],
                              k_v: float, k_a: float, k_g: float) -> np.ndarray[float]:
    kalman_filter = KalmanFilter(k_v, k_a, k_g, 1, 1, 0.05)
    next_states = kalman_filter.predict_vector(states, powers)


def main() -> None:
    r = 0.01

    t, pow, pos, vels = generate_data(600, k_v = 0.01, k_a = 0.001, r=r)

    kalman_states = filter_data(pow, pos, r=r)
    kalman_positions = kalman_states[0]
    kalman_velocities = kalman_states[1]
    kalman_accelerations = kalman_states[2]

    naive_velocities = np.diff(pos) / 0.05
    naive_accelerations = np.diff(naive_velocities) / 0.05

    # plt.plot(t, pow * 100)
    # plt.plot(t, pos)
    # plt.plot(t[:-1], naive_velocities, c='r')
    # plt.plot(t, vels, c='k')
    # plt.plot(t, kalman_velocities, c='g')
    plt.plot(t[:-2], naive_accelerations, c='r')
    # plt.plot(t, vels, c='k')
    plt.plot(t, kalman_accelerations, c='g')
    
    plt.show()

    # k_v = 1
    # k_a = 1
    # k_g = 0.25
    # k_vs = [k_v]
    # k_as = [k_a]
    # k_gs = [k_g]
    # estimated_sensor_variance = []
    # for i in tqdm(range(20)):
    #     kalman_states = filter_data(pow, pos, k_v=k_v, k_a=k_a, k_g=k_g)
    #     kalman_positions = kalman_states[0]
    #     kalman_velocities = kalman_states[1]
    #     kalman_accelerations = kalman_states[2]
    #     estimated_sensor_variance.append(
    #         estimate_sensor_covariance(kalman_positions, kalman_velocities, pow)
    #     )
    #     k_v, k_a, k_g = estimate_model(pow, kalman_velocities, kalman_accelerations)
    #     k_vs.append(k_v)
    #     k_as.append(k_a)
    #     k_gs.append(k_g)

        # plt.plot(t, np.zeros(len(pow)), color='black')
        # plt.plot(t, pow * 100, color='red')
        # plt.plot(t, kalman_positions / 5, color='green')
        # plt.plot(t, kalman_velocities, color='blue')
        # plt.plot(t, vels, color='purple')
        # plt.show()

    # plt.plot(np.array(k_as))
    # plt.show()
    # k_vs = np.array(k_vs)
    # k_as = np.array(k_as)
    # k_gs = np.array(k_gs)
    # print(k_vs)
    # print(k_as)
    # print(k_gs)
    # print(estimated_sensor_variance)


if __name__ == "__main__":
    main()
