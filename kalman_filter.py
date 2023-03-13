import numpy as np
from scipy import signal
from tqdm import tqdm


class KalmanFilter:
    def __init__(self, k_v: float, k_a: float, k_g: float, q: np.ndarray, r: float, dt: float) -> None:
        self.A = np.array([
            [0.0, 1.0],
            [0.0, -k_v/k_a]
        ])
        self.B = np.array([
            [0.0],
            [1/k_a]
        ])
        self.C = np.array([
            [1.0, 0.0]
        ])
        self.D = np.array([
            [0.0]
        ])

        self.A, self.B, self.C, self.D, self.dt = signal.cont2discrete((self.A, self.B, self.C, self.D), dt)

        self.Q = q
        self.r = r

        self.P = np.array([
            [0.0, 0.0],
            [0.0, 0.0]
        ])
        self.x = np.array([
            [0.0],
            [0.0]
        ])
        self.ff = k_g
    
    def predict(self, u: np.ndarray):
        self.x = self.A @ self.x + self.B @ (u - self.ff)
        self.P = self.A @ self.P @ self.A.transpose() + self.Q
    
    def predict_vector(self, xs: np.ndarray[float], us: np.ndarray[float]) -> np.ndarray[float]:
        return self.A @ self.xs + self.B @ (us - self.ff)
    
    def correct(self, y: np.ndarray):
        y_res = y - self.C @ self.x
        S = self.C @ self.P @ self.C.transpose() + np.array([[np.abs(self.x[1, 0]) * self.r]])
        K = self.P @ self.C.transpose() @ np.linalg.inv(S)
        self.x += K @ y_res
        self.P = (np.identity(2) - K @ self.C) @ self.P
    
    def run(self, us: np.ndarray[float], ys: np.ndarray[float]) -> np.ndarray[float]:
        pri_states = []
        post_states = []
        pri_covs = []
        post_covs = []

        for i, (u, y) in enumerate(zip(us, ys)):
            u = np.array([[u]])
            y = np.array([[y]])
            self.predict(u)
            pri_states.append(self.x.copy())
            pri_covs.append(self.P.copy())
            self.correct(y)
            post_states.append(self.x.copy())
            post_covs.append(self.P.copy())
        
        pri_states = np.array(pri_states)
        pri_covs = np.array(pri_covs)
        post_states = np.array(post_states)
        post_covs = np.array(post_covs)

        smoothed_states = [self.x.copy()]
        smoothed_covs = [self.P.copy()]

        for i in range(len(us) - 2, -1, -1):
            C = post_covs[i] @ self.A.transpose() @ np.linalg.pinv(pri_covs[i + 1])
            self.x = post_states[i] + C @ (self.x - pri_states[i + 1])
            self.P = post_covs[i] + C @ (self.P - pri_covs[i + 1]) @ C.transpose()

            smoothed_states.append(self.x.copy())
            smoothed_covs.append(self.P.copy())
        
        smoothed_states = np.array(smoothed_states[::-1])
        smoothed_covs = np.array(smoothed_covs[::-1])

        positions = smoothed_states[:, 0].flatten()
        velocities = smoothed_states[:, 1].flatten()
        accelerations = (velocities[1:] - velocities[:-1]) / self.dt
        accelerations = (np.insert(accelerations, 0, 0) + np.append(accelerations, 0)) / 2
        accelerations[0] *= 2
        accelerations[-1] *= 2

        states = np.array([
            positions,
            velocities,
            accelerations
        ])

        return states, post_covs
