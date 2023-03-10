import numpy as np
from scipy import signal
from tqdm import tqdm


class KalmanFilter:
    def __init__(self, k_v: float, k_a: float, q: np.ndarray, r: float, dt: float) -> None:
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

        self.A, self.B, self.C, self.D, dt = signal.cont2discrete((self.A, self.B, self.C, self.D), dt)

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
    
    def predict(self, u: np.ndarray):
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.transpose() + self.Q
    
    def correct(self, y: np.ndarray):
        y_res = y - self.C @ self.x
        S = self.C @ self.P @ self.C.transpose() + np.array([[np.abs(self.x[1, 0]) * self.r]])
        K = self.P @ self.C.transpose() @ np.linalg.inv(S)
        self.x += K @ y_res
        self.P = (np.identity(2) - K @ self.C) @ self.P
    
    def run(self, us: np.ndarray[float], ys: np.ndarray[float]) -> np.ndarray[float]:
        states = np.zeros((len(us), 2))
        covariances = []
        for i, (u, y) in tqdm(enumerate(zip(us, ys))):
            u = np.array([[u]])
            y = np.array([[y]])
            self.predict(u)
            self.correct(y)
            states[i] = self.x[:, 0]
            covariances.append(self.P)
        return states, np.array(covariances)
