import numpy as np
from scipy.linalg import expm


def get_calculation(a):
    A = np.array([
        [ a, 1.0 ],
        [ 0, 0.2 ]
    ])
    B = np.array([
        [ 0 ],
        [ 0.1 ]
    ])
    A_d = expm(A)
    B_d = np.linalg.inv(A) @ (A_d - np.eye(2)) @ B
    block = np.concatenate(A, B, axis=1)
    return block


def get_matrix(a):
    A = np.array([
        [ a, 1.0 ],
        [ 0, 0.2 ]
	])
    # return np.linalg.pinv(A) @ (expm(A) - np.eye(2))
    P, D = np.linalg.eig(A)
    return expm(A)


def main():
    for i in range(10):
        print(get_calculation(0.1 ** i))
    print(get_calculation(0))


if __name__ == "__main__":
    main()
