import numpy as np
import cmath, math


def signum(x):
    return math.cos(cmath.phase(x))


def main() -> None:
    tests = np.linspace(-10, 10, 101)
    results = np.vectorize(signum)(tests)
    print(np.array([tests, results]).T)


if __name__ == "__main__":
    main()
