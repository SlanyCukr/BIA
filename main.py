import matplotlib
matplotlib.use("Qt5Agg")

from solutions import BlindSearchSolution

from functions import sphere


if __name__ == '__main__':
    solution = BlindSearchSolution(2, 10, -10)
    solution.solve(20000, sphere)