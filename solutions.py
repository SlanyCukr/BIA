from abc import ABC, abstractmethod
from typing import Callable, List, Optional
import numpy as np
from matplotlib import pyplot as plt, animation, cm
from sympy.physics.continuum_mechanics.beam import numpy


class Solution(ABC):
    def __init__(self, dimension: int, lower_bound: float, upper_bound: float):
        """
        Constructor - sets dimension and bounds used later.
        :param dimension:   Dimension of solution - 2D, 3D, maybe more dimension later
        :param lower_bound: Restriction for solution
        :param upper_bound: Restriction for solution
        """
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.parameters = numpy.zeros(self.dimension)  # solution parameters

    @staticmethod
    def random_range(n: int, value_min: float, value_max: float) -> List:
        """
        Generates range of random values.
        :param n:           How many random values?
        :param value_min:   Minimum value of random value
        :param value_max:   Maximum value of random value
        :return:
        """
        return (value_max - value_min) * np.random.rand(n) + value_min

    def draw(self, fnc: Callable, axes):
        """
        Draws the actual function - sphere, ...
        :param fnc:     Function to call
        :param axes:    Axes object
        :return:
        """
        x_values = np.linspace(self.lower_bound, self.upper_bound, 200)
        y_values = np.linspace(self.lower_bound, self.upper_bound, 200)
        x_values, y_values = np.meshgrid(x_values, y_values)
        z_values = fnc([x_values, y_values])
        axes.plot_surface(x_values, y_values, z_values, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.6)

    @staticmethod
    def update_points(frame: int, x: List, y: List, z: List, scatter: List, axes):
        """
        Updates scatter points based on supplied x, y, z values and frame index.
        :param frame:   Number of the frame being animated
        :param x:       X values
        :param y:       Y values
        :param z:       Z values
        :param scatter: Old scatter object
        :param axes:    Axes object used for creating new scatter object
        """
        scatter[0].remove()
        scatter[0] = axes.scatter([x[frame]], [y[frame]], [z[frame]], c='red')
        print(f"Frame number: {frame}.")

    @abstractmethod
    def solve(self, **kwargs):
        pass

    @abstractmethod
    def visualize(self, **kwargs):
        pass


class BlindSearchSolution(Solution):
    def solve(self, point_count: int, fnc: Callable) -> Optional[float]:
        """
        Blindly searches for point with the lowest value.
        :param point_count: How many points to try
        :param fnc:         Function to call to calculate Z value (sphere, ...)
        """
        best_points = []
        params = []

        # create random values for each dimension
        for x in range(self.dimension):
            params.append(self.random_range(point_count, self.lower_bound, self.upper_bound))

        # blindly search for best values
        best_value = np.inf
        for i in range(point_count):

            # select correct arguments for function
            arg = []
            for param in params:
                arg.append(param[i])

            calculated_value = fnc(arg)

            if calculated_value < best_value:
                best_value = calculated_value
                best_points.append((arg[0], arg[1], calculated_value))

        if self.dimension == 2:
            self.visualize(best_points, fnc)
        return best_value

    def visualize(self, points: List, fnc: Callable):
        """
        Visualizes Blind search algorithm
        :param points:  Calculated points
        :param fnc:     Function to draw
        """
        fig = plt.figure()
        axes = plt.subplot(111, projection="3d")

        # draw function
        self.draw(fnc, axes)

        x_values = [x[0] for x in points]
        y_values = [x[1] for x in points]
        z_values = [x[2] for x in points]

        scatter = axes.scatter([], [], [], c='red')

        # this function continually displays precomputed data inside x_value, y_value, z_values
        animation_ = animation.FuncAnimation(fig, self.update_points, len(points), interval=500, fargs=(x_values, y_values, z_values, [scatter], axes),
                                             repeat=False)
        plt.show()
