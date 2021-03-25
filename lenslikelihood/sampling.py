from scipy.interpolate import RegularGridInterpolator
import numpy as np
"""
This class interpolates a likelihood, and returns a probability given a point in parameter space
"""


class InterpolatedLikelihood(object):

    def __init__(self, independent_densities, param_names, param_ranges):

        prod = independent_densities.density.T
        norm = np.max(prod)
        self.density = prod / norm
        self.param_names, self.param_ranges = param_names, param_ranges

        nbins = np.shape(prod)[0]
        points = []
        for range in param_ranges:
            points.append(np.linspace(range[0], range[-1], nbins))

        self.interp = RegularGridInterpolator(points, self.density)

    def sample(self, n):

        """
        Generates n samples of the parameters in param_names from the likelihood through rejection sampling

        if param_names is None, will generate samples of all param_names used to initialize the class
        """

        shape = (n, len(self.param_names))
        samples = np.empty(shape)
        count = 0
        while True:
            proposal = []
            for ran in self.param_ranges:
                proposal.append(np.random.uniform(ran[0], ran[1]))
            proposal = tuple(proposal)
            like = self(proposal)
            u = np.random.uniform(0, 1)
            if like > u:
                samples[count, :] = proposal
                count += 1
            if count == n:
                break
        return samples

    def __call__(self, point):

        """
        Evaluates the liklihood at a point in parameter space
        :param point: a tuple with length equal to len(param_names) that contains a point in parameter space, param ordering
        between param_names, param_ranges, and in point must be the same

        Returns the likelihood
        """
        point = np.array(point)

        for i, value in enumerate(point):
            if value is None:
                new_point = np.random.uniform(self.param_ranges[i][0], self.param_ranges[i][1])
                point[i] = new_point
            elif value < self.param_ranges[i][0] or value > self.param_ranges[i][1]:
                raise Exception('point was out of bounds: ', point)
        return float(self.interp(point))
