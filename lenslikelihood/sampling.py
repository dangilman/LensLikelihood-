from scipy.interpolate import RegularGridInterpolator, interpn
import numpy as np
import itertools

"""
This class interpolates a likelihood, and returns a probability given a point in parameter space
"""


class InterpolatedLikelihood(object):

    def __init__(self, independent_densities, param_names, param_ranges, extrapolate=False):

        prod = independent_densities.density.T
        norm = np.max(prod)
        self.density = prod / norm
        self.param_names, self.param_ranges = param_names, param_ranges

        nbins = np.shape(prod)[0]
        points = []
        for ran in param_ranges:
            points.append(np.linspace(ran[0], ran[-1], nbins))

        if extrapolate:
            kwargs_interpolator = {'bounds_error': False, 'fill_value': None}
        else:
            kwargs_interpolator = {}

        self._extrapolate = extrapolate

        self.interp = RegularGridInterpolator(points, self.density, **kwargs_interpolator)

    def sample(self, n, nparams=None, pranges=None):

        """
        Generates n samples of the parameters in param_names from the likelihood through rejection sampling

        if param_names is None, will generate samples of all param_names used to initialize the class
        """

        if nparams is None:
            nparams = len(self.param_names)
        if pranges is None:
            pranges = self.param_ranges

        shape = (n, nparams)
        samples = np.empty(shape)
        count = 0
        last_ratio = 0

        readout = n / 10

        while True:
            proposal = []
            for ran in pranges:
                proposal.append(np.random.uniform(ran[0], ran[1]))
            proposal = tuple(proposal)
            like = self(proposal)
            u = np.random.uniform(0, 1)
            if like > u:
                samples[count, :] = proposal
                count += 1
            if count == n:
                break

            current_ratio = count/n

            if count%readout == 0 and last_ratio != current_ratio:
                print('sampling.... '+str(np.round(100*count/n, 2))+'%')
                last_ratio = count/n

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
                if self._extrapolate is False:
                    raise Exception('point was out of bounds: ', point)

        p = float(self.interp(point))

        return min(1., max(p, 0.))
