import numpy as np

class GaussianWeight(object):

    def __init__(self, mean, sigma):

        self.mean, self.sigma = mean, sigma

    def __call__(self, values):

        dx = (self.mean - values)/self.sigma
        exponent = -0.5 * dx ** 2
        w = np.exp(exponent)
        norm = np.max(w)
        return w/norm

class InvertedGaussianWeight(object):

    def __init__(self, mean, sigma):

        self.gaussian = GaussianWeight(mean, sigma)

    def __call__(self, values):

        weight_gaussian = self.gaussian(values)
        return 1/weight_gaussian

class TransformedGaussianWeight(object):

    def __init__(self, mean_old, sigma_old, mean_new, sigma_new):

        self.inverted_gaussian = InvertedGaussianWeight(mean_old, sigma_old)
        self.gaussian = GaussianWeight(mean_new, sigma_new)

    def __call__(self, values):

        weights = self.inverted_gaussian(values)
        weights *= self.gaussian(values)
        return weights
