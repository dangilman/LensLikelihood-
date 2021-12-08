import numpy as np
from scipy.interpolate import interp1d
import os
import pickle
import inspect

local_path = inspect.getfile(inspect.currentframe())[0:-22]

def _load_theory_power_spectrum():

    _pk_galacticus = np.loadtxt(local_path + 'pk_galacticus.txt')
    k_galacticus, tk_galacitucs, pk_galacticus = _pk_galacticus[:, 0], _pk_galacticus[:, 1], _pk_galacticus[:, 2]
    pk_galacticus *= 1.05409
    # we include this rescaling because galacituc was run with a slightly different cosmology than the existing measurements we
    # compare our measurement with, so the LCDM prediction differs by about 5%. We can get around this annoying issue by
    # rescalingthe LCDM prediction "by hand"
    pk_galacticus_interp = interp1d(np.log10(k_galacticus), np.log10(pk_galacticus))
    transfer_function_interp = interp1d(np.log10(k_galacticus), np.log10(tk_galacitucs),
                                        bounds_error=False, fill_value='extrapolate')
    return pk_galacticus_interp, transfer_function_interp

pk_galacticus_interp, transfer_function_interp_galacticus = _load_theory_power_spectrum()

def relative_error_systematic(value, relative_error):

    return value * (relative_error + 1)

def absolute_error_systematic(value, absolute_error):

    return value + absolute_error

def func_to_min(b, r, values):
    cmbeta = r ** -b
    dx = np.log10(cmbeta) - np.log10(values)
    return np.sum(np.absolute(dx))

def load_interpolated_mapping(mass_function_model, pivot):

    path = os.getcwd() + '/../notebooks_Pk/interpolated_mappings/'
    fname = path + 'mapping_' + mass_function_model + '_pivot'+str(pivot)
    if os.path.exists(fname):
        f = open(fname, 'rb')
        return pickle.load(f)
    else:
        raise Exception('mapping interpolation does not exist')

def sample_pk(n, ar, ar2, wavenumber, k0, ns_fixed_above_pivot=None):

    nk = n + ar * np.log(wavenumber / k0) + ar2 * np.log(wavenumber / k0) ** 2
    pk = (wavenumber / k0) ** nk
    if isinstance(wavenumber, np.ndarray) or isinstance(wavenumber, list):
        inds_lower = np.where(wavenumber < k0)
        wavenumber = np.array(wavenumber)
        if ns_fixed_above_pivot is not None:
            pk[inds_lower] = (wavenumber[inds_lower] / k0) ** ns_fixed_above_pivot
        else:
            pk[inds_lower] = (wavenumber[inds_lower] / k0) ** n
    else:
        if wavenumber < k0:
            if ns_fixed_above_pivot is not None:
                pk = (wavenumber / k0) ** ns_fixed_above_pivot
            else:
                pk = (wavenumber / k0) ** n

    # normalize
    amp_at_pivot = 10 ** pk_galacticus_interp(np.log10(k0))
    t = 10**transfer_function_interp_galacticus(np.log10(wavenumber))
    t_at_pivot = 10**transfer_function_interp_galacticus(np.log10(k0))
    return amp_at_pivot * pk * (t / t_at_pivot) ** 2

def Pk_from_likelihood(samples, sample_weights, k0, k_eval=100, n_max=2000,
                       nbins=25, x_range=None, prior_sample_factor=10000, edge_cut_factor=20,
                       ns_fixed_above_pivot=None):

    counter = 0
    pk_at_point = []

    sample_weights = np.array(sample_weights)
    sample_weights *= np.max(sample_weights) ** -1
    pk_at_point_prior = []

    while len(pk_at_point_prior) < prior_sample_factor * nbins:
        i = np.random.randint(0, int(samples.shape[0]))
        q = (samples[i, 0], samples[i, 1], samples[i, 2])
        prior_point = sample_pk(*q, k_eval, k0, ns_fixed_above_pivot)
        pk_at_point_prior.append(prior_point)

    while True:

        i = np.random.randint(0, int(samples.shape[0]))
        u = np.random.uniform(0, 1)
        q = (samples[i, 0], samples[i, 1], samples[i, 2])

        if u < sample_weights[i]:
            new = sample_pk(*q, k_eval, k0)
            pk_at_point.append(new)
            counter += 1

        if counter > min(n_max, len(sample_weights)):
            break

    if x_range is None:
        # helps avoid shot noise at the edge of the pdf
        low = np.min(np.log10(pk_at_point_prior))
        high = np.max(np.log10(pk_at_point_prior))
        edge_cut_dex = (high - low) / 2 / edge_cut_factor
        ran = (low + edge_cut_dex, high - edge_cut_dex)

    else:
        ran = x_range

    h_prior, x = np.histogram(np.log10(pk_at_point_prior), range=ran, bins=nbins)
    h, _ = np.histogram(np.log10(pk_at_point), range=ran, bins=nbins)
    dx = (x[1] - x[0]) / 2
    x = x[0:-1] + dx

    valid_inds = np.where(h_prior > 0)[0]
    invalid_inds = np.where(h_prior == 0)
    likelihood = np.empty(len(h))
    likelihood[invalid_inds] = 0.
    likelihood[valid_inds] = h[valid_inds] / h_prior[valid_inds]

    likelihood *= float(np.max(likelihood)) ** -1
    likelihood_cdf = interp1d(likelihood, x)
    u = np.random.uniform(min(likelihood), max(likelihood), n_max)
    log10_pk_at_point = likelihood_cdf(u)

    posterior = h * float(np.max(h)) ** -1
    prior = h_prior * float(np.max(h_prior)) ** -1

    pk_at_point = 10 ** np.array(log10_pk_at_point)

    return pk_at_point, likelihood, x, prior, posterior

def compute_confidence_intervals(sample, num_sigma, thresh=None):
    """
    computes the upper and lower sigma from the median value.
    This functions gives good error estimates for skewed pdf's
    :param sample: 1-D sample
    :return: median, lower_sigma, upper_sigma
    """
    if thresh is not None and num_sigma > 3:
        raise ValueError("Number of sigma-constraints restricted to three. %s not valid" % num_sigma)
    num = len(sample)
    median = np.median(sample)
    sorted_sample = np.sort(sample)

    if thresh is None:
        num_threshold1 = int(round((num-1)*0.841345))
        num_threshold2 = int(round((num-1)*0.977249868))
        num_threshold3 = int(round((num-1)*0.998650102))

        if num_sigma == 1:
            upper_sigma1 = sorted_sample[num_threshold1 - 1]
            lower_sigma1 = sorted_sample[num - num_threshold1 - 1]
            return median, [median-lower_sigma1, upper_sigma1-median]
        if num_sigma == 2:
            upper_sigma2 = sorted_sample[num_threshold2 - 1]
            lower_sigma2 = sorted_sample[num - num_threshold2 - 1]
            return median, [median-lower_sigma2, upper_sigma2-median]
    else:

        assert thresh <= 1
        thresh = (1 + thresh)/2
        num_threshold = int(round((num-1) * thresh))
        upper = sorted_sample[num_threshold - 1]
        lower = sorted_sample[num - num_threshold - 1]
        return median, [median - lower, upper - median]

def sample_power_spectra_with_systematic_interp(n_samples, param_ranges_pk, param_ranges_lensing, structure_formation_interp,
                                                interpolated_lens_likelihood, systematic_error_interp,
                                                extrapolate=False, extrapolate_ranges=None,
                                                log10c8_sys=True, delta_los_sys=True,
                                                delta_alpha_sys=True, beta_sys=True, three_D=False):

    samples = np.empty((n_samples, 3))
    likelihoods = []

    for j in range(0, n_samples):

        ar_value = np.random.uniform(param_ranges_pk[1][0], param_ranges_pk[1][1])
        n_value = np.random.uniform(param_ranges_pk[0][0], param_ranges_pk[0][1])
        ar2_value = np.random.uniform(param_ranges_pk[2][0], param_ranges_pk[2][1])
        samples[j, 0] = n_value
        samples[j, 1] = ar_value
        samples[j, 2] = ar2_value

        los_norm_range = param_ranges_lensing[0]
        beta_range = param_ranges_lensing[1]
        c0_range = param_ranges_lensing[2]
        delta_alpha_range = param_ranges_lensing[3]
        ranges = [los_norm_range, beta_range, c0_range, delta_alpha_range]
        _point = structure_formation_interp(n_value, ar_value, ar2_value)
        _point = np.array(_point)
        if three_D:
            (delta_dlos, delta_beta, delta_c8, delta_alpha) = systematic_error_interp((n_value, ar_value, ar2_value))
        else:
            (delta_dlos, delta_beta, delta_c8, delta_alpha) = systematic_error_interp((ar_value, ar2_value))

        if log10c8_sys is False:
            delta_c8 = 0
        if delta_los_sys is False:
            delta_los_sys = 0
        if delta_alpha_sys is False:
            delta_alpha = 0
        if beta_sys is False:
            delta_beta = 0

        perturbation = np.array([delta_dlos, delta_beta, delta_c8, delta_alpha])
        _point += perturbation

        point = []

        for i, x in enumerate(_point):

            if extrapolate:
                if extrapolate_ranges is not None:
                    x = max(x, extrapolate_ranges[i][0])
                    x = min(x, extrapolate_ranges[i][1])

            else:
                x = max(x, ranges[i][0])
                x = min(x, ranges[i][1])

            point.append(x)

        sigma_sub_point = np.random.uniform(param_ranges_lensing[4][0], param_ranges_lensing[4][1])

        point += [float(sigma_sub_point)]

        like = interpolated_lens_likelihood(point)

        likelihoods.append(like)

    return samples, np.array(likelihoods)

def sample_power_spectra(n_samples, param_ranges_pk, param_ranges_lensing, structure_formation_interp, interpolated_lens_likelihood,
                         delta_c8=0., delta_beta=0., delta_delta_los=0., delta_delta_alpha=0,
                         extrapolate=False, extrapolate_ranges=None):

    samples = np.empty((n_samples, 3))
    likelihoods = []

    for j in range(0, n_samples):

        ar_value = np.random.uniform(param_ranges_pk[1][0], param_ranges_pk[1][1])
        n_value = np.random.uniform(param_ranges_pk[0][0], param_ranges_pk[0][1])
        ar2_value = np.random.uniform(param_ranges_pk[2][0], param_ranges_pk[2][1])
        samples[j, 0] = n_value
        samples[j, 1] = ar_value
        samples[j, 2] = ar2_value

        los_norm_range = param_ranges_lensing[0]
        beta_range = param_ranges_lensing[1]
        c0_range = param_ranges_lensing[2]
        delta_alpha_range = param_ranges_lensing[3]
        ranges = [los_norm_range, beta_range, c0_range, delta_alpha_range]
        _point = structure_formation_interp(n_value, ar_value, ar2_value)

        _point[0] = relative_error_systematic(_point[0], delta_delta_los)
        _point[1] = relative_error_systematic(_point[1], delta_beta)
        _point[2] = relative_error_systematic(_point[2], delta_c8)
        _point[3] = relative_error_systematic(_point[3], delta_delta_alpha)

        point = []

        for i, x in enumerate(_point):

            if extrapolate:
                if extrapolate_ranges is not None:
                    x = max(x, extrapolate_ranges[i][0])
                    x = min(x, extrapolate_ranges[i][1])

            else:
                x = max(x, ranges[i][0])
                x = min(x, ranges[i][1])

            point.append(x)

        sigma_sub_point = np.random.uniform(param_ranges_lensing[4][0], param_ranges_lensing[4][1])

        point += [float(sigma_sub_point)]

        like = interpolated_lens_likelihood(point)

        likelihoods.append(like)

    return samples, np.array(likelihoods)

