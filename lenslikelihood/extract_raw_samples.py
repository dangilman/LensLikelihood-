import numpy as np
import os


class RawLensSamples(object):
    """
    A class that stores individual parameter samples and flux ratios for one lens
    """

    def __init__(self, fluxes_modeled, samples_array, param_names_list):

        assert samples_array.shape[-1] == len(param_names_list)
        assert samples_array.shape[0] == fluxes_modeled.shape[0]

        self.param_names = param_names_list
        self.samples_dictionary = {}
        for i, param_name in enumerate(param_names_list):
            self.samples_dictionary[param_name] = samples_array[:, i]
        self.samples_array = samples_array
        self.fluxes_modeled = fluxes_modeled

    @classmethod
    def from_parameter_cut(cls, raw_lens_samples, param_name, param_min, param_max):

        param_names = raw_lens_samples.param_names
        idx = param_names.index(param_name)
        condition = np.logical_and(raw_lens_samples.samples_array[:, idx] <= param_max,
                                   raw_lens_samples.samples_array[:, idx] >= param_min)
        inds = np.where(condition)[0]
        samples_array = raw_lens_samples.samples_array[inds, :]
        fluxes = raw_lens_samples.fluxes_modeled[inds, :]
        return RawLensSamples(fluxes, samples_array, param_names)

    def sample_fluxes_from_params(self, param_dictionary):

        penalty = 0
        for key in param_dictionary.keys():
            mean, sigma = param_dictionary[key][0], param_dictionary[key][1]
            penalty += np.exp(-0.5 * (mean - self.samples_dictionary[key]) ** 2 / sigma ** 2)
        idx = np.argmin(penalty)
        return self.fluxes_modeled[idx, :]

    def sample_with_abc(self, fluxes_measured, param_names, flux_sigmas_model, flux_ratio_index, n_keep,
                        n_draw=5, uncertaintiy_in_ratios=False, flux_sigmas_measured=None, importance_sampling_weights=None):

        if importance_sampling_weights is not None:
            print('number of samples before importance sampling: ', self.samples_array.shape[0])
            samples_array_full, fluxes_modeled = self._importance_sampling(importance_sampling_weights)
            print('number of samples after importance sampling: ', samples_array_full.shape[0])
        else:
            samples_array_full, fluxes_modeled = self.samples_array, self.fluxes_modeled

        if flux_sigmas_measured is not None:
            fluxes_measured = self.perturb_fluxes(fluxes_measured, flux_sigmas_measured)

        for j in range(0, n_draw):
            if uncertaintiy_in_ratios:
                flux_ratios_modeled = self.flux_ratios_from_fluxes(fluxes_modeled, flux_ratio_index)
                flux_ratios_modeled = self.perturb_fluxes(flux_ratios_modeled, flux_sigmas_model)
            else:
                fluxes_modeled_array = self.perturb_fluxes(fluxes_modeled, flux_sigmas_model)
                flux_ratios_modeled = self.flux_ratios_from_fluxes(fluxes_modeled_array, flux_ratio_index)

            flux_ratios_measured = self.flux_ratios_from_fluxes(fluxes_measured)

            stat = 0
            for i in range(0, flux_ratios_modeled.shape[1]):
                if flux_sigmas_model[i] is not None:
                    stat += (flux_ratios_measured[i] - flux_ratios_modeled[:, i]) ** 2
            stat = np.sqrt(stat)
            keep_inds = np.argsort(stat)[0:n_keep]
            if j == 0:
                statistic = stat[keep_inds]
                params = samples_array_full[keep_inds, :]
            else:
                params = np.vstack((params, samples_array_full[keep_inds, :]))
                statistic = np.append(statistic, stat[keep_inds])

        keep_columns = []
        for param_name in param_names:
            keep_columns.append(self.param_names.index(param_name))

        return params[:, keep_columns], params, statistic

    def _importance_sampling(self, weights):

        p = 1
        for param in weights.keys():
            mean, sigma = weights[param][0], weights[param][1]
            idx = self.param_names.index(param)
            x = self.samples_array[:, idx]
            dx = (x - mean)/sigma
            p *= np.exp(-0.5 * dx ** 2)
        u = np.random.uniform(0, 1, self.samples_array.shape[0])
        inds_keep = np.where(u < p)[0]
        return self.samples_array[inds_keep, :], self.fluxes_modeled[inds_keep, :]

    @staticmethod
    def perturb_fluxes(fluxes, sigmas):

        fluxes_perturbed = np.empty_like(fluxes)
        for i, si in enumerate(sigmas):

            if si is None:
                df = 0
            else:

                if isinstance(si, list):
                    df = []
                    # first entry is lower sigma, second entry is higher sigma e.g. plus/minus lower/higher
                    u = np.random.rand(fluxes.shape[0])
                    for j, uj in enumerate(u):
                        if uj < 0.5:
                            dfj = -abs(np.random.normal(0, si[0])) * fluxes[j, i]
                        else:
                            dfj = abs(np.random.normal(0, si[1])) * fluxes[j, i]
                        df.append(dfj)
                    df = np.array(df)
                else:
                    df = np.random.normal(0, si * fluxes[:, i])
            fluxes_perturbed[:, i] = fluxes[:, i] + df
        return fluxes_perturbed

    @staticmethod
    def flux_ratios_from_fluxes(fluxes, flux_ratio_index=0):

        keep_inds = [i for i in np.arange(0, 4) if i != flux_ratio_index]
        keep_inds = np.array(keep_inds)
        nrows = fluxes.shape[0]
        flux_ratios = np.empty((nrows, 3))
        if fluxes.ndim == 2:
            for row in range(0, nrows):
                fr = fluxes[row, :] / fluxes[row, flux_ratio_index]
                flux_ratios[row, :] = fr[keep_inds]
        else:
            fr = fluxes / fluxes[flux_ratio_index]
            flux_ratios = fr[keep_inds]

        return flux_ratios

    @classmethod
    def fromlist(cls, samples_list):

        init = True
        for samples_class in samples_list:

            if init:
                param_names = samples_class.param_names
                fluxes = samples_class.fluxes_modeled
                samples_array = np.empty_like(samples_class.samples_array)
                for i, pname in enumerate(samples_class.param_names):
                    samples_array[:, i] = samples_class.samples_dictionary[pname]
            else:
                fluxes = np.vstack((fluxes, samples_class.fluxes_modeled))
                new_array = np.empty_like(samples_class.samples_array)
                for i, pname in enumerate(samples_class.param_names):
                    new_array[:, i] = samples_class.samples_dictionary[pname]
                samples_array = np.vstack((samples_array, new_array))

        return RawLensSamples(fluxes, samples_array, param_names)


def compile_raw_samples(fname_base_list, i_max):
    """
    Loads the parameter samples and flux ratios from text files that contain the output from a computation run
    in parrallel on a computing cluster.
    """
    if not isinstance(fname_base_list, list):
        fname_base_list = [fname_base_list]

    init = True

    for fname_base in fname_base_list:

        with open(fname_base + '/chain_1/parameters.txt', 'r') as f:
            header = f.readlines()[0]
            if init:
                param_names = header.split(' ')[0:-1]
                shuffle_columns_params = list(np.arange(0, len(param_names)))
            else:
                param_names_new = header.split(' ')[0:-1]
                shuffle_columns_params = [param_names_new.index(param) for param in param_names]
        with open(fname_base + '/chain_1/macro.txt', 'r') as f:
            header_macro = f.readlines()[0]
            if init:
                param_names_macro = header_macro.split(' ')[0:-1]
                shuffle_columns_macro = list(np.arange(0, len(param_names_macro)))
            else:
                param_names_new = header_macro.split(' ')[0:-1]
                shuffle_columns_macro = [param_names_new.index(param) for param in param_names_macro]

        for i in range(1, i_max+1):
            file_name_params = fname_base + '/chain_' + str(i) + '/parameters.txt'
            file_name_fluxes = fname_base + '/chain_' + str(i) + '/fluxes.txt'
            file_name_macro = fname_base + '/chain_' + str(i) + '/macro.txt'
            if os.path.exists(file_name_params) and os.path.exists(file_name_fluxes):
                if init:
                    params = np.loadtxt(file_name_params, skiprows=1)
                    fluxes = np.loadtxt(file_name_fluxes)
                    macro = np.loadtxt(file_name_macro, skiprows=1)
                    if macro.shape[0] != fluxes.shape[0] or macro.shape[0] != params.shape[0]:
                        pass
                    else:
                        params = np.loadtxt(file_name_params, skiprows=1)
                        fluxes = np.loadtxt(file_name_fluxes)
                        macro = np.loadtxt(file_name_macro, skiprows=1)
                        init = False
                else:
                    new_params = np.loadtxt(file_name_params, skiprows=1)
                    new_fluxes = np.loadtxt(file_name_fluxes)
                    new_macro = np.loadtxt(file_name_macro, skiprows=1)
                    if new_macro.shape[0] != new_fluxes.shape[0] or new_macro.shape[0] != new_params.shape[0]:
                        pass
                    else:
                        new_params = new_params[:, shuffle_columns_params]
                        new_macro = new_macro[:, shuffle_columns_macro]
                        params = np.vstack((params, new_params))
                        fluxes = np.vstack((fluxes, new_fluxes))
                        macro = np.vstack((macro, new_macro))

    param_names_full = param_names + param_names_macro
    params_full = np.column_stack((params, macro))

    return RawLensSamples(fluxes, params_full, param_names_full)
