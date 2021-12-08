import numpy as np

# name of each lens system
all_param_ranges = {'sigma_sub': [0, 0.125], 'LOS_normalization': [0., 2.0], 'delta_power_law_index': [-0.6, 0.9],
               'c0': [1, 200], 'beta': [0., 6.0], 'delta_power_law_index_coupling': [0., 1.], 'a_m_1': [-0.04, 0.04]}
all_param_ranges_loguniformc0 = {'sigma_sub': [0, 0.1], 'LOS_normalization': [0., 2.0], 'delta_power_law_index': [-0.6, 0.9],
               'c0': [0, 2.3], 'beta': [0., 4.0], 'delta_power_law_index_coupling': [0., 1.], 'a_m_1': [-0.04, 0.04]}
all_param_ranges_version2 = {'sigma_sub': [0, 0.125], 'LOS_normalization': [0., 2.5], 'delta_power_law_index': [-0.6, 0.9],
               'log10c0': [0, 4], 'beta': [-0.2, 15.0], 'delta_power_law_index_coupling': [0.7, 1.], 'a_m_1': [-0.04, 0.04]}

all_lens_names = [
            'HE0435',
            'WGD2038',
            'B1422',
            'WFI2033',
            'PSJ1606',
            'WFI2026',
            'RXJ0911',
            'MG0414',
            'PG1115', 'RXJ1131',
                  'WGDJ0405']

# the measured image fluxes for each system
_mg0414_fluxes = np.array([1, 0.83, 0.36, 0.16]) # Stacey et al. 2020
_mg0414_fluxes_katz = np.array([1, 0.903, 0.389, 0.145]) # Katz et al. 1997
_pg1115_fluxes = np.array([1., 0.93, 0.16, 0.21]) # Chiba et al. 2005
_he0435_fluxes = np.array([0.96, 0.976, 1., 0.65]) # Nierenberg et al. 2017
_b1422_fluxes = np.array([0.88 , 1., 0.474, 0.025]) # Nierenberg et al. 2014
_wgd2038_fluxes = np.array([0.86, 1., 0.79, 0.40]) # Nierenberg et al. 2020
_wgdj0405_fluxes = np.array([0.80, 0.52, 1., 0.94]) # Nierenberg et al. 2020
_wfi2033_fluxes = np.array([1., 0.65, 0.5, 0.53]) # Nierenberg et al. 2020
_psj1606_fluxes = np.array([1., 1., 0.59, 0.79]) # Nierenberg et al. 2020
_wfi2026_fluxes = np.array([1., 0.75, 0.31, 0.28]) # Nierenberg et al. 2020
_rxj0911_fluxes = np.array([0.56, 1., 0.53, 0.24]) # Nierenberg et al. 2020
_rxj1131_fluxes = np.array([1., 0.61, 0.73, 0.12]) # Sugai et al. 20??
_b0128_fluxes = np.array([1., 0.584, 0.520, 0.506]) # Phillips 2000
_mg0414_sigmas = [0.05/0.83, 0.04/0.36, 0.04/0.34]
_mg0414_sigmas_katz = [0.1, 0.1, 0.1]
_pg1115_sigmas = [0.06/0.93, 0.07/0.16, 0.04/0.21]
_he0435_sigmas = [0.05, 0.049, 0.048, 0.056]
_b1422_sigmas = [0.01/0.88, 0.01, 0.006/0.47, None]
_wgd2038_sigmas = [0.01, 0.02/1.16, 0.02/0.92, 0.01/0.46]
_wgdj0405_sigmas = [0.04, 0.04/0.65, 0.03/1.25, 0.04/1.17]
_wfi2033_sigmas = [0.03, 0.03/0.64, 0.02/0.5, 0.02/0.53]
_psj1606_sigmas = [0.03, 0.03, 0.02/0.6, 0.02/0.78]
_wfi2026_sigmas = [0.02, 0.02/0.75, 0.02/0.31, 0.01/0.28]
_rxj0911_sigmas = [0.04/0.56, 0.05, 0.04/0.53, 0.04/0.24]
_b0128_sigmas = [0.029/0.584, 0.029/0.520, 0.032/0.506]

# Sugai et al. quote an asymmetric two-sided uncertainty for the flux ratios. The simple uncertainty model below takes the largest
# value of the two-sided uncertainty to be conservative, while the _asymmetric model implements the uncertainties quoted
# by Sugai et al.
_rxj1131_sigmas = [0.04/1.63, 0.12/1.19, None]
_rxj1131_sigmas_asymmetric = [[0.02/1.62, 0.04/1.62], [0.12/1.19, 0.03/1.19], None]

flux_measurements = {'B1422': _b1422_fluxes,
                    'HE0435': _he0435_fluxes,
                     'WGD2038': _wgd2038_fluxes,
                     'WGDJ0405': _wgdj0405_fluxes,
                     'WFI2033': _wfi2033_fluxes,
                     'PSJ1606': _psj1606_fluxes,
                     'WFI2026': _wfi2026_fluxes,
                     'RXJ0911': _rxj0911_fluxes,
                     'RXJ1131_symmetric_uncertainties': _rxj1131_fluxes,
                     'MG0414': _mg0414_fluxes,
                     'MG0414_Katz': _mg0414_fluxes_katz,
                     'PG1115': _pg1115_fluxes,
                     'RXJ1131': _rxj1131_fluxes,
                     'B0128': _b0128_fluxes}

# the measurement uncertainties for the image fluxes
# syntax is {'lens_name': (sigma, number, uncertaintiy_in_ratio)} where sigma is the 1-sigma measurement uncertainty, number
# specifies the image we want to use as the reference when computing the flux ratios, and "uncertaintiy_in_ratio" is
# True/False. True means the measurement uncertainties were presented in the literature for the flux ratios,
# while False means the measurement uncertainties were specified for the image fluxes
_uncertainty_in_flux_ratio = False
_reference_index = 0
flux_measurement_uncertainties = {'B1422': (_b1422_sigmas, _reference_index, _uncertainty_in_flux_ratio),
                                 'HE0435': (_he0435_sigmas, _reference_index, _uncertainty_in_flux_ratio),
                                 'WGD2038': (_wgd2038_sigmas, _reference_index, _uncertainty_in_flux_ratio),
                                 'WGDJ0405': (_wgdj0405_sigmas, _reference_index, _uncertainty_in_flux_ratio),
                                 'WFI2033': (_wfi2033_sigmas, _reference_index, _uncertainty_in_flux_ratio),
                                 'PSJ1606': (_psj1606_sigmas, _reference_index, _uncertainty_in_flux_ratio),
                                 'WFI2026': (_wfi2026_sigmas, _reference_index, _uncertainty_in_flux_ratio),
                                 'RXJ0911': (_rxj0911_sigmas, _reference_index, _uncertainty_in_flux_ratio),
                                 'RXJ1131_symmetric_uncertainties': (_rxj1131_sigmas, 1, True),
                                 'RXJ1131': (_rxj1131_sigmas_asymmetric, 1, True),
                                 'MG0414': (_mg0414_sigmas, 0, True),
                                 'PG1115': (_pg1115_sigmas, 0, True),
                                  'B0128': (_b0128_sigmas, 0, True),
                                  'MG0414_Katz': (_mg0414_sigmas_katz, 0, True)}

