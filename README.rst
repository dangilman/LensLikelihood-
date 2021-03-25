==============
LensLikelihood
==============


.. image:: https://img.shields.io/pypi/v/lenslikelihood.svg
        :target: https://pypi.python.org/pypi/lenslikelihood

.. image:: https://img.shields.io/travis/dangilman /lenslikelihood.svg
        :target: https://travis-ci.com/dangilman /lenslikelihood

.. image:: https://readthedocs.org/projects/lenslikelihood/badge/?version=latest
        :target: https://lenslikelihood.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

lenslikelihood contains the joint inference of several parameters describing dark matter substructure in the Universe with a sample of 12 strong gravitational lenses. To get started using this data product, you'll need to download the likelihoods from dropbox: https://www.dropbox.com/s/lp3hftzerj0oksj/precomputed_likelihoods.zip?dl=0
and then place them in the directory lenslikelihood/precomputed_likelihoods

You can then evaluate the likelihood, or sample from it. See the example notebooks "inference_5D" and "folding_in_priors" for usage examples. 

In order to use this package, you'll also need to clone this repository: https://github.com/dangilman/trikde, which handles the joint likelihood computation and plotting routines: 

Features
--------

Evaluate the joint likelihood of several different hyper-parameters describing dark matter substructure given the data from 12 strong gravitational lenses. The hyper-parameters include: 1) the normalization of the subhalo mass function 2) the normalization of the field halo mass function 3) the normalization of the mass-concentration relation 4) the logarithmic slope of the mass concentration relation 5) the logarithmic slope of the halo mass function 

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
