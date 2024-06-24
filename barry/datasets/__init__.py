"""
======
Models
======

.. currentmodule:: barry.datasets

Generic
=======

.. autosummary::
   :toctree: generated/

   Dataset -- Generic dataset
   MultiDataset -- Multiple datasets combined

Concrete
========

.. autosummary::
   :toctree: generated/

   PowerSpectrum_SDSS_DR12
   PowerSpectrum_Beutler2019
   CorrelationFunction_ROSS_DR12
"""

from barry.datasets.dataset import Dataset, MultiDataset
from barry.datasets.dataset_correlation_function import (
    CorrelationFunction_AbacusSummit,
    # CorrelationFunction_SDSS_DR12_Z061_NGC,
    CorrelationFunction_ROSS_DR12,
)
from barry.datasets.dataset_power_spectrum import (
    PowerSpectrum_AbacusSummit,
    PowerSpectrum_SDSS_DR12,
    PowerSpectrum_Beutler2019,
    PowerSpectrum_DESIMockChallenge_Post,
)

__all__ = [
    "PowerSpectrum_AbacusSummit",
    "CorrelationFunction_AbacusSummit",
    "PowerSpectrum_Beutler2019",
    # "CorrelationFunction_SDSS_DR12_Z061_NGC",
    "CorrelationFunction_ROSS_DR12",
]
