import sys

sys.path.append("..")
from barry.cosmology.camb_generator import getCambGenerator
from barry.postprocessing import BAOExtractor
from barry.config import setup
from barry.utils import weighted_avg_and_std, get_model_comparison_dataframe
from barry.models import PowerDing2018
from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.samplers import DynestySampler
from barry.fitter import Fitter
import numpy as np
import pandas as pd


# Check if B17 and D18 results change if we apply the BAO extractor technique.
# Spoiler: They do not.
if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)
    fitter = Fitter(dir_name, remove_output=False)

    sampler = DynestySampler(temp_dir=dir_name, nlive=200)

    d = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=True, realisation=0)
    ding = PowerDing2018(recon=True)
    

    d.set_realisation(0)
    fitter.add_model_and_dataset(ding, d, name=f"D18", linestyle="-", color="p", realisation=0)
           
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.set_num_concurrent(700)
    fitter.fit(file)

    model_results, summary = get_model_comparison_dataframe(fitter)

    print(model_results)

