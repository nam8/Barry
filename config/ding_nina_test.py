import sys

sys.path.append("..")
from barry.cosmology.camb_generator import getCambGenerator
from barry.postprocessing import BAOExtractor
from barry.config import setup
from barry.utils import weighted_avg_and_std, get_model_comparison_dataframe
from barry.models import PowerDing2018, PowerBeutler2017
from barry.datasets import PowerSpectrum_SDSS_DR12, PowerSpectrum_AbacusSummit
from barry.samplers import DynestySampler
from barry.fitter import Fitter
import numpy as np
import pandas as pd

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)
    fitter = Fitter(dir_name, remove_output=False)

    sampler = DynestySampler(temp_dir=dir_name, nlive=200)

    # redshift_bin = 2 --> z = 0.51. Default is redshift_bin = 3 (z = 0.61). 
    d = PowerSpectrum_SDSS_DR12(recon="iso", realisation=0, redshift_bin=2) 

    ding = PowerDing2018(recon=d_as.recon)
    beutler = PowerBeutler2017(recon=d_as.recon)
    
    # this returns alpha ~ 1
    ding.sanity_check(d)
    # and so do this 
    beutler.sanity_check(d)

    # but dynesty [below] will give:
    # {'D18':    realisation    avg       std       max     posterior
    #    0         0          1.641669  0.277169  1.279457 -14.720054
    #    1         0          1.643549  0.271477  1.374330 -14.823846}

    fitter.add_model_and_dataset(beutler, d, name=f"D18", linestyle="-", color="p", realisation=0)
    fitter.add_model_and_dataset(ding, d, name=f"D18", linestyle="-", color="p", realisation=0)
           
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.fit(file)

    model_results, summary = get_model_comparison_dataframe(fitter)
    print(model_results)

