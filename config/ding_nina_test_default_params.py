import sys, os
sys.path.append("..")
from barry.cosmology.camb_generator import getCambGenerator
from barry.postprocessing import BAOExtractor
from barry.config import setup
from barry.utils import weighted_avg_and_std, get_model_comparison_dataframe
from barry.models import PowerBeutler2017, CorrBeutler2017
from barry.datasets import PowerSpectrum_SDSS_DR12, PowerSpectrum_AbacusSummit, CorrelationFunction_AbacusSummit
from barry.samplers import Optimiser
from barry.fitter import Fitter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import pdb
import logging


RECON_BASE_DIR = "/mnt/marvin1/nam/scratch/data_mocks_summit_new/"
Z = "z0.800"

if __name__ == "__main__":

    assert(len(sys.argv)==2)
    phase = sys.argv[1]

    logging.basicConfig(filename=f"logs/barrylog_fixedParams_{Z}_ph{str(phase).zfill(3)}.log",
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

    logger = logging.getLogger('nam-barry-defpars')

    pfn, dir_name, file = setup(__file__)
    fitter = Fitter(dir_name)
    sampler = Optimiser(temp_dir = dir_name)

# vary fixed params +- 20 percent. 
    FIX_PARAMS = {'sigma_s': [3.2,3.6,4,4.4,4.8],  'beta': [0.32,0.36,0.4,0.44,0.48],\
                  'sigma_nl_par': {'ani': [8,9,10,11,12], 'none': [3.2,3.6,4,4.4,4.8]},\
                  'sigma_nl_perp' :{'ani': [4.8,5.4,6,6.6,7.2], 'none': [2.0,2.25,2.5,2.75,3]}} 

    for tracer in ["LRG_rd", "ELG_rd"]:
          
        sim_name = 'AbacusSummit_base_c000_ph'+str(phase).zfill(3) 
            
        recon_dir = f"{RECON_BASE_DIR}/{sim_name}/{Z}/recon/{tracer}"
        hod_recon_models = ['Baseline_NoBias']

        for hod_recon_model in hod_recon_models:
            for recon in ['iso', None]:
                for p in FIX_PARAMS.keys():
                    if p not in ['sigma_nl_par', 'sigma_nl_perp']:
                        fixed_vals = FIX_PARAMS[p]
                    else:
                        r_string = 'none' if recon is None else recon
                        fixed_vals = FIX_PARAMS[p][r_string] 

                    for val in fixed_vals: 


                        logging.info(f"====== STARTING BARRY for {sim_name} {Z} {tracer} {hod_recon_model} recon: {recon} ======")

                        data = CorrelationFunction_AbacusSummit(box = sim_name, redshift = Z, recon = recon,\
                                                min_dist=55, field_type = tracer, hod_recon_model = hod_recon_model,\
                                                isotropic=False, fit_poles=[0, 2])

                        model = CorrBeutler2017(recon=data.recon, isotropic=data.isotropic,\
                                                marg="full", poly_poles=[0, 2],\
                                                fix_params=["om", "sigma_s", "beta", "sigma_nl_par", "sigma_nl_perp"]) #fix_params=["om"], 

                        model.set_default(p, val)
#                        print(f"{sim_name}_{Z}_{tracer}_{hod_recon_model}_recon-{recon}_fixParams_{p}_{val}_xi")
                        fitter.add_model_and_dataset(model, data, name=f"{sim_name}_{Z}_{tracer}_{hod_recon_model}_recon-{recon}_fixParams_{p}_{val}_xi")

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.fit(file)

