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
#source: http://techies-world.com/how-to-redirect-stdout-and-stderr-to-a-logger-in-python/
class StreamToLogger(object):
      """
      Fake file-like stream object that redirects writes to a logger instance.
      """
      def __init__(self, logger, log_level=logging.INFO):
            self.logger = logger
            self.log_level = log_level
            self.linebuf = ''

      def write(self, buf):
            for line in buf.rstrip().splitlines():
                  self.logger.log(self.log_level, line.rstrip())


M_CYC = 13

if __name__ == "__main__":
    assert(len(sys.argv)==2)
    phase = sys.argv[1]

    firstsim = 'AbacusSummit_base_c000_ph'+str(phase).zfill(3) 
    sim_set_name = f"CYC_{firstsim}_ph"+str((int(phase)+M_CYC-1)%25).zfill(3)

    logging.basicConfig(filename=f"logs/barrylog_{Z}_{sim_set_name}.log",
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

    stdout_logger = logging.getLogger('STDOUT')
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger('STDERR')
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl
    
    pfn, dir_name, file = setup(__file__)
    fitter = Fitter(dir_name)
    sampler = Optimiser(temp_dir = dir_name)

    for tracer in ["ELG_rd", "LRG_rd"]:
        recon_dir = f"{RECON_BASE_DIR}/{firstsim}/{Z}/recon/{tracer}"
        hod_recon_models = os.listdir(recon_dir) 
        assert(set(hod_recon_models) == set(hod_recon_models))

        print(hod_recon_models)
        hod_recon_models = ['gaussian_sigma_7']

        for hod_recon_model in hod_recon_models:

            for recon in ['iso', None]:
                logging.info(f"====== STARTING BARRY for {sim_set_name} {Z} {tracer} {hod_recon_model} recon: {recon} ======")

                data = CorrelationFunction_AbacusSummit(box = sim_set_name, redshift = Z, recon = recon,\
                                        min_dist=55, field_type = tracer, hod_recon_model = hod_recon_model,\
                                        isotropic=False, fit_poles=[0, 2])

                model = CorrBeutler2017(recon=data.recon, isotropic=data.isotropic,\
                                        marg="full", poly_poles=[0, 2], tracer = tracer,\
                                        fix_params=["om", "sigma_s", "beta", "sigma_nl_par", "sigma_nl_perp"]) #fix_params=["om"], 

                
                # minv, p = model.sanity_check(data, figname = f"{sim_set_name}_{Z}_{tracer}_{hod_recon_model}_recon-{recon}_analyticCov_NoCovDivM_cyc_xi.png", plt_errs = True)
#                logging.info(f"Model optimisation with value {minv:0.3f} has parameters are {dict(p)}")

                fitter.add_model_and_dataset(model, data, name=f"{sim_set_name}_{Z}_{tracer}_{hod_recon_model}_recon-{recon}_analyticCov_xi")

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.fit(file)

