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
import argparse


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

def process_params(mcyc, nopoly, fixbs, marg, name, fix_params, recons, tracers):
    if marg is None: #detect marg setting automatically
        marg = None if (nopoly or fixbs) else 'full'
    else: #use user-input, but check that we're allowed to do marg (can't if either nopoly or fixbs is true)
        marg = 'full' if (marg and not (nopoly or fixbs)) else None 

    if name is None:
        name = f"nopoly_{nopoly}_fixbs_{fixbs}_marg_{marg}"

    else:
        name = f"nopoly_{nopoly}_fixbs_{fixbs}_marg_{marg}_{name}"

    if fix_params is None:
        fix_params = ['om']

    if recons is None:
        recons = [None, 'ani']

    if tracers is None:
        tracers = ["ELG_rd", "LRG_rd"]

    return marg, name, fix_params, recons, tracers

def main(phase, mcyc, nopoly, fixbs, marg, name, fix_params, recons, tracers):

    marg, name, fix_params, recons, tracers = process_params(mcyc, nopoly, fixbs, marg, name, fix_params, recons, tracers)

    firstsim = 'AbacusSummit_base_c000_ph'+str(phase).zfill(3) 
    sim_set_name = f"CYC_{firstsim}_ph"+str((int(phase)+mcyc-1)%25).zfill(3)

    print(name, phase, mcyc, sim_set_name, fix_params, recons, tracers)
    pdb.set_trace()

    # logging.basicConfig(filename=f"logs/barrylog_{Z}_{sim_set_name}.log",
    #                         filemode='a',
    #                         format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    #                         datefmt='%H:%M:%S',
    #                         level=logging.INFO)

    # stdout_logger = logging.getLogger('STDOUT')
    # sl = StreamToLogger(stdout_logger, logging.INFO)
    # sys.stdout = sl

    # stderr_logger = logging.getLogger('STDERR')
    # sl = StreamToLogger(stderr_logger, logging.ERROR)
    # sys.stderr = sl
    
    pfn, dir_name, file = setup(__file__)
    fitter = Fitter(dir_name)
    sampler = Optimiser(temp_dir = dir_name)

    for tracer in tracers: 
        recon_dir = f"{RECON_BASE_DIR}/{firstsim}/{Z}/recon/{tracer}"
        hod_recon_models = os.listdir(recon_dir) 
        assert(set(hod_recon_models) == set(hod_recon_models))
        hod_recon_models = ["Baseline_NoBias"]
        print("HACK HACK HACK", hod_recon_models)
        for hod_recon_model in hod_recon_models:
 
            for recon in recons: 
            # for recon in ['ani', None]:
                logging.info(f"====== STARTING BARRY for {sim_set_name} {Z} {tracer} {hod_recon_model} recon: {recon}, marg: {marg} ======")

                data = CorrelationFunction_AbacusSummit(box = sim_set_name, redshift = Z, recon = recon,\
                                        min_dist=55, field_type = tracer, hod_recon_model = hod_recon_model,\
                                        isotropic=False, fit_poles=[0, 2])

                model = CorrBeutler2017(recon=data.recon, isotropic=data.isotropic,\
                                        marg=marg, poly_poles=[0, 2],  tracer = tracer, no_poly = nopoly, fix_bs_to_b0 = fixbs,\
                                        fix_params=fix_params) #fix_params=["om"], 

                
                minv, p = model.sanity_check(data, figname = f"{name}_{sim_set_name}_{Z}_{tracer}_{hod_recon_model}_recon-{recon}.png", plt_errs = True)
                logging.info(f"Model optimisation with value {minv:0.3f} has parameters are {dict(p)}")


                # pdb.set_trace()
                # fitter.add_model_and_dataset(model, data, name=f"{sim_set_name}_{Z}_{tracer}_{hod_recon_model}_recon-{recon}_analyticCov_sigmaSfloat_xi")

    # fitter.set_sampler(sampler)
    # fitter.set_num_walkers(1)
    # fitter.fit(file)


class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":

    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('phase', type=int, help='Which phase to fit (or starting phase in cyclic set)')
    parser.add_argument('--mcyc', help='Number of boxes to cyclically co-add', default=13, type=int)
    parser.add_argument('--nopoly', help='nopoly = True turns off polynomial terms', default=False, type=bool)
    parser.add_argument('--fixbs', help='fixbs = True fixes b2 to b0', default=False, type=bool)
    parser.add_argument('--marg', help='marg = True forces full analytical marginalisation', default=None, type=bool)
    parser.add_argument('--name', help='Name to label output files', default=None)
    parser.add_argument('--fix_params', nargs='+', help='List of fit params to fix', default=None)
    parser.add_argument('--recons', nargs='+', help='Which recon types to fit', choices=['None', 'ani'], default=None)
    parser.add_argument('--tracers', nargs='+', help='Which tracers to fit', choices=['ELG_rd', 'LRG_rd'], default=None)

    args = vars(parser.parse_args())

    main(**args)





