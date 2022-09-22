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
# BOX = "AbacusSummit_base_c000_ph000"
# TRACER = 'LRG_rd'

if __name__ == "__main__":

    # logging.basicConfig(filename="z0.8-barry.log",
                            # filemode='a',
                            # format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            # datefmt='%H:%M:%S',
                            # level=logging.INFO)

    # logger = logging.getLogger('nam-barry')

    pfn, dir_name, file = setup(__file__)
    fitter = Fitter(dir_name)
    sampler = Optimiser(temp_dir = dir_name)

    for tracer in ["LRG_rd", "ELG_rd"]:
        for phase in range(0,1): # only have barry pickles for ph000. will need to organise directories by box...?
            sim_name = 'AbacusSummit_base_c000_ph'+str(phase).zfill(3) 
            
            recon_dir = f"{RECON_BASE_DIR}/{sim_name}/{Z}/recon/{tracer}"
            hod_recon_models = os.listdir(recon_dir) 

#            hod_recon_models = ['gaussian_sigma_15', 'gaussian_sigma_20', 'gaussian_sigma_25', 'Baseline_NoBias']

            for hod_recon_model in hod_recon_models:

                for recon in ['iso', None]:
                    logging.info(f"====== STARTING BARRY for {sim_name} {Z} {tracer} {hod_recon_model} recon: {recon} ======")

                    data = CorrelationFunction_AbacusSummit(box = sim_name, redshift = Z, recon = recon,\
                                        min_dist=55, field_type = tracer, hod_recon_model = hod_recon_model,\
                                        isotropic=False, fit_poles=[0, 2])

                    model = CorrBeutler2017(recon=data.recon, isotropic=data.isotropic,\
                                        marg="full", poly_poles=[0, 2],\
                                        fix_params=["om", "sigma_s", "beta", "sigma_nl_par", "sigma_nl_perp"]) #fix_params=["om"], 

#                    minv, p = model.sanity_check(data, figname = f"{sim_name}_{Z}_{tracer}_{hod_recon_model}_recon-{recon}_analyticCov_xi.png", plt_errs = True)
#                    logging.info(f"Model optimisation with value {minv:0.3f} has parameters are {dict(p)}")

                    fitter.add_model_and_dataset(model, data, name=f"{sim_name}_{Z}_{tracer}_{hod_recon_model}_recon-{recon}_analyticCov_xi")

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.fit(file)

    exit(1)

    # sdss_pre_recon_zbin2      = PowerSpectrum_SDSS_DR12(recon=None,  realisation=0, redshift_bin=2) 
    # sdss_post_recon_iso_zbin2 = PowerSpectrum_SDSS_DR12(recon="iso", realisation=0, redshift_bin=2) #z-bin 2: z=0.51

    # realisation integer (0, 1, etc) can be used to index into which mock catalogue you want.
    # realisation = "data" is particles. 
    nrandom = ["nrandom1e7", "nrandom1e8", "nrandom1e9"]
    # nrandom = ["nrandom1e8"]
    for field_type in ["ELG", "LRG"]:
        reconstr = 'ani' if "rd" in field_type else 'iso'
        abacus_datasets = []
        abacus_no_recon = []
        
        for n in nrandom: 
            # abacus_no_recon.append(PowerSpectrum_AbacusSummit(box = BOX, redshift = Z, recon = None,     recon_nrandom = n, field_type= field_type))
            # abacus_datasets.append(PowerSpectrum_AbacusSummit(box = BOX, redshift = Z, recon = reconstr, recon_nrandom = n, field_type= field_type)) 
            abacus_no_recon.append(CorrelationFunction_AbacusSummit(box = BOX, redshift = Z, recon = reconstr, recon_nrandom = n, field_type = field_type))
            abacus_datasets.append(CorrelationFunction_AbacusSummit(box = BOX, redshift = Z, recon = None, recon_nrandom = n, field_type = field_type))
    #############
    #     fig, ax = plt.subplots() 
    #     ref = abacus_datasets[-1]
    #     for n in range(len(nrandom) - 1):
    #         d = abacus_datasets[n]
    #         ax.scatter(d.ks, d.mock_data[0][:,0] / ref.mock_data[0][:,0], label=nrandom[n], s = 1)
    #     plt.axhline(y=1.0, color='k', linestyle='--')
    #     ax.legend() 
    #     # ax.set_xlim(0,0.35)
    #     # ax.set_ylim(0.9, 2.6)
    #     ax.set_title(f'{BOX} {Z} {field_type} {reconstr}. recon\n' + r'nrandom$_{i}$ / nrandom$_{ref}$')
    #     ax.set_xlabel(r'k (Mpc $h^{-1}$)')
    #     ax.set_ylabel(r'$P_0$(k)$_i$ / $P_0$(k)$_{ref}$ (nrandom = 1e9)')
    #     plt.savefig(f'{BOX}_{Z}_{field_type}_Pk_nrandom_ratio.png')

    # ############
    #     fig, ax = plt.subplots() 
    #     for n in range(len(nrandom)):
    #         d = abacus_datasets[n]
    #         ax.scatter(abacus_no_recon[n].ks, abacus_no_recon[n].ks * abacus_no_recon[n].mock_data[0][:,0] , label="No recon" if n == len(nrandom) - 1 else None, s = 1)
    #         ax.scatter(d.ks, d.ks * d.mock_data[0][:,0] , label=nrandom[n], s = 1)

    #     ax.set_yscale('log')
    #     ax.legend() 
    #     # ax.set_xlim(0,0.35)
    #     ax.set_title(f'{BOX} {Z} {field_type} {reconstr}. recon\nall nrandoms')
    #     ax.set_xlabel(r'k (Mpc $h^{-1}$)')
    #     ax.set_ylabel(r'k x $P_0$(k)')

    #     figname = f'{BOX}_{Z}_{field_type}_kPk_nrandom_all.png'
    #     print(f"Saving {figname}.")
    #     plt.savefig(figname)

    #############


        # beutler    = PowerBeutler2017(recon = reconstr)
        beutler = CorrBeutler2017( recon = reconstr)

    #     # from scipy import interpolate
    #     # # f_abac_pre_recon_iso = interpolate.interp1d(d_abac_pre_recon_iso.ks, d_abac_pre_recon_iso.true_data[0][:,0])
    #     # f_sdss_pre_recon  = interpolate.interp1d(d_sdss_pre_recon_zbin2.ks,   d_sdss_pre_recon_zbin2.true_data[0][:,0])
    #     # f_sdss_post_recon = interpolate.interp1d(d_sdss_post_recon_iso_zbin2.ks, d_sdss_post_recon_iso_zbin2.true_data[0][:,0])
    #     # knew = np.arange(0.002, 0.85, 0.0005)
    #     # # abac_pre_recon_iso_new = f_abac_pre_recon_iso(knew)
    #     # sdss_interp_pre_recon  =  f_sdss_pre_recon(knew)
    #     # sdss_interp_post_recon = f_sdss_post_recon(knew)

        for n in range(len(nrandom)):
            # d = abacus_datasets[n]
            # beutler.sanity_check(d, figname = f"{BOX}_{Z}_{field_type}_{nrandom[n]}.png")
            d = abacus_datasets[n]
            # xi_beutler.sanity_check(d, figname = f"{BOX}_{Z}_{field_type}_{nrandom[n]}_xi.png", plt_errs = False)

            fitter.add_model_and_dataset(beutler, d, name=f"{BOX}_{Z}_{field_type}_{nrandom[n]}_B17")
               
        fitter.set_sampler(sampler)
        fitter.set_num_walkers(3)
        fitter.fit(file)

        # model_results, summary = get_model_comparison_dataframe(fitter) 
        # print(model_results)

