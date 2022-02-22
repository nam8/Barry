import sys

sys.path.append("..")
from barry.cosmology.camb_generator import getCambGenerator
from barry.postprocessing import BAOExtractor
from barry.config import setup
from barry.utils import weighted_avg_and_std, get_model_comparison_dataframe
from barry.models import PowerBeutler2017
from barry.datasets import PowerSpectrum_SDSS_DR12, PowerSpectrum_AbacusSummit
from barry.samplers import Optimiser
from barry.fitter import Fitter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate


Z = "z0.500"
BOX = "AbacusSummit_base_c000_ph000"

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)
    fitter = Fitter(dir_name)
    sampler = Optimiser(temp_dir = dir_name)

    # sdss_pre_recon_zbin2      = PowerSpectrum_SDSS_DR12(recon=None,  realisation=0, redshift_bin=2) 
    sdss_post_recon_iso_zbin2 = PowerSpectrum_SDSS_DR12(recon="iso", realisation=0, redshift_bin=2) #z-bin 2: z=0.51

    # realisation integer (0, 1, etc) can be used to index into which mock catalogue you want.
    # realisation = "data" is particles. 
    nrandom = ["nrandom1e07", "nrandom1e08", "nrandom1e09", "nrandom1e10"]
    abacus_datasets = []
    for n in nrandom: 
        abacus_datasets.append(PowerSpectrum_AbacusSummit(box = BOX, redshift = Z, recon_nrandom = n, realisation="data")) 

#############
    fig, ax = plt.subplots() 
    ref = abacus_datasets[-1]
    for n in range(len(nrandom) - 1):
        d = abacus_datasets[n]
        ax.scatter(d.ks, d.true_data[0][:,0] / ref.true_data[0][:,0], label=nrandom[n], s = 1)
    plt.axhline(y=1.0, color='k', linestyle='--')
    ax.legend() 
    ax.set_xlim(0,0.35)
    ax.set_ylim(0.9, 2.6)
    ax.set_title(f'{BOX} {Z} iso. recon\n' + r'nrandom$_{i}$ / nrandom$_{ref}$')
    ax.set_xlabel(r'k (Mpc $h^{-1}$)')
    ax.set_ylabel(r'$P_0$(k)$_i$ / $P_0$(k)$_{ref}$ (nrandom = 1e10)')
    plt.savefig(f'{BOX}_{Z}_Pk_nrandom_ratio.png')

#############
    fig, ax = plt.subplots() 
    for n in range(len(nrandom)):
        d = abacus_datasets[n]
        ax.scatter(d.ks, d.true_data[0][:,0] , label=nrandom[n], s = 1)

    ax.set_yscale('log')
    ax.legend() 
    ax.set_xlim(0,0.35)
    ax.set_title(f'{BOX} {Z} iso. recon\nall nrandoms')
    ax.set_xlabel(r'k (Mpc $h^{-1}$)')
    ax.set_ylabel(r'$P_0$(k)')
    plt.savefig(f'{BOX}_{Z}_Pk_nrandom_all.png')

#############


    beutler = PowerBeutler2017(recon='iso')

    # from scipy import interpolate
    # # f_abac_pre_recon_iso = interpolate.interp1d(d_abac_pre_recon_iso.ks, d_abac_pre_recon_iso.true_data[0][:,0])
    # f_sdss_pre_recon  = interpolate.interp1d(d_sdss_pre_recon_zbin2.ks,   d_sdss_pre_recon_zbin2.true_data[0][:,0])
    # f_sdss_post_recon = interpolate.interp1d(d_sdss_post_recon_iso_zbin2.ks, d_sdss_post_recon_iso_zbin2.true_data[0][:,0])
    # knew = np.arange(0.002, 0.85, 0.0005)
    # # abac_pre_recon_iso_new = f_abac_pre_recon_iso(knew)
    # sdss_interp_pre_recon  =  f_sdss_pre_recon(knew)
    # sdss_interp_post_recon = f_sdss_post_recon(knew)

    for n in range(len(nrandom)):
        d = abacus_datasets[n]
        # beutler.sanity_check(d, figname = f"{BOX}_{Z}_{nrandom[n]}.png")
        fitter.add_model_and_dataset(beutler, d, name=f"B17 nrandom {nrandom[n]}")
           
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(3)
    fitter.fit(file)

    # for posterior, weight, chain, evidence, model, data, extra in fitter.load():


    model_results, summary = get_model_comparison_dataframe(fitter)
    print(model_results)

