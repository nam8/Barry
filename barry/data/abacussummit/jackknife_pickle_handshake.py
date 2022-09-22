import pickle5 as pickle
import pandas as pd
import numpy as np
import os
from pathlib import Path
import pdb
from scipy.io import mmread, mmwrite

# TODO : stick this into some global config file, so there's no duplication b/w
# hod, recon, and barry

# directory with results
RESULTS_BASE_DIR = "/mnt/marvin1/nam/scratch/data_mocks_summit_new"

REDSHIFT = "z0.800"
Z = float(REDSHIFT.split("z")[1]) 

# match Chris' output keys to barry's expected keys
keys = { ("particles",  "delta") :                 "pre-recon data" ,
         ("particles",  "delta-reconstructed") :  "post-recon data" ,
         ("mocks", "delta") :                      "pre-recon mocks",
         ("mocks", "delta-reconstructed") :       "post-recon mocks",
       }  

def emptyresdict():
    res = { "pre-recon data": None, 
           "post-recon data": None, 
           "pre-recon mocks": None,
          "post-recon mocks": None
      }
    return res 
    
def getxi(loc):
    res = emptyresdict() 

    for deltatype in ["delta-reconstructed", "delta"]:

        bias =  recon_res['info']['recon config']['bias'] if deltatype is "delta" else 1.0 

        df = pd.DataFrame(columns = ["s", "xi0", "xi2", "xi4"])
        df["s"]   = recon_res["xi"][deltatype]["r"]
        df["xi0"] = recon_res["xi"][deltatype]["xi"][:,0] * bias**2
        df["xi2"] = recon_res["xi"][deltatype]["xi"][:,1] * bias**2
        df["xi4"] = 0
        mask   = df["s"] <= 200.0
        masked = df.loc[mask, ["s", "xi0", "xi2", "xi4"]]
        res[keys[("mocks", deltatype)]] = masked.astype(np.float32)
    return res


def getwin(ks):
    res = {"w_ks_input": ks.copy(), "w_k0_scale": np.zeros(ks.size), "w_transform": np.eye(5 * ks.size), "w_ks_output": ks.copy()}
    return {1: res}  # Step size is one


def getcomp(ks):
    matrix = np.zeros((5 * ks.size, 3 * ks.size))
    matrix[: ks.size, : ks.size] = np.diag(np.ones(ks.size))
    matrix[2 * ks.size : 3 * ks.size, ks.size : 2 * ks.size] = np.diag(np.ones(ks.size))
    matrix[4 * ks.size :, 2 * ks.size :] = np.diag(np.ones(ks.size))
    return matrix

def cyclic_set_results(all_box_results):
    # expected structure: 
    # recon_res = {'info': {}, 'xi': {'delta-reconstructed': {}, 'delta': {}}}, 'ps': {'delta-reconstructed': {}, 'delta': {}}} }
    avg_res = {'info': None, 'xi': {'delta-reconstructed': {'r': None, 'xi': None}, 'delta': {'r': None, 'xi': None}}}

    # make sure all results were run with the same HOD, recon, redshift, tracer, etc. parameters. 
    # also that all results' r bins for xi are the same. 
    i = 0
    for box in all_box_results.keys():
        if i == 0:
            firstbox = box
        elif i == M_CYC-1:
            lastbox = box
        print(f"Comparing {firstbox} and {box}")

        for d in ["delta-reconstructed", "delta"]:
            assert np.array_equal(all_box_results[firstbox]['xi'][d]['r'], all_box_results[box]['xi'][d]['r'])

        for key in all_box_results[box]['info'].keys():
            if key != 'simname':
                assert all_box_results[firstbox]['info'][key] == all_box_results[box]['info'][key]
        i = i+1

    # if we've passed the asserts above, then we're safe to set avg_res info to first box's info. 
    avg_res['info'] = all_box_results[firstbox]['info']
    # but we do need to update the simname.s
    lastphase = lastbox.split("_")[-1]
    avg_res['info']['simname'] = f"CYC_{firstbox}_{lastphase}"


    # okay! Now let's average the xis. (not going to bother with power spectra, at least for now)
    # this selects the monopole, quadrupole, r's

    for d in ["delta-reconstructed", "delta"]:
        avg_res['xi'][d]['r'] = all_box_results[firstbox]['xi'][d]['r'] 

        xi_0s = np.empty((M_CYC, len(avg_res['xi'][d]['r'])))
        xi_2s = np.empty((M_CYC, len(avg_res['xi'][d]['r'])))
        b = 0
        for box in all_box_results.keys():
            xi_0s[b] = np.array(all_box_results[box]['xi'][d]['xi'][:,0])
            xi_2s[b] = np.array(all_box_results[box]['xi'][d]['xi'][:,1])
            b = b+1

        avg_res['xi'][d]['xi'] = np.empty((len(avg_res['xi'][d]['r']) , 2))

        avg_res['xi'][d]['xi'][:,0] = np.mean(xi_0s, axis=0) 
        avg_res['xi'][d]['xi'][:,1] = np.mean(xi_2s, axis=0)

    # NOTE! this is not the place to compare averaged results to previous fits, because the 
    # bias^2 multiplication for pre-recon xis occurs in getxi() above. 
    # recall that we added this b/c chris' code only multiples by bias**2 for reconstructed results. 

    return avg_res

M_CYC = 13

if __name__ == "__main__":
    # we are going to do a modified, cyclic jacknife. Loop over all boxes. Treat box phase as starting box of cyclic set w/ M_CYC boxes. 
    for s in range(0, 25): 
        phases = [(s + i)%25 for i in range(M_CYC)]
        sims = ['AbacusSummit_base_c000_ph'+str(phase).zfill(3) for phase in phases] 

        for dataset in ["ELG_rd", "LRG_rd"]: 
            dataset_dirs = [os.path.join(RESULTS_BASE_DIR, sim, REDSHIFT, "recon", dataset) for sim in sims]
            hod_recon_model_dirs = [os.listdir(dataset_dir) for dataset_dir in dataset_dirs]

            for i in range(M_CYC-1):
                assert set(hod_recon_model_dirs[i]) == set(hod_recon_model_dirs[i+1])
                assert len(hod_recon_model_dirs[i]) == len(hod_recon_model_dirs[i+1])
            print(hod_recon_model_dirs[0], "HACK HACK HACK")
            hod_recon_model_dirs[0] = ['gaussian_sigma_7']
            for hod_recon_model in hod_recon_model_dirs[0]: # if we passed the asserts above, we can just use one box's list of HOD models. 
                model_dirs = [os.path.join(dataset_dir, hod_recon_model) for dataset_dir in dataset_dirs] 
		
                all_box_files = [[os.path.join(model_dir, f) for f in os.listdir(model_dir) if "pickle" in f] for model_dir in model_dirs] 
                assert(len(all_box_files) == M_CYC)
                print(f"Processing {len(all_box_files)} files: {all_box_files}")

                all_box_res = {}
                # cyclic jackknifing only applies to mocks. 
                # load all pickles 
                i = 0
                for this_box_files in all_box_files:
                    for fn in this_box_files:
                        with open(fn, 'rb') as handle:
                            all_box_res[f"{sims[i]}"] = pickle.load(handle)
                    i = i+1

                # make dictionary that combines the M_CYC mocks. Average xis. 
                recon_res = cyclic_set_results(all_box_res)

                cosmology = {
                # AbacusSummit read the docs says:
                # remember that Omega_M = (om_b + om_cdm + om_nu ) / h^2 
                # therefore om_b = Om_b * h^2 
                # so we copy/paste the AS values here, directly. 
                # Omega_Nu (Ncdm) * h^2 = sum(mi) / 93.14 eV 
                # for base boxes, we only have one N_Ncdm and 2 N_ur
                        "om": (0.12 + 0.02237 + 0.0006442) / 0.6736 ** 2,
                        "h0": 0.6736,
                        "z": Z,
                        "ob": 0.02237 / 0.6736 ** 2,
                        "ns": 0.9649,
                        "mnu": 0.0006442 * 93.14, 
                        "reconsmoothscale": recon_res['info']['recon config']['gaussian_sigma'], 
                        }

                base_pickle_name = f"{recon_res['info']['simname']}_{REDSHIFT}_{dataset}_{hod_recon_model}"
                xi_pickle_name = f"{base_pickle_name}_xi.pkl"

                if not os.path.exists( f"{REDSHIFT}/{recon_res['info']['simname']}/{dataset}/"):
                    os.makedirs(f"{REDSHIFT}/{recon_res['info']['simname']}/{dataset}/")

                ss = recon_res["xi"]["delta"]["r"]

                cov_pre_rec_xi02  = mmread(f"xi_cov_analytic_{REDSHIFT}_ph000_pre_recon.mtx")
                cov_post_rec_xi02 = mmread(f"xi_cov_analytic_{REDSHIFT}_ph000_post_recon.mtx")

                assert(np.shape(cov_post_rec_xi02) == (2*len(ss), 2*len(ss)))
                assert(np.shape( cov_pre_rec_xi02) == (2*len(ss), 2*len(ss)))

                cov_pre_rec_full  = np.zeros((3 * len(ss), 3 * len(ss)))
                cov_post_rec_full = np.zeros((3 * len(ss), 3 * len(ss)))
                
                cov_pre_rec_full[ :2*len(ss), :2*len(ss)] = cov_pre_rec_xi02/M_CYC
                cov_post_rec_full[:2*len(ss), :2*len(ss)] = cov_post_rec_xi02/M_CYC

                res = getxi(recon_res)

                split = {
                    "pre-recon data":   None ,
                    "pre-recon cov":    cov_pre_rec_full,
                    "post-recon data":  None ,
                    "post-recon cov":   cov_post_rec_full,
                    "pre-recon mocks":  [res["pre-recon mocks"]] ,
                    "post-recon mocks": [res["post-recon mocks"]] ,
                    "cosmology": cosmology,
                    "name": f"{recon_res['info']['simname']}, {REDSHIFT}, {dataset}, {hod_recon_model}, correlation function",
                    "info": recon_res['info'],
                }

                print(f"Writing pickle to {REDSHIFT}/{recon_res['info']['simname']}/{dataset}/{xi_pickle_name}")
                with open(f"{REDSHIFT}/{recon_res['info']['simname']}/{dataset}/{xi_pickle_name}", "wb") as f:
                    pickle.dump(split, f)

