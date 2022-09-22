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

    for dataset in recon_res.keys(): # particles, mocks. 
        for deltatype in ["delta-reconstructed", "delta"]:

            # pdb.set_trace()

            bias =  recon_res[dataset]['info']['recon config']['bias'] if deltatype is "delta" else 1.0         

            df = pd.DataFrame(columns = ["s", "xi0", "xi2", "xi4"])
            df["s"]   = recon_res[dataset]["xi"][deltatype]["r"]
            df["xi0"] = recon_res[dataset]["xi"][deltatype]["xi"][:,0] * bias**2
            df["xi2"] = recon_res[dataset]["xi"][deltatype]["xi"][:,1] * bias**2
            df["xi4"] = 0
            mask   = df["s"] <= 200.0
            masked = df.loc[mask, ["s", "xi0", "xi2", "xi4"]]
            res[keys[(dataset, deltatype)]] = masked.astype(np.float32)
    return res


def getpk(recon_res):
    res = emptyresdict()

    for dataset in recon_res.keys():
        for deltatype in ["delta-reconstructed", "delta"]:
            df = pd.DataFrame(columns = ["k", "pk0", "pk1", "pk2", "pk3", "pk4"])

            # pdb.set_trace()

            bias =  recon_res[dataset]['info']['recon config']['bias'] if deltatype is "delta" else 1.0 

            df["k"]   = recon_res[dataset]["ps"][deltatype]["k"]
            df["pk0"] = recon_res[dataset]["ps"][deltatype]["ps"][:,0] * bias**2
            df["pk1"] = 0
            df["pk2"] = recon_res[dataset]["ps"][deltatype]["ps"][:,1] * bias**2
            df["pk3"] = 0
            df["pk4"] = 0
            mask   = df["k"] <= 0.5
            masked = df.loc[mask, ["k", "pk0", "pk1", "pk2", "pk3", "pk4"]]
            res[keys[(dataset, deltatype)]] = masked.astype(np.float32)

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


if __name__ == "__main__":
    # TODO: in Barry/config/pk_abacussummit_handshake.py, you can actually pass lots of different
    # mock realisations for the same box. 
    # that's why barry expects a list of mocks in post/pre-recon-mocks.
    # so adapt this once you have more mocks! to read in all mocks for a given box. 
    for phase in range(0, 25): # only have barry pickles for ph000. will need to organise directories by box...?
        sim = 'AbacusSummit_base_c000_ph'+str(phase).zfill(3) 

        for dataset in ["ELG_rd",'LRG_rd']:
            dataset_dir = os.path.join(RESULTS_BASE_DIR, sim, REDSHIFT, "recon", dataset)
            hod_recon_model_dirs = os.listdir(dataset_dir)

            for hod_recon_model in hod_recon_model_dirs: 
                model_dir = os.path.join(dataset_dir, hod_recon_model)
                # find all pickles! one pickle per data type (e.g. particles? or galaxies?). each pickle contains pre and post recon PSs and Xis. 
                files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if "pickle" in f]
                print(f"Processing: {files}")

                recon_res = {} 
                # will want to load all pickles, both data and mocks, and then use them to populate dictionary below.
                have_particles = False
                have_mocks = False 
                for fn in files:
                    with open(fn, 'rb') as handle:
                        if "particles" in os.path.basename(fn): 
                            recon_data_res  = pickle.load(handle)
                            recon_res["particles"] = recon_data_res
                            have_particles = True
                        elif "ELG" in os.path.basename(fn) or "LRG" in os.path.basename(fn): 
                            recon_mocks_res = pickle.load(handle)
                            recon_res["mocks"] = recon_mocks_res
                            have_mocks = True

                # for every dict in recon_res (data, mocks)
                #     for every ps in dict (delta, delta_recon)
                #         assert that ks are the same as previous dict. 
                ks = recon_res["mocks"]["ps"]["delta"]["k"]
                for res_key in recon_res.keys():
                    res_value_ps = recon_res[res_key]["ps"]
                    # res_key takes values "particles", "mocks"
                    for ps_key in res_value_ps.keys(): 
                        # ps_key takes values "delta", "delta_recon"
                        assert np.array_equal(ks, res_value_ps[ps_key]["k"]), "All power spectra ks should be the same! (for {particles/mocks}/{delta/delta_recon})"
                    


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
                        "reconsmoothscale": recon_res['mocks']['info']['recon config']['gaussian_sigma'], 
                        }


                base_pickle_name = f"{sim}_{REDSHIFT}_{dataset}_{hod_recon_model}"
                ps_pickle_name = f"{base_pickle_name}_ps.pkl"
                xi_pickle_name = f"{base_pickle_name}_xi.pkl"

                if not os.path.exists( f"{REDSHIFT}/{sim}/{dataset}/"):
                    os.makedirs(f"{REDSHIFT}/{sim}/{dataset}/")

                ss = recon_res["mocks"]["xi"]["delta"]["r"]
                # xi is xi0, xi2, xi4 (no odd terms!)
                # set cov for multipoles 0 and 2 
                # cov = np.zeros((3 * len(ss), 3 * len(ss)))
                # cov[:len(ss),                       :len(ss)] = 0.01 * np.eye(len(ss), len(ss))
                # cov[len(ss):2*len(ss),     len(ss):2*len(ss)] = np.eye(len(ss), len(ss))
                # cov[2*len(ss):3*len(ss), 2*len(ss):3*len(ss)] = np.eye(len(ss), len(ss))

                cov_pre_rec_xi02  = mmread(f"xi_cov_analytic_{REDSHIFT}_ph000_pre_recon.mtx")
                cov_post_rec_xi02 = mmread(f"xi_cov_analytic_{REDSHIFT}_ph000_post_recon.mtx")

                assert(np.shape(cov_post_rec_xi02) == (2*len(ss), 2*len(ss)))
                assert(np.shape( cov_pre_rec_xi02) == (2*len(ss), 2*len(ss)))

                cov_pre_rec_full  = np.zeros((3 * len(ss), 3 * len(ss)))
                cov_post_rec_full = np.zeros((3 * len(ss), 3 * len(ss)))
                
                cov_pre_rec_full[ :2*len(ss), :2*len(ss)] = cov_pre_rec_xi02
                cov_post_rec_full[:2*len(ss), :2*len(ss)] = cov_post_rec_xi02


#                cov = np.eye(3 * len(ss))
#                cov[2*len(ss):3*len(ss), 2*len(ss):3*len(ss)] = 10**10 * np.eye(len(ss))
#                cov_pre_rec_full = cov
#                cov_post_rec_full = cov

                res = getxi(recon_res)
                #pdb.set_trace()


                split = {
                    "pre-recon data":   [res["pre-recon data"]] if have_particles else None ,
                    "pre-recon cov":    cov_pre_rec_full,
                    "post-recon data":  [res["post-recon data"]] if have_particles else None ,
                    "post-recon cov":   cov_post_rec_full,
                    "pre-recon mocks":  [res["pre-recon mocks"]] if have_mocks else None ,
                    "post-recon mocks": [res["post-recon mocks"]] if have_mocks else None ,
                    "cosmology": cosmology,
                    "name": f"{sim}, {REDSHIFT}, {dataset}, {hod_recon_model}, correlation function",
                    "info": recon_res['mocks']['info'],
                }
                print(f"Writing pickle to {REDSHIFT}/{sim}/{dataset}/{xi_pickle_name}")
                with open(f"{REDSHIFT}/{sim}/{dataset}/{xi_pickle_name}", "wb") as f:
                    pickle.dump(split, f)


                # set cov for multipoles 0 and 2 
                # cov = np.zeros((5 * len(ks), 5 * len(ks)))
                # cov[:len(ks),                       :len(ks)] = np.eye(len(ks), len(ks))
                # cov[2*len(ks):3*len(ks), 2*len(ks):3*len(ks)] = 0.2 * np.eye(len(ks), len(ks))


                # set cov between multipoles 2 and 0. 
                # cov[:len(ks),            2*len(ks):3*len(ks)] = np.eye(len(ks), len(ks))
                # cov[2*len(ks):3*len(ks), :len(ks)]            = np.eye(len(ks), len(ks))
                cov = None 

                res = getpk(recon_res)     
                split = {
                    "pre-recon data":   [res["pre-recon data"]] if have_particles else None ,
                    "pre-recon cov":    cov,
                    "post-recon data":  [res["post-recon data"]] if have_particles else None ,
                    "post-recon cov":   cov,
                    "pre-recon mocks":  [res["pre-recon mocks"]] if have_mocks else None ,
                    "post-recon mocks": [res["post-recon mocks"]] if have_mocks else None ,
                    "cosmology": cosmology,
                    "name": f"{sim}, {REDSHIFT}, {dataset}, {hod_recon_model}, power spectrum",
                    "info": recon_res['mocks']['info'],
                    "winfit": getwin(ks),
                    "winpk": None,  # We can set this to None; Barry will set it to zeroes given the length of the data vector.
                    "m_mat": getcomp(ks),
                }

                print(f"Writing pickle to {REDSHIFT}/{sim}/{dataset}/{ps_pickle_name}")
                with open(f"{REDSHIFT}/{sim}/{dataset}/{ps_pickle_name}", "wb") as f:
                    pickle.dump(split, f)
