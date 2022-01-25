import pickle5 as pickle
import pandas as pd
import numpy as np
import os


# TODO : stick this into some global config file, so there's no duplication b/w
# hod, recon, and barry

# directory with results
RESULTS_BASE_DIR = "/mnt/marvin1/nam/scratch/data_mocks_summit_new"

SIM = "AbacusSummit_base_c000_ph000"
REDSHIFT = "z0.500"

# match Chris' output keys to barry's expected keys
keys = { ("data",  "delta") :                 "pre-recon data" ,
         ("data",  "delta-reconstructed") :  "post-recon data" ,
         ("mocks", "delta") :                 "pre-recon mocks",
         ("mocks", "delta-reconstructed") :  "post-recon mocks",
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

    for dataset in ["data", "mocks"]:
        for deltatype in ["delta", "delta-reconstructed"]:
            df = pd.DataFrame(columns = ["s", "xi0", "xi2", "xi4"])
            df["s"] = recon_res[dataset]["xi"][deltatype]["r"]
            df["xi0"] = recon_res[dataset]["xi"][deltatype]["xi"][:,0]
            df["xi2"] = recon_res[dataset]["xi"][deltatype]["xi"][:,1]
            df["xi4"] = 0
            mask = df["s"] <= 200.0
            masked = df.loc[mask, ["s", "xi0", "xi2", "xi4"]]
            res[keys[(dataset, deltatype)]] = masked.astype(np.float32)
    return res


def getpk(recon_res):
    res = emptyresdict()

    for dataset in ["data", "mocks"]:
        for deltatype in ["delta", "delta-reconstructed"]:
            df = pd.DataFrame(columns = ["k", "pk0", "pk1", "pk2", "pk3", "pk4"])
            df["k"] = recon_res[dataset]["ps"][deltatype]["k"]
            df["pk0"] = recon_res[dataset]["ps"][deltatype]["ps"][:,0]
            df["pk1"] = 0
            df["pk2"] = recon_res[dataset]["ps"][deltatype]["ps"][:,1]
            df["pk3"] = 0
            df["pk4"] = 0
            mask = df["k"] <= 0.5
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

    ds = os.path.join(RESULTS_BASE_DIR, SIM, REDSHIFT, "recon/")
    # ds = os.path.join("/Users/nina/Desktop/Harvard/phd/bao_project/")

    # find all pickles! one pickle per data type (e.g. particles? or galaxies?). each pickle contains pre and post recon PSs and Xis. 
    files = [ds + f for f in os.listdir(ds) if "pickle" in f]
    print(files)
    assert len(files) == 2, "There should be two pickles! One for particle data, one for mocks."

    # will want to load all pickles, both data and mocks, and then use them to populate dictionary below.
    for fn in files:
        with open(fn, 'rb') as handle:
            if "data" in os.path.basename(fn): 
                recon_data_res  = pickle.load(handle)
            elif "mocks" in os.path.basename(fn): 
                recon_mocks_res = pickle.load(handle)


    recon_res = {"data": recon_data_res, "mocks": recon_mocks_res}

    # for every dict in recon_res (data, mocks)
    #     for every ps in dict (delta, delta_recon)
    #         assert that ks are the same as previous dict. 
    ks = recon_res["data"]["ps"]["delta"]["k"]
    for res_key in recon_res.keys():
        res_value_ps = recon_res[res_key]["ps"]
        # res_key takes values "data", "mocks"
        for ps_key in res_value_ps.keys(): 
            # ps_key takes values "delta", "delta_recon"
            assert np.array_equal(ks, res_value_ps[ps_key]["k"]), "All power spectra ks should be the same! (for {data/mocks}/{delta/delta_recon})"
        
    cov = np.eye(5 * len(ks), 5 * len(ks))

    cosmology = {
    # AbacusSummit read the docs says:
    # remember that Omega_M = (om_b + om_cdm + om_nu ) / h^2 
    # therefore om_b = Om_b * h^2 
    # so we copy/paste the AS values here, directly. 
    # Omega_Nu (Ncdm) * h^2 = sum(mi) / 93.14 eV 
    # for base boxes, we only have one N_Ncdm and 2 N_ur
            "om": (0.12 + 0.02237 + 0.0006442) / 0.6736 ** 2,
            "h0": 0.6736,
            "z": 0.5,
            "ob": 0.02237 / 0.6736 ** 2,
            "ns": 0.9649,
            "mnu": 0.0006442 * 93.14, 
            "reconsmoothscale": 10, 
            }

    res = getpk(recon_res)

    print(res["pre-recon data"])
    
    split = {
        "pre-recon data":   [res["pre-recon data"]] ,
        "pre-recon cov":    cov,
        "post-recon data":  [res["post-recon data"]] ,
        "post-recon cov":   cov,
        "pre-recon mocks":  [res["pre-recon mocks"]] ,
        "post-recon mocks": [res["post-recon data"]] ,
        "cosmology": cosmology,
        "name": f"AbacusSummit_base_c000_ph000 test model Ps",
        "winfit": getwin(ks),
        "winpk": None,  # We can set this to None; Barry will set it to zeroes given the length of the data vector.
        "m_mat": getcomp(ks),
    }

    with open(f"../abacussummit_base_c000_ph000 ps_test.pkl", "wb") as f:
        pickle.dump(split, f)


    res = getxi(recon_res)

    split = {
        "pre-recon data":   [res["pre-recon data"]] ,
        "pre-recon cov":    cov,
        "post-recon data":  [res["post-recon data"]] ,
        "post-recon cov":   cov,
        "pre-recon mocks":  [res["pre-recon mocks"]] ,
        "post-recon mocks": [res["post-recon data"]] ,
        "cosmology": cosmology,
        "name": f"AbacusSummit_base_c000_ph000 test model Xi",
    }

    with open(f"../abacussummit_base_c000_ph000 xi_test.pkl", "wb") as f:
        pickle.dump(split, f)
