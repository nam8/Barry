import pickle5 as pickle
import pandas as pd
import numpy as np
import os
from pathlib import Path
import pdb
from scipy.io import mmread, mmwrite
import argparse

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
    
def getxi(recon_res):
    res = emptyresdict() 

    for deltatype in ["delta-reconstructed", "delta"]:
        # pdb.set_trace()
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


# def getpk(recon_res):
#     res = emptyresdict()

#     for dataset in recon_res.keys():
#         for deltatype in ["delta-reconstructed", "delta"]:
#             df = pd.DataFrame(columns = ["k", "pk0", "pk1", "pk2", "pk3", "pk4"])

#             # pdb.set_trace()

#             bias =  recon_res[dataset]['info']['recon config']['bias'] if deltatype is "delta" else 1.0 

#             df["k"]   = recon_res[dataset]["ps"][deltatype]["k"]
#             df["pk0"] = recon_res[dataset]["ps"][deltatype]["ps"][:,0] * bias**2
#             df["pk1"] = 0
#             df["pk2"] = recon_res[dataset]["ps"][deltatype]["ps"][:,1] * bias**2
#             df["pk3"] = 0
#             df["pk4"] = 0
#             mask   = df["k"] <= 0.5
#             masked = df.loc[mask, ["k", "pk0", "pk1", "pk2", "pk3", "pk4"]]
#             res[keys[(dataset, deltatype)]] = masked.astype(np.float32)

#     return res


def getwin(ks):
    res = {"w_ks_input": ks.copy(), "w_k0_scale": np.zeros(ks.size), "w_transform": np.eye(5 * ks.size), "w_ks_output": ks.copy()}
    return {1: res}  # Step size is one


def getcomp(ks):
    matrix = np.zeros((5 * ks.size, 3 * ks.size))
    matrix[: ks.size, : ks.size] = np.diag(np.ones(ks.size))
    matrix[2 * ks.size : 3 * ks.size, ks.size : 2 * ks.size] = np.diag(np.ones(ks.size))
    matrix[4 * ks.size :, 2 * ks.size :] = np.diag(np.ones(ks.size))
    return matrix

def cyclic_set_results(all_box_results, mcyc):
    avg_res = {'info': None, 'xi': {'delta-reconstructed': {'r': None, 'xi': None}, 'delta': {'r': None, 'xi': None}}}

    # make sure all results were run with the same HOD, recon, redshift, tracer, etc. parameters, and that all results' r bins for xi are the same. 
    i = 0
    avg_bias = 0.0
    for box in all_box_results.keys():
        if i == 0:
            firstbox = box
        if i == mcyc-1:
            lastbox = box

        for d in ["delta-reconstructed", "delta"]:
            assert np.array_equal(all_box_results[firstbox]['xi'][d]['r'], all_box_results[box]['xi'][d]['r'])

        for key in all_box_results[box]['info'].keys():
            if key == 'simname':
                continue
            if key == 'recon config':
                for recon_config_key in all_box_results[box]['info'][key].keys():
                    if recon_config_key == 'bias':
                        print(f"\t\t{all_box_results[box]['info'][key][recon_config_key]}")
                        avg_bias = avg_bias + all_box_results[box]['info'][key][recon_config_key]
                    else:
                        assert all_box_results[firstbox]['info'][key][recon_config_key] == all_box_results[box]['info'][key][recon_config_key]

            else: 
                assert all_box_results[firstbox]['info'][key] == all_box_results[box]['info'][key]
        i = i+1

    avg_bias = avg_bias / mcyc
    print(f"\t\tAverage bias: {avg_bias}")

    # if we've passed the asserts above, then we're safe to set avg_res info to first box's info. 
    avg_res['info'] = all_box_results[firstbox]['info']
    avg_res['info']['recon config']['bias'] = avg_bias
    # but we do need to update the simname
    lastphase = lastbox.split("_")[-1]
    avg_res['info']['simname'] = f"CYC_{firstbox}_{lastphase}"

    # okay! Now let's average the xis. (not going to bother with power spectra, at least for now)
    for d in ["delta-reconstructed", "delta"]:
        avg_res['xi'][d]['r'] = all_box_results[firstbox]['xi'][d]['r'] 

        xi_0s = np.empty((mcyc, len(avg_res['xi'][d]['r'])))
        xi_2s = np.empty((mcyc, len(avg_res['xi'][d]['r'])))
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

def get_cov(tracer, hod_recon_model, addshot):
    pre_rec  = mmread(os.path.join(os.getcwd(), "covariance_matrices", tracer, f"{REDSHIFT}_{tracer}_{hod_recon_model}_cov_pre_recon.mtx"))
    post_rec = mmread(os.path.join(os.getcwd(), "covariance_matrices", tracer, f"{REDSHIFT}_{tracer}_{hod_recon_model}_cov_post_recon.mtx"))

    # sanity checks
    assert(np.shape(pre_rec) == np.shape(post_rec))
    assert(np.shape(pre_rec)[0] == np.shape(pre_rec)[1])
    ndim = np.shape(pre_rec)[0]

    # add shot noise to diagnoal, if requested at runtime
    pre_rec  =  pre_rec + np.eye(ndim) * addshot
    post_rec = post_rec + np.eye(ndim) * addshot

    return pre_rec, post_rec

def main(mcyc, addshot): 
    # we are going to do a modified, cyclic jacknife. Loop over all boxes. Treat first phase as starting box of cyclic set w/ mcyc boxes. 
    
    assert((mcyc > 0) and (mcyc <= 25))
    max_s = 25 if mcyc < 25 else 1

    for s in range(0, max_s): 
        phases = [(s + i)%25 for i in range(mcyc)]
        sims = ['AbacusSummit_base_c000_ph'+str(phase).zfill(3) for phase in phases] 

        for tracer in ["LRG_rd", "ELG_rd"]: 
            tracer_dirs = [os.path.join(RESULTS_BASE_DIR, sim, REDSHIFT, "recon", tracer) for sim in sims]
            hod_recon_model_dirs = [os.listdir(tracer_dir) for tracer_dir in tracer_dirs]

            for i in range(mcyc-1):
                assert set(hod_recon_model_dirs[i]) == set(hod_recon_model_dirs[i+1])
                assert len(hod_recon_model_dirs[i]) == len(hod_recon_model_dirs[i+1])

            for hod_recon_model in hod_recon_model_dirs[0]: # if we passed the asserts above, we can just use one box's list of HOD models. 
                model_dirs = [os.path.join(tracer_dir, hod_recon_model) for tracer_dir in tracer_dirs] 
                all_box_files = [[os.path.join(model_dir, f) for f in os.listdir(model_dir) if "pickle" in f] for model_dir in model_dirs] 
                assert(len(all_box_files) == mcyc)
                print(f"Processing {len(all_box_files)} files: {all_box_files}")

                # cyclic jackknifing only applies to mocks. 
                # load all pickles 
                all_box_res = {}
                i = 0
                for this_box_files in all_box_files:
                    for fn in this_box_files:
                        with open(fn, 'rb') as handle:
                            all_box_res[f"{sims[i]}"] = pickle.load(handle)
                    i = i+1

                # make dictionary that combines the mcyc mocks. Average xis. 
                recon_res = cyclic_set_results(all_box_res, mcyc)

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
                        "shotnoise": addshot
                }

                base_pickle_name = f"{recon_res['info']['simname']}_{REDSHIFT}_{tracer}_{hod_recon_model}"
                xi_pickle_name = f"{base_pickle_name}_xi.pkl"

                if not os.path.exists( f"{REDSHIFT}/{recon_res['info']['simname']}/{tracer}/"):
                    os.makedirs(f"{REDSHIFT}/{recon_res['info']['simname']}/{tracer}/")

                ss = recon_res["xi"]["delta"]["r"]

                cov_pre_rec_xi02, cov_post_rec_xi02 = get_cov(tracer, hod_recon_model, addshot)

                assert(np.shape(cov_post_rec_xi02) == (2*len(ss), 2*len(ss)))
                assert(np.shape( cov_pre_rec_xi02) == (2*len(ss), 2*len(ss)))

                cov_pre_rec_full  = np.zeros((3 * len(ss), 3 * len(ss)))
                cov_post_rec_full = np.zeros((3 * len(ss), 3 * len(ss)))
                
                cov_pre_rec_full[ :2*len(ss), :2*len(ss)] = cov_pre_rec_xi02/mcyc
                cov_post_rec_full[:2*len(ss), :2*len(ss)] = cov_post_rec_xi02/mcyc

                res = getxi(recon_res)

                split = {
                    "pre-recon data":   None ,
                    "pre-recon cov":    cov_pre_rec_full,
                    "post-recon data":  None ,
                    "post-recon cov":   cov_post_rec_full,
                    "pre-recon mocks":  [res["pre-recon mocks"]] ,
                    "post-recon mocks": [res["post-recon mocks"]] ,
                    "cosmology": cosmology,
                    "name": f"{recon_res['info']['simname']}, {REDSHIFT}, {tracer}, {hod_recon_model}, correlation function",
                    "info": recon_res['info'],
                }

                print(f"Writing pickle to {REDSHIFT}/{recon_res['info']['simname']}/{tracer}/{xi_pickle_name}")
                with open(f"{REDSHIFT}/{recon_res['info']['simname']}/{tracer}/{xi_pickle_name}", "wb") as f:
                    pickle.dump(split, f)

            pdb.set_trace()


class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":

    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--mcyc', help='Number of boxes to cyclically co-add', default=13, type=int)
    parser.add_argument('--addshot', help='How much shot noise to add to cov. matrices diagonals', default=0, type=float)
    args = vars(parser.parse_args())
    main(**args)
