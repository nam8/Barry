import logging
import os
import numpy as np
from barry.samplers.sampler import Sampler


class DynestySampler(Sampler):
    def __init__(self, temp_dir=None, max_iter=None, dynamic=False, nlive=500):

        self.logger = logging.getLogger("barry")
        self.max_iter = max_iter
        self.nlive = nlive
        # dynesty.utils.merge_runs()
        self.temp_dir = temp_dir
        if temp_dir is not None and not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        self.dynamic = dynamic

    def get_filename(self, uid):
        if self.dynamic:
            return os.path.join(self.temp_dir, f"{uid}_dyn_chain.npy")
        else:
            return os.path.join(self.temp_dir, f"{uid}_nest_chain.npy")

    def fit(self, log_likelihood, start, num_dim, prior_transform, save_dims=None, uid=None):

        import dynesty

        filename = self.get_filename(uid)
        if os.path.exists(filename):
            self.logger.info("Not sampling, returning result from file.")
            return self.load_file(filename)
        self.logger.info("Sampling posterior now")

        if save_dims is None:
            save_dims = num_dim
        self.logger.info("Fitting framework with %d dimensions and %d live points" % (num_dim , self.nlive))
        self.logger.info("Using dynesty Sampler")
        if self.dynamic:
            sampler = dynesty.DynamicNestedSampler(
                log_likelihood, prior_transform, num_dim, nlive_init=self.nlive, nlive_batch=200, maxbatch=10
            )
        else:
            sampler = dynesty.NestedSampler(log_likelihood, prior_transform, num_dim, nlive=self.nlive)

            

        # £££££££££££££££££££££££££££££££££ 
        # sampler.run_nested(maxiter=1000, print_progress=True)
        sampler.run_nested(maxiter=self.max_iter, print_progress=True)
        # res = sampler.results
        # import matplotlib
        # from matplotlib import pyplot as plt
        # from dynesty import plotting as dyplot

        # print(res.keys())

        # dynesty.plotting.traceplot(res) 
        # plt.savefig("/home/nam/bao_fit_project/dynesty-traceplot.png")
        # exit(1)

        # compute effective 'multi' volumes
        # multi_logvols = [0.]  # unit cube
        # for bound in res.bound[1:]:  # skip unit cube
        #     logvol, funit = bound.monte_carlo_logvol(rstate=rstate, return_overlap=True)
        #     multi_logvols.append(logvol +np.log( funit))  # numerical estimate via Monte Carlo methods
        # multi_logvols = np.array(multi_logvols)

        # # plot results as a function of ln(volume)
        # plt.figure(figsize=(12,6))
        # plt.xlabel(r'$-\ln X_i$')
        # plt.ylabel(r'$\ln V_i$')

        # # 'multi'
        # x, it = -res.logvol, res.bound_iter
        # y = multi_logvols[it]
        # plt.plot(x, y, lw=3, label='multi')
        # plt.legend(loc='best', fontsize=24);
        # plt.savefig("/home/nam/bao_fit_project/dynesty.png")
        # exit(1)

        # £££££££££££££££££££££££££££££££££ 


        self.logger.debug("Fit finished")

        dresults = sampler.results
        logz = dresults["logz"]
        chain = dresults["samples"]
        weights = np.exp(dresults["logwt"] - dresults["logz"][-1])
        max_weight = weights.max()
        trim = max_weight / 1e5
        mask = weights > trim
        likelihood = dresults["logl"]
        self._save(chain[mask, :], weights[mask], likelihood[mask], filename, logz[mask], save_dims)
        return {"chain": chain[mask, :], "weights": weights[mask], "posterior": likelihood[mask], "evidence": logz}

    def _save(self, chain, weights, likelihood, filename, logz, save_dims):
        res = np.vstack((likelihood, weights, logz, chain[:, :save_dims].T)).T
        np.save(filename, res.astype(np.float32))

    def load_file(self, filename):
        results = np.load(filename)
        likelihood = results[:, 0]
        weights = results[:, 1]
        logz = results[:, 2]
        flat_chain = results[:, 3:]
        return {"chain": flat_chain, "posterior": likelihood, "evidence": logz, "weights": weights}
