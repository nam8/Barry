from functools import lru_cache
import numpy as np

from barry.cosmology.pk2xi import PowerToCorrelationGauss
from barry.cosmology.power_spectrum_smoothing import validate_smooth_method, smooth
from barry.models.model import Model
from barry.models.bao_power import PowerSpectrumFit
from scipy.interpolate import splev, splrep
from scipy.integrate import simps

from barry.utils import break_vector_and_get_blocks


class CorrelationFunctionFit(Model):
    """ A generic model for computing correlation functions."""

    def __init__(
        self,
        name="Corr Basic",
        smooth_type="hinton2017",
        fix_params=("om"),
        smooth=False,
        correction=None,
        isotropic=True,
        poly_poles=(0, 2),
        marg=None,
    ):

        """ Generic correlation function model

        Parameters
        ----------
        name : str, optional
            Name of the model
        smooth_type : str, optional
            The sort of smoothing to use. Either 'hinton2017' or 'eh1998'
        fix_params : list[str], optional
            Parameter names to fix to their defaults. Defaults to just `[om]`.
        smooth : bool, optional
            Whether to generate a smooth model without the BAO feature. Defaults to `false`.
        correction : `Correction` enum. Defaults to `Correction.SELLENTIN
        """
        super().__init__(name, correction=correction, isotropic=isotropic, marg=marg)
        self.parent = PowerSpectrumFit(
            fix_params=fix_params, smooth_type=smooth_type, correction=correction, isotropic=isotropic, poly_poles=poly_poles, marg=marg
        )
        self.poly_poles = poly_poles
        self.smooth_type = smooth_type.lower()
        if not validate_smooth_method(smooth_type):
            exit(0)

        self.declare_parameters()
        self.set_fix_params(fix_params)

        # Set up data structures for model fitting
        self.smooth = smooth

        self.nmu = 100
        self.mu = np.linspace(0.0, 1.0, self.nmu)
        self.pk2xi_0 = None
        self.pk2xi_2 = None
        self.pk2xi_4 = None

    def set_data(self, data):
        """ Sets the models data, including fetching the right cosmology and PT generator.

        Note that if you pass in multiple datas (ie a list with more than one element),
        they need to have the same cosmology.

        Parameters
        ----------
        data : dict, list[dict]
            A list of datas to use
        """
        super().set_data(data)
        self.pk2xi_0 = PowerToCorrelationGauss(self.camb.ks, ell=0)
        self.pk2xi_2 = PowerToCorrelationGauss(self.camb.ks, ell=2)
        self.pk2xi_4 = PowerToCorrelationGauss(self.camb.ks, ell=4)

    def declare_parameters(self):
        """ Defines model parameters, their bounds and default value. """
        self.add_param("om", r"$\Omega_m$", 0.1, 0.5, 0.31)  # Cosmology
        self.add_param("alpha", r"$\alpha$", 0.8, 1.2, 1.0)  # Stretch for monopole
        if not self.isotropic:
            self.add_param("epsilon", r"$\epsilon$", -0.2, 0.2, 0.0)  # Stretch for multipoles
        for pole in self.poly_poles:
            self.add_param(f"b{{{pole}}}", f"$b{{{pole}}}$", 0.01, 10.0, 1.0)  # Linear galaxy bias for each multipole

    @lru_cache(maxsize=1024)
    def compute_basic_power_spectrum(self, om):
        """ Computes the smoothed linear power spectrum and the wiggle ratio.

        Uses a fixed h0 as determined by the dataset cosmology.

        Parameters
        ----------
        om : float
            The Omega_m value to generate a power spectrum for

        Returns
        -------
        array
            pk_smooth - The power spectrum smoothed out
        array
            pk_ratio_dewiggled - the ratio pk_lin / pk_smooth

        """
        # Get base linear power spectrum from camb
        res = self.camb.get_data(om=om, h0=self.camb.h0)
        pk_smooth_lin = smooth(
            self.camb.ks, res["pk_lin"], method=self.smooth_type, om=om, h0=self.camb.h0
        )  # Get the smoothed power spectrum
        pk_ratio = res["pk_lin"] / pk_smooth_lin - 1.0  # Get the ratio
        return pk_smooth_lin, pk_ratio

    @lru_cache(maxsize=32)
    def get_alphas(self, alpha, epsilon):
        """ Computes values of alpha_par and alpha_perp from the input values of alpha and epsilon

        Parameters
        ----------
        alpha : float
            The isotropic dilation scale
        epsilon: float
            The anisotropic warping

        Returns
        -------
        alpha_par : float
            The dilation scale parallel to the line-of-sight
        alpha_perp : float
            The dilation scale perpendicular to the line-of-sight

        """
        return alpha * (1.0 + epsilon) ** 2, alpha / (1.0 + epsilon)

    @lru_cache(maxsize=32)
    def get_sprimefac(self, epsilon):
        """ Computes the prefactor to dilate a s value given epsilon, such that sprime = s * sprimefac * alpha

        Parameters
        ----------
        epsilon: float
            The anisotropic warping

        Returns
        -------
        kprimefac : np.ndarray
            The mu dependent prefactor for dilating a k value

        """
        musq = self.mu ** 2
        epsilonsq = (1.0 + epsilon) ** 2
        sprimefac = np.sqrt(musq * epsilonsq ** 2 + (1.0 - musq) / epsilonsq)
        return sprimefac

    @lru_cache(maxsize=32)
    def get_muprime(self, epsilon):
        """ Computes dilated values of mu given input values of epsilon for the correlation function

        Parameters
        ----------
        epsilon: float
            The anisotropic warping

        Returns
        -------
        muprime : np.ndarray
            The dilated mu values

        """
        musq = self.mu ** 2
        muprime = self.mu / np.sqrt(musq + (1.0 - musq) / (1.0 + epsilon) ** 6)
        return muprime

    def integrate_mu(self, xi2d, isotropic=False):
        xi0 = simps(xi2d, self.mu, axis=1)
        if isotropic:
            xi2 = None
            xi4 = None
        else:
            xi2 = 3.0 * simps(xi2d * self.mu ** 2, self.mu, axis=1)
            xi4 = 35.0 * simps(xi2d * self.mu ** 4, self.mu, axis=1)
        return xi0, xi2, xi4

    def compute_correlation_function(self, dist, p, smooth=False):
        """ Computes the dilated correlation function multipoles at distance d given the supplied params

        Parameters
        ----------
        dist : np.ndarray
            Array of distances in the correlation function to compute
        p : dict
            dictionary of parameter name to float value pairs
        smooth : bool, optional
            Whether or not to generate a smooth model without the BAO feature

        Returns
        -------
        sprime : np.ndarray
            distances of the computed xi
        xi0 : np.ndarray
            the model monopole interpolated to sprime.
        xi2 : np.ndarray
            the model quadrupole interpolated to sprime. Will be 'None' if the model is isotropic

        """
        # Generate the power spectrum multipoles at the undilated k-values without shape additions
        ks = self.camb.ks
        kprime, pk0, pk2, pk4 = self.parent.compute_power_spectrum(ks, p, smooth=smooth, shape=False, dilate=False)

        xi = [np.zeros(len(dist)), np.zeros(len(dist)), np.zeros(len(dist))]
        if self.isotropic:
            sprime = p["alpha"] * dist
            xi[0] = p["b0"] * self.pk2xi_0.__call__(ks, pk0, sprime)
        else:
            # Construct the dilated 2D correlation function by splineing the undilated multipoles. We could have computed these
            # directly at sprime, but sprime depends on both s and mu, so splining is probably quicker
            epsilon = np.round(p["epsilon"], decimals=5)
            sprime = np.outer(dist * p["alpha"], self.get_sprimefac(epsilon))
            muprime = self.get_muprime(epsilon)
            xi0 = splev(sprime, splrep(dist, self.pk2xi_0.__call__(ks, pk0, dist)))
            xi2 = splev(sprime, splrep(dist, self.pk2xi_2.__call__(ks, pk2, dist)))
            xi4 = splev(sprime, splrep(dist, self.pk2xi_4.__call__(ks, pk4, dist)))
            xi2d = xi0 + 0.5 * (3.0 * muprime ** 2 - 1) * xi2 + 0.125 * (35.0 * muprime ** 4 - 30.0 * muprime ** 2 + 3.0) * xi4

            xi0, xi2, xi4 = self.integrate_mu(xi2d)
            xi[0] = p["b{0}"] * xi0
            xi[1] = 2.5 * (p["b{2}"] * xi2 - xi[0])
            if 4 in self.poly_poles:
                xi[2] = 1.125 * (p["b{4}"] * xi4 - 10.0 * p["b{2}"] * xi2 + 3.0 * p["b{0}"] * xi0)
            else:
                xi[2] = 1.125 * (xi4 - 10.0 * p["b{2}"] * xi2 + 3.0 * p["b{0}"] * xi0)

        return sprime, xi[0], xi[1], xi[2], np.zeros((1, len(dist)))

    def get_model(self, p, d, smooth=False):
        """ Gets the model prediction using the data passed in and parameter location specified

        Parameters
        ----------
        p : dict
            A dictionary of parameter names to parameter values
        d : dict
            A specific set of data to compute the model for. For correlation functions, this needs to
            have a key of 'dist' which contains the Mpc/h value of distances to compute.
        smooth : bool, optional
            Whether to only generate a smooth model without the BAO feature

        Returns
        -------
        xi_model : np.ndarray
            The concatenated xi_{\ell}(s) predictions at the dilated distances given p and data['dist']
        poly_model : np.ndarray
            the functions describing any polynomial terms, used for analytical marginalisation
            k values correspond to d['dist']

        """

        dist, xi0, xi2, xi4, poly = self.compute_correlation_function(d["dist"], p, smooth=smooth)

        xi_model = xi0 if self.isotropic else np.concatenate([xi0, xi2])
        if 4 in d["poles"] and not self.isotropic:
            xi_model = np.concatenate([xi_model, xi4])

        poly_model = None
        if self.marg:
            len_poly = len(d["dist"])
            if not self.isotropic:
                len_poly *= len(d["fit_pole_indices"])
            poly_model = np.empty((np.shape(poly)[0], len_poly))
            for n in range(np.shape(poly)[0]):
                poly_model[n] = poly[n] if self.isotropic else np.concatenate([poly[n, 0], poly[n, 1]])
                if 4 in d["poles"] and not self.isotropic:
                    poly_model[n] = np.concatenate([poly_model, poly[n, 2]])

        return xi_model, poly_model

    def get_likelihood(self, p, d):
        """ Uses the stated likelihood correction and `get_model` to compute the likelihood

        Parameters
        ----------
        p : dict
            A dictionary of parameter names to parameter values
        d : dict
            A specific set of data to compute the model for. For correlation functions, this needs to
            have a key of 'dist' which contains the Mpc/h value of distances to compute.

        Returns
        -------
        log_likelihood : float
            The corrected log likelihood
        """
        num_mocks = d["num_mocks"]
        num_params = len(self.get_active_params())

        xi_model, poly_model = self.get_model(p, d, smooth=self.smooth)

        if self.marg_type == "partial":
            return self.get_chi2_partial_marg_likelihood(d["xi"], xi_model, poly_model, d["icov"], None, None, num_mocks=num_mocks)
        elif self.marg_type == "full":
            return self.get_chi2_marg_likelihood(d["xi"], xi_model, poly_model, d["icov"], None, None, num_mocks=num_mocks)
        else:
            return self.get_chi2_likelihood(d["xi"], xi_model, d["icov"], None, None, num_mocks=num_mocks, num_params=num_params)

    def plot(self, params, smooth_params=None, figname=None):
        self.logger.info("Create plot")
        import matplotlib.pyplot as plt

        # Ensures we plot the window convolved model
        ss = self.data[0]["dist"]
        err = np.sqrt(np.diag(self.data[0]["cov"]))
        mod, polymod = self.get_model(params, self.data[0])
        if smooth_params is not None:
            smooth, polysmooth = self.get_model(smooth_params, self.data[0], smooth=True)
        else:
            smooth, polysmooth = self.get_model(params, self.data[0], smooth=True)

        if self.marg:
            if self.isotropic:
                mod_fit = mod
                smooth_fit = smooth
            else:
                mod_fit = break_vector_and_get_blocks(mod, len(self.data[0]["poles"]), self.data[0]["fit_pole_indices"])
                smooth_fit = break_vector_and_get_blocks(smooth, len(self.data[0]["poles"]), self.data[0]["fit_pole_indices"])
            polymod_fit = np.empty((np.shape(polymod)[0], len(self.data[0]["fit_pole_indices"]) * len(self.data[0]["dist"])))
            polysmooth_fit = np.empty((np.shape(polysmooth)[0], len(self.data[0]["fit_pole_indices"]) * len(self.data[0]["dist"])))
            for n in range(np.shape(polymod)[0]):
                polymod_fit[n] = break_vector_and_get_blocks(
                    polymod[n], np.shape(polymod)[1] / len(self.data[0]["dist"]), self.data[0]["fit_pole_indices"]
                )
                polysmooth_fit[n] = break_vector_and_get_blocks(
                    polysmooth[n], np.shape(polysmooth)[1] / len(self.data[0]["dist"]), self.data[0]["fit_pole_indices"]
                )
            bband = self.get_ML_nuisance(polymod_fit, mod_fit, self.data[0]["xi"], self.data[0]["icov"], None, None)
            mod += bband @ polymod
            print(f"Maximum likelihood nuisance parameters at maximum a posteriori point are {bband}")
            bband = self.get_ML_nuisance(polysmooth_fit, smooth_fit, self.data[0]["xi"], self.data[0]["icov"], None, None)
            smooth += bband @ polysmooth

        # Split up the different multipoles if we have them
        if len(err) > len(ss):
            assert len(err) % len(ss) == 0, f"Cannot split your data - have {len(err)} points and {len(ss)} bins"
        errs = [row for row in err.reshape((-1, len(ss)))]
        mods = [row for row in mod.reshape((-1, len(ss)))]
        smooths = [row for row in smooth.reshape((-1, len(ss)))]
        if self.isotropic:
            names = [f"xi0"]
        else:
            names = [f"xi{n}" for n in self.data[0]["poles"]]
        labels = [f"$\\xi_{{{n}}}(s)$" for n in self.data[0]["poles"]]
        num_rows = len(names)
        cs = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
        height = 2 + 1.4 * num_rows

        fig, axes = plt.subplots(figsize=(9, height), nrows=num_rows, ncols=2, sharex=True, squeeze=False)
        ratio = (height - 1) / height
        plt.subplots_adjust(left=0.1, top=ratio, bottom=0.05, right=0.85, hspace=0, wspace=0.3)
        for ax, err, mod, smooth, name, label, c in zip(axes, errs, mods, smooths, names, labels, cs):

            # Plot ye old data
            ax[0].errorbar(ss, ss ** 2 * self.data[0][name], yerr=ss ** 2 * err, fmt="o", ms=4, label="Data", c=c)
            ax[1].errorbar(ss, ss ** 2 * (self.data[0][name] - smooth), yerr=ss ** 2 * err, fmt="o", ms=4, label="Data", c=c)

            # Plot ye old model
            ax[0].plot(ss, ss ** 2 * mod, c=c, label="Model")
            ax[1].plot(ss, ss ** 2 * (mod - smooth), c=c, label="Model")

            ax[0].set_ylabel("$s^{2} \\times $ " + label)

            if name not in [f"xi{n}" for n in self.data[0]["fit_poles"]]:
                ax[0].set_facecolor("#e1e1e1")
                ax[1].set_facecolor("#e1e1e1")

        # Show the model parameters
        string = f"$\\mathcal{{L}}$: {self.get_likelihood(params, self.data[0]):0.3g}\n"
        if self.marg:
            string += "\n".join([f"{self.param_dict[l].label}={v:0.4g}" for l, v in params.items() if v not in self.fix_params])
        else:
            string += "\n".join([f"{self.param_dict[l].label}={v:0.4g}" for l, v in params.items()])
        va = "center" if self.postprocess is None else "top"
        ypos = 0.5 if self.postprocess is None else 0.98
        fig.text(0.99, ypos, string, horizontalalignment="right", verticalalignment=va)
        axes[-1, 0].set_xlabel("s")
        axes[-1, 1].set_xlabel("s")
        axes[0, 0].legend(frameon=False)

        if self.postprocess is None:
            axes[0, 1].set_title("$\\xi(s) - \\xi_{\\rm smooth}(s)$")
        else:
            axes[0, 1].set_title("$\\xi(s) - data$")
        axes[0, 0].set_title("$s^{2} \\times \\xi(s)$")

        fig.suptitle(self.data[0]["name"] + " + " + self.get_name())
        if figname is not None:
            fig.savefig(figname, bbox_inches="tight", transparent=True, dpi=300)
        plt.show()


if __name__ == "__main__":
    print("Calling a Generic model class as main does not do anything. Try running one of the Concrete classes: ")
    print("bao_correlation_Beutler2017.py")
    print("bao_correlation_Ding2018.py")
    print("bao_correlation_Seo2016.py")
