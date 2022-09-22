import logging
import sys, pdb

sys.path.append("../..")

from barry.models import PowerBeutler2017
from barry.models.bao_correlation import CorrelationFunctionFit
from scipy.interpolate import splev, splrep
import numpy as np


class CorrBeutler2017(CorrelationFunctionFit):
    """xi(s) model inspired from Beutler 2017 and Ross 2017."""

    def __init__(
        self,
        name="Corr Beutler 2017",
        fix_params=("om",),
        smooth_type="hinton2017",
        recon=None,
        smooth=False,
        correction=None,
        isotropic=True,
        poly_poles=(0, 2),
        marg=None,
        tracer=None,
        no_poly=False,
        fix_bs_to_b0=False
    ):
        self.tracer = tracer
        self.recon = False
        self.recon_type = "None"
        if recon is not None:
            if recon.lower() is not "None":
                self.recon_type = "iso"
                if recon.lower() == "ani":
                    self.recon_type = "ani"
                self.recon = True

        self.recon_smoothing_scale = None
        self.no_poly = no_poly
        self.fix_bs_to_b0 = fix_bs_to_b0

        assert marg is not (self.no_poly or self.fix_bs_to_b0)

        if isotropic:
            poly_poles = [0]
        if marg is not None:
            fix_params = list(fix_params)
            for pole in poly_poles:
                fix_params.extend([f"b{{{pole}}}"])
                fix_params.extend([f"a{{{pole}}}_1", f"a{{{pole}}}_2", f"a{{{pole}}}_3"])


                    

        super().__init__(
            name=name,
            fix_params=fix_params,
            smooth_type=smooth_type,
            smooth=smooth,
            correction=correction,
            isotropic=isotropic,
            poly_poles=poly_poles,
            marg=marg,
            fix_bs_to_b0=fix_bs_to_b0,
        )
        self.parent = PowerBeutler2017(
            fix_params=fix_params,
            smooth_type=smooth_type,
            recon=recon,
            smooth=smooth,
            correction=correction,
            isotropic=isotropic,
            marg=marg,
        )
        if self.marg:
            for pole in self.poly_poles:
                self.set_default(f"b{{{pole}}}", 1.0)
                self.set_default(f"a{{{pole}}}_1", 0.0)
                self.set_default(f"a{{{pole}}}_2", 0.0)
                self.set_default(f"a{{{pole}}}_3", 0.0)


    def declare_parameters(self):
        super().declare_parameters()
        if 'ELG' in self.tracer:
            print("ELG sigma_s 2")
            self.add_param("sigma_s", r"$\Sigma_s$", 1.5, 2.5, 2.0)  # Fingers-of-god damping
            # self.add_param("sigma_s", r"$\Sigma_s$", 0.01, 10.0, 0.0)  # Fingers-of-god damping
        elif 'LRG' in self.tracer: 
            print("LRG sigma_s 3.0")
            self.add_param("sigma_s", r"$\Sigma_s$", 2.5, 3.5, 3.0)  # Fingers-of-god damping
            # self.add_param("sigma_s", r"$\Sigma_s$", 0.01, 10.0, 0.0)  # Fingers-of-god damping


        if self.isotropic:
            self.add_param("sigma_nl", r"$\Sigma_{nl}$", 0.01, 12.0, 10.0)  # BAO damping
        else:
            # if self.recon:
            #     self.add_param("beta", r"$\beta$", -0.5, 0.5, 0.0)  # RSD parameter f/b
            #     self.add_param("sigma_nl_par", r"$\Sigma_{nl,||}$", 0.01, 6.0, 4.0)  # BAO damping parallel to LOS 
            #     self.add_param("sigma_nl_perp", r"$\Sigma_{nl,\perp}$", 0.01, 5.0, 2.5)  # BAO damping perpendicular to LOS
            # else:
            #     self.add_param("beta", r"$\beta$", 0.01, 4.0, 0.2)  # RSD parameter f/b
            #     self.add_param("sigma_nl_par", r"$\Sigma_{nl,||}$", 0.01, 8.0, 6.0)  # BAO damping parallel to LOS
            #     self.add_param("sigma_nl_perp", r"$\Sigma_{nl,\perp}$", 0.01, 4.0, 3)  # BAO damping perpendicular to LOS
            if self.recon:
                self.add_param("beta", r"$\beta$", -0.5, 0.5, 0.0)  # RSD parameter f/b
                self.add_param("sigma_nl_par", r"$\Sigma_{nl,||}$", 0.01, 6.0, 4.0)  # BAO damping parallel to LOS 
                self.add_param("sigma_nl_perp", r"$\Sigma_{nl,\perp}$", 0.01, 5.0, 2.5)  # BAO damping perpendicular to LOS
            else:
                self.add_param("beta", r"$\beta$", 0.01, 4.0, 0.2)  # RSD parameter f/b
                self.add_param("sigma_nl_par", r"$\Sigma_{nl,||}$", 0.01, 8.0, 6.0)  # BAO damping parallel to LOS
                self.add_param("sigma_nl_perp", r"$\Sigma_{nl,\perp}$", 0.01, 4.0, 3.0)  # BAO damping perpendicular to LOS
           
        for pole in self.poly_poles:
            if not self.fix_bs_to_b0:
                self.add_param(f"b{{{pole}}}", r"$b$", 0.1, 10.0, 1.0)  # Galaxy bias
            if self.no_poly:
                self.add_param(f"a{{{pole}}}_1", f"$a_{{{pole},1}}$", 0, 0, 0)  # Monopole Polynomial marginalisation 1
                self.add_param(f"a{{{pole}}}_2", f"$a_{{{pole},2}}$", 0, 0, 0)  # Monopole Polynomial marginalisation 2
                self.add_param(f"a{{{pole}}}_3", f"$a_{{{pole},3}}$", 0, 0, 0)  # Monopole Polynomial marginalisation 3
            else: 
                self.add_param(f"a{{{pole}}}_1", f"$a_{{{pole},1}}$", -100.0, 100.0, 0)  # Monopole Polynomial marginalisation 1
                self.add_param(f"a{{{pole}}}_2", f"$a_{{{pole},2}}$", -2.0, 2.0, 0)  # Monopole Polynomial marginalisation 2
                self.add_param(f"a{{{pole}}}_3", f"$a_{{{pole},3}}$", -0.2, 0.2, 0)  # Monopole Polynomial marginalisation 3
                
        if self.fix_bs_to_b0 and 0 in self.poly_poles:
            self.add_param(f"b{{0}}", r"$b$", 0.1, 10.0, 1.0)  # Galaxy bias

        # pdb.set_trace()
        for p in self.params:
            print(f"\tSetting {p.name}: {p.min}-{p.max}, default: {p.default}")

    def compute_correlation_function(self, dist, p, smooth=False, plotting=False):
        """Computes the correlation function model using the Beutler et. al., 2017 power spectrum
            and 3 bias parameters and polynomial terms per multipole

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
        xi : np.ndarray
            the model monopole, quadrupole and hexadecapole interpolated to sprime.
        poly: np.ndarray
            the additive terms in the model, necessary for analytical marginalisation

        """
        sprime, xi_comp = self.compute_basic_correlation_function(dist, p, smooth=smooth)    
        xi, poly = self.add_three_poly(dist, p, xi_comp, plotting=plotting)

        return sprime, xi, poly


if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    from barry.datasets.dataset_correlation_function import CorrelationFunction_ROSS_DR12
    from barry.config import setup_logging
    from barry.models.model import Correction

    setup_logging()

    print("Checking isotropic data")
    dataset = CorrelationFunction_ROSS_DR12(isotropic=True, recon="iso", realisation="data")
    model = CorrBeutler2017(recon=dataset.recon, marg="full", isotropic=dataset.isotropic, correction=Correction.NONE)
    model.sanity_check(dataset)

    print("Checking anisotropic data")
    dataset = CorrelationFunction_ROSS_DR12(isotropic=False, recon="iso", fit_poles=[0, 2], realisation="data")
    model = CorrBeutler2017(
        recon=dataset.recon,
        isotropic=dataset.isotropic,
        marg="full",
        fix_params=["om"],
        poly_poles=[0, 2],
        correction=Correction.NONE,
    )
    model.sanity_check(dataset)
