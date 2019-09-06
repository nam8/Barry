import logging
from barry.models import CorrBeutler2017

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    from barry.datasets import CorrelationFunction_SDSS_DR12_Z061_NGC

    for recon in [True, False]:
        dataset1 = CorrelationFunction_SDSS_DR12_Z061_NGC(recon=recon)
        data1 = dataset1.get_data()

        # Run on mean
        model = CorrBeutler2017(name=f"Beutler2017, recon={recon}")
        model.set_data(data1)
        p, minv = model.optimize(maxiter=2000, niter=200)
        sigma_nl = p["sigma_nl"]

        print(f"recon={recon}, Sigma_nl found to be {sigma_nl:0.3f} with prob {minv:0.3f}")
        model.set_default("sigma_nl", sigma_nl)
        model.set_fix_params(["om", "sigma_nl"])

        # Recon 5.0
        # Prerecon 8.0
