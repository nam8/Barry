import sys

sys.path.append("..")
from barry.cosmology.camb_generator import getCambGenerator
from barry.postprocessing import BAOExtractor, PureBAOExtractor
from barry.config import setup
from barry.models import PowerSeo2016, PowerBeutler2017, PowerDing2018, PowerNoda2019
from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.samplers import DynestySampler
from barry.fitter import Fitter

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = getCambGenerator()
    r_s = c.get_data()[0]
    p = BAOExtractor(r_s)

    sampler = DynestySampler(temp_dir=dir_name)
    fitter = Fitter(dir_name)

    cs = ["#262232", "#116A71", "#48AB75", "#D1E05B"]

    for r in [True, False]:
        t = "Recon" if r else "Prerecon"
        ls = "-" if r else "--"
        d = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r)
        de = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, postprocess=p)

        # Fix sigma_nl for one of the Beutler models
        model = PowerBeutler2017(recon=r)
        sigma_nl = 6.0 if r else 9.3
        model.set_default("sigma_nl", sigma_nl)
        model.set_fix_params(["om", "sigma_nl"])

        fitter.add_model_and_dataset(PowerBeutler2017(recon=r), d, name=f"Beutler 2017 {t}", linestyle=ls, color=cs[0])
        fitter.add_model_and_dataset(model, d, name=f"Beutler 2017 Fixed $\\Sigma_{{nl}}$ {t}", linestyle=ls, color=cs[0])
        fitter.add_model_and_dataset(PowerSeo2016(recon=r), d, name=f"Seo 2016 {t}", linestyle=ls, color=cs[1])
        fitter.add_model_and_dataset(PowerDing2018(recon=r), d, name=f"Ding 2018 {t}", linestyle=ls, color=cs[2])
        fitter.add_model_and_dataset(PowerNoda2019(recon=r, postprocess=p), de, name=f"Noda 2019 {t}", linestyle=ls, color=cs[3])

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(10)
    fitter.fit(file)

    if fitter.should_plot():
        import logging

        logging.info("Creating plots")

        if False:
            from chainconsumer import ChainConsumer

            c = ChainConsumer()
            for posterior, weight, chain, evidence, model, data, extra in fitter.load():
                c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
                print(extra["name"], chain.shape, weight.shape, posterior.shape)
            c.configure(shade=True, bins=30, legend_artists=True)
            c.analysis.get_latex_table(filename=pfn + "_params.txt", parameters=[r"$\alpha$"])
            c.plotter.plot_summary(filename=pfn + "_summary.png", extra_parameter_spacing=1.5, errorbar=True, truth={"$\\Omega_m$": 0.31, "$\\alpha$": 1.0})
            c.plotter.plot_summary(
                filename=[pfn + "_summary2.png", pfn + "_summary2.pdf"],
                extra_parameter_spacing=1.5,
                parameters=1,
                errorbar=True,
                truth={"$\\Omega_m$": 0.31, "$\\alpha$": 1.0},
            )
            # c.plotter.plot(filename=pfn + "_contour.png", truth={"$\\Omega_m$": 0.31, '$\\alpha$': 1.0})
            # c.plotter.plot_walks(filename=pfn + "_walks.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})

        # Plots the average recon mock measurements and the best-fit models from each of the models tested.
        # We'll also plot the ratio for everything against the smooth Beutler2017 model.
        if True:

            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            fig, axes = plt.subplots(figsize=(6, 8), nrows=2, sharex=True, gridspec_kw={"hspace": 0.06, "height_ratios": [4.3, 1]})
            inner = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=axes[0], hspace=0.04)
            ax = plt.subplot(inner[0:])
            ax.spines["top"].set_color("none")
            ax.spines["bottom"].set_color("none")
            ax.spines["left"].set_color("none")
            ax.spines["right"].set_color("none")
            ax.set_ylabel(r"$P(k) / P_{\mathrm{smooth}}(k)$")
            ax.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)

            counter = 0
            for posterior, weight, chain, evidence, model, data, extra in fitter.load():
                if not "Recon" in extra["name"]:
                    continue
                if extra["name"] == "Noda 2019 Recon":
                    model.postprocess = PureBAOExtractor(r_s)
                model.set_data(data)
                p = model.get_param_dict(chain[np.argmax(posterior)])
                smooth = model.get_model(p, data[0], smooth=True)
                bestfit = model.get_model(p, data[0])
                if counter == 0:
                    ks = data[0]["ks"]
                    pk = data[0]["pk"]
                    err = np.sqrt(np.diag(data[0]["cov"]))
                    rk = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=True, postprocess=PureBAOExtractor(r_s)).get_data()[0]
                if extra["name"] == "Noda 2019 Recon":
                    axes[1].errorbar(ks, rk["pk"], yerr=np.sqrt(np.diag(rk["cov"])), fmt="o", c="k", ms=4, zorder=0)
                    axes[1].plot(ks, bestfit, color=extra["color"], label=extra["name"], linestyle=extra["linestyle"], zorder=1, linewidth=1.8)
                    axes[1].set_ylabel(r"$R(k)$", labelpad=-5)
                    axes[1].legend()
                else:
                    ax = fig.add_subplot(inner[counter])
                    ax.errorbar(ks, pk / smooth, yerr=err / smooth, fmt="o", c="k", ms=4, zorder=0)
                    ax.plot(ks, bestfit / smooth, color=extra["color"], label=extra["name"], linestyle=extra["linestyle"], zorder=1, linewidth=1.8)
                    ax.set_ylim(0.88, 1.12)
                    ax.tick_params(axis="x", which="both", labelcolor="none", bottom=False, labelbottom=False)
                    ax.legend()
                counter += 1
            axes[1].set_xlabel(r"$k\,(h\,\mathrm{Mpc^{-1}})$")
            plt.savefig(pfn + "_bestfits.pdf")

            fig, axes = plt.subplots(figsize=(6, 8), nrows=2, sharex=True, gridspec_kw={"hspace": 0.04, "height_ratios": [1.5, 1]})

            for posterior, weight, chain, evidence, model, data, extra in fitter.load():
                if not "Recon" in extra["name"]:
                    continue
                if extra["name"] == "Noda 2019 Recon":
                    model.postprocess = None
                model.set_data(data)
                p = model.get_param_dict(chain[np.argmax(posterior)])
                smooth = model.get_model(p, data[0], smooth=True)
                bestfit = model.get_model(p, data[0])
                if extra["name"] == "Beutler 2017 Recon":
                    ks = data[0]["ks"]
                    pk = data[0]["pk"]
                    err = np.sqrt(np.diag(data[0]["cov"]))
                    axes[0].errorbar(ks, ks * pk, yerr=ks * err, fmt="o", c="k", ms=4, zorder=0)
                    axes[1].fill_between(ks, 1.0 + err / pk, 1.0 - err / pk, color="k", zorder=0, alpha=0.15)
                axes[0].plot(ks, ks * bestfit, color=extra["color"], label=extra["name"], linestyle=extra["linestyle"], zorder=1, linewidth=1.5)
                axes[1].plot(ks, bestfit / pk, color=extra["color"], label=extra["name"], linestyle=extra["linestyle"], zorder=1, linewidth=1.5)
            axes[0].set_ylim(750.0, 1650.0)
            axes[1].set_ylim(0.95, 1.05)
            axes[0].set_ylabel(r"$kP(k) (h^{-2}\,\mathrm{Mpc^{2}})$")
            axes[1].set_ylabel(r"$\mathrm{Bestfit} / \mathrm{Data}$")
            axes[0].tick_params(axis="x", which="both", labelcolor="none", bottom=False, labelbottom=False)
            axes[0].legend()
            axes[1].set_xlabel(r"$k\,(h\,\mathrm{Mpc^{-1}})$")
            plt.savefig(pfn + "_bestfits_2.pdf")
