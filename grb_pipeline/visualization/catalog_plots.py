"""Catalog-level plotting utilities for GRB analysis."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from scipy import stats
from scipy.stats import gaussian_kde


class CatalogPlotter:
    """Plotter for GRB catalog-level analysis visualizations."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the CatalogPlotter.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary for plot parameters
        """
        self.config = config or {}

        # Default figure parameters
        self.figsize = self.config.get("figsize", (10, 6))
        self.dpi = self.config.get("dpi", 300)
        self.font_size = self.config.get("font_size", 11)
        self.label_size = self.config.get("label_size", 12)
        self.title_size = self.config.get("title_size", 14)

        self._setup_style()

    def _setup_style(self):
        """Configure matplotlib rcParams."""
        rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.2,
            "axes.labelsize": self.label_size,
            "axes.titlesize": self.title_size,
            "xtick.labelsize": self.font_size,
            "ytick.labelsize": self.font_size,
            "legend.fontsize": self.config.get("legend_size", 10),
            "font.size": self.font_size,
            "font.family": "serif",
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "grid.alpha": 0.3,
        })

    def plot_amati_relation(
        self,
        epeaks: np.ndarray,
        eisos: np.ndarray,
        epeak_errs: Optional[np.ndarray] = None,
        eiso_errs: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
        highlight_grb: Optional[str] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot Epeak vs Eiso scatter with best-fit line and 2-sigma region.

        Parameters
        ----------
        epeaks : np.ndarray
            Peak energy values (keV)
        eisos : np.ndarray
            Isotropic equivalent energy values (erg)
        epeak_errs : np.ndarray, optional
            Peak energy errors
        eiso_errs : np.ndarray, optional
            Eiso errors
        labels : list, optional
            GRB names/labels
        highlight_grb : str, optional
            Name of GRB to highlight
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Filter valid data
        valid_mask = (np.isfinite(epeaks) & np.isfinite(eisos) &
                     (epeaks > 0) & (eisos > 0))
        ep_clean = epeaks[valid_mask]
        eiso_clean = eisos[valid_mask]

        # Plot data
        ax.scatter(ep_clean, eiso_clean, s=80, color="#1f77b4", alpha=0.6,
                  edgecolors="black", linewidth=0.5, label="GRBs")

        # Error bars
        if epeak_errs is not None and eiso_errs is not None:
            ep_err_clean = epeak_errs[valid_mask]
            eiso_err_clean = eiso_errs[valid_mask]
            ax.errorbar(ep_clean, eiso_clean, xerr=ep_err_clean, yerr=eiso_err_clean,
                       fmt="none", ecolor="#1f77b4", elinewidth=1, alpha=0.3, capsize=2)

        # Fit Amati relation: Eiso ~ Ep^beta
        log_ep = np.log10(ep_clean)
        log_eiso = np.log10(eiso_clean)

        # Linear fit in log space
        fit = np.polyfit(log_ep, log_eiso, 1)
        log_ep_fit = np.array([log_ep.min(), log_ep.max()])
        log_eiso_fit = np.polyval(fit, log_ep_fit)
        ep_fit = 10 ** log_ep_fit
        eiso_fit = 10 ** log_eiso_fit

        ax.plot(ep_fit, eiso_fit, "r--", linewidth=2.5, label=f"Best fit (β={fit[0]:.2f})")

        # 2-sigma scatter region
        residuals = log_eiso - np.polyval(fit, log_ep)
        sigma = np.std(residuals)
        for sign in [-2, 2]:
            log_eiso_scatter = np.polyval(fit, log_ep_fit) + sign * sigma
            eiso_scatter = 10 ** log_eiso_scatter
            ax.plot(ep_fit, eiso_scatter, "g:", linewidth=1.5, alpha=0.5)

        # Highlight specific GRB if requested
        if highlight_grb is not None and labels is not None:
            if highlight_grb in labels:
                idx = labels.index(highlight_grb)
                if valid_mask[idx]:
                    ax.scatter([epeaks[idx]], [eisos[idx]], s=200, color="red",
                             marker="*", edgecolors="darkred", linewidth=1.5,
                             label=f"Highlight: {highlight_grb}", zorder=5)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$E_{\rm peak}$ (keV)", fontsize=self.label_size)
        ax.set_ylabel(r"$E_{\rm iso}$ (erg)", fontsize=self.label_size)
        ax.set_title("Amati Relation", fontsize=self.title_size)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=self.config.get("legend_size", 10))

        fig.tight_layout()
        return fig

    def plot_yonetoku_relation(
        self,
        epeaks: np.ndarray,
        lpeaks: np.ndarray,
        epeak_errs: Optional[np.ndarray] = None,
        lpeak_errs: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
        highlight_grb: Optional[str] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot Epeak vs Lpeak (Yonetoku relation).

        Parameters
        ----------
        epeaks : np.ndarray
            Peak energy values (keV)
        lpeaks : np.ndarray
            Peak luminosity values (erg/s)
        epeak_errs : np.ndarray, optional
            Peak energy errors
        lpeak_errs : np.ndarray, optional
            Peak luminosity errors
        labels : list, optional
            GRB names/labels
        highlight_grb : str, optional
            Name of GRB to highlight
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Filter valid data
        valid_mask = (np.isfinite(epeaks) & np.isfinite(lpeaks) &
                     (epeaks > 0) & (lpeaks > 0))
        ep_clean = epeaks[valid_mask]
        lp_clean = lpeaks[valid_mask]

        # Plot data
        ax.scatter(ep_clean, lp_clean, s=80, color="#ff7f0e", alpha=0.6,
                  edgecolors="black", linewidth=0.5, label="GRBs")

        # Error bars
        if epeak_errs is not None and lpeak_errs is not None:
            ep_err_clean = epeak_errs[valid_mask]
            lp_err_clean = lpeak_errs[valid_mask]
            ax.errorbar(ep_clean, lp_clean, xerr=ep_err_clean, yerr=lp_err_clean,
                       fmt="none", ecolor="#ff7f0e", elinewidth=1, alpha=0.3, capsize=2)

        # Fit Yonetoku relation: Lpeak ~ Ep^gamma
        log_ep = np.log10(ep_clean)
        log_lp = np.log10(lp_clean)

        fit = np.polyfit(log_ep, log_lp, 1)
        log_ep_fit = np.array([log_ep.min(), log_ep.max()])
        log_lp_fit = np.polyval(fit, log_ep_fit)
        ep_fit = 10 ** log_ep_fit
        lp_fit = 10 ** log_lp_fit

        ax.plot(ep_fit, lp_fit, "r--", linewidth=2.5, label=f"Best fit (γ={fit[0]:.2f})")

        # 2-sigma scatter region
        residuals = log_lp - np.polyval(fit, log_ep)
        sigma = np.std(residuals)
        for sign in [-2, 2]:
            log_lp_scatter = np.polyval(fit, log_ep_fit) + sign * sigma
            lp_scatter = 10 ** log_lp_scatter
            ax.plot(ep_fit, lp_scatter, "g:", linewidth=1.5, alpha=0.5)

        # Highlight specific GRB if requested
        if highlight_grb is not None and labels is not None:
            if highlight_grb in labels:
                idx = labels.index(highlight_grb)
                if valid_mask[idx]:
                    ax.scatter([epeaks[idx]], [lpeaks[idx]], s=200, color="red",
                             marker="*", edgecolors="darkred", linewidth=1.5,
                             label=f"Highlight: {highlight_grb}", zorder=5)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$E_{\rm peak}$ (keV)", fontsize=self.label_size)
        ax.set_ylabel(r"$L_{\rm peak}$ (erg/s)", fontsize=self.label_size)
        ax.set_title("Yonetoku Relation", fontsize=self.title_size)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=self.config.get("legend_size", 10))

        fig.tight_layout()
        return fig

    def plot_ghirlanda_relation(
        self,
        epeaks: np.ndarray,
        egammas: np.ndarray,
        epeak_errs: Optional[np.ndarray] = None,
        egamma_errs: Optional[np.ndarray] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot collimation-corrected Epeak vs Egamma relation.

        Parameters
        ----------
        epeaks : np.ndarray
            Peak energy values (keV)
        egammas : np.ndarray
            Collimation-corrected energy (erg)
        epeak_errs : np.ndarray, optional
            Peak energy errors
        egamma_errs : np.ndarray, optional
            Egamma errors
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Filter valid data
        valid_mask = (np.isfinite(epeaks) & np.isfinite(egammas) &
                     (epeaks > 0) & (egammas > 0))
        ep_clean = epeaks[valid_mask]
        eg_clean = egammas[valid_mask]

        # Plot data
        ax.scatter(ep_clean, eg_clean, s=80, color="#2ca02c", alpha=0.6,
                  edgecolors="black", linewidth=0.5, label="GRBs")

        # Error bars
        if epeak_errs is not None and egamma_errs is not None:
            ep_err_clean = epeak_errs[valid_mask]
            eg_err_clean = egamma_errs[valid_mask]
            ax.errorbar(ep_clean, eg_clean, xerr=ep_err_clean, yerr=eg_err_clean,
                       fmt="none", ecolor="#2ca02c", elinewidth=1, alpha=0.3, capsize=2)

        # Fit relation
        log_ep = np.log10(ep_clean)
        log_eg = np.log10(eg_clean)

        fit = np.polyfit(log_ep, log_eg, 1)
        log_ep_fit = np.array([log_ep.min(), log_ep.max()])
        log_eg_fit = np.polyval(fit, log_ep_fit)
        ep_fit = 10 ** log_ep_fit
        eg_fit = 10 ** log_eg_fit

        ax.plot(ep_fit, eg_fit, "r--", linewidth=2.5, label=f"Best fit (slope={fit[0]:.2f})")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$E_{\rm peak}$ (keV)", fontsize=self.label_size)
        ax.set_ylabel(r"$E_\gamma$ (erg)", fontsize=self.label_size)
        ax.set_title("Ghirlanda Relation (Collimation-corrected)", fontsize=self.title_size)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=self.config.get("legend_size", 10))

        fig.tight_layout()
        return fig

    def plot_correlation_grid(
        self,
        catalog_data: pd.DataFrame,
        **kwargs
    ) -> plt.Figure:
        """
        Plot grid of major correlations in subplots.

        Parameters
        ----------
        catalog_data : pd.DataFrame
            DataFrame with columns: epeak, eiso, lpeak, t90, redshift
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=self.dpi)
        axes = axes.flatten()

        # Amati relation
        if "epeak" in catalog_data.columns and "eiso" in catalog_data.columns:
            valid = (np.isfinite(catalog_data["epeak"]) &
                    np.isfinite(catalog_data["eiso"]) &
                    (catalog_data["epeak"] > 0) & (catalog_data["eiso"] > 0))
            axes[0].scatter(catalog_data[valid]["epeak"], catalog_data[valid]["eiso"],
                          s=60, color="#1f77b4", alpha=0.6, edgecolors="black", linewidth=0.5)
            axes[0].set_xscale("log")
            axes[0].set_yscale("log")
            axes[0].set_xlabel(r"$E_{\rm peak}$ (keV)", fontsize=self.label_size)
            axes[0].set_ylabel(r"$E_{\rm iso}$ (erg)", fontsize=self.label_size)
            axes[0].set_title("Amati Relation", fontsize=self.label_size)
            axes[0].grid(True, alpha=0.3, which="both")

        # Yonetoku relation
        if "epeak" in catalog_data.columns and "lpeak" in catalog_data.columns:
            valid = (np.isfinite(catalog_data["epeak"]) &
                    np.isfinite(catalog_data["lpeak"]) &
                    (catalog_data["epeak"] > 0) & (catalog_data["lpeak"] > 0))
            axes[1].scatter(catalog_data[valid]["epeak"], catalog_data[valid]["lpeak"],
                          s=60, color="#ff7f0e", alpha=0.6, edgecolors="black", linewidth=0.5)
            axes[1].set_xscale("log")
            axes[1].set_yscale("log")
            axes[1].set_xlabel(r"$E_{\rm peak}$ (keV)", fontsize=self.label_size)
            axes[1].set_ylabel(r"$L_{\rm peak}$ (erg/s)", fontsize=self.label_size)
            axes[1].set_title("Yonetoku Relation", fontsize=self.label_size)
            axes[1].grid(True, alpha=0.3, which="both")

        # T90 vs Epeak
        if "t90" in catalog_data.columns and "epeak" in catalog_data.columns:
            valid = (np.isfinite(catalog_data["t90"]) &
                    np.isfinite(catalog_data["epeak"]) &
                    (catalog_data["t90"] > 0) & (catalog_data["epeak"] > 0))
            axes[2].scatter(catalog_data[valid]["t90"], catalog_data[valid]["epeak"],
                          s=60, color="#2ca02c", alpha=0.6, edgecolors="black", linewidth=0.5)
            axes[2].set_xscale("log")
            axes[2].set_yscale("log")
            axes[2].set_xlabel(r"T$_{90}$ (s)", fontsize=self.label_size)
            axes[2].set_ylabel(r"$E_{\rm peak}$ (keV)", fontsize=self.label_size)
            axes[2].set_title("Hardness-Duration", fontsize=self.label_size)
            axes[2].grid(True, alpha=0.3, which="both")

        # Epeak vs Redshift
        if "epeak" in catalog_data.columns and "redshift" in catalog_data.columns:
            valid = (np.isfinite(catalog_data["epeak"]) &
                    np.isfinite(catalog_data["redshift"]) &
                    (catalog_data["epeak"] > 0) & (catalog_data["redshift"] >= 0))
            axes[3].scatter(catalog_data[valid]["redshift"], catalog_data[valid]["epeak"],
                          s=60, color="#d62728", alpha=0.6, edgecolors="black", linewidth=0.5)
            axes[3].set_yscale("log")
            axes[3].set_xlabel(r"Redshift ($z$)", fontsize=self.label_size)
            axes[3].set_ylabel(r"$E_{\rm peak}$ (keV)", fontsize=self.label_size)
            axes[3].set_title("Epeak vs Redshift", fontsize=self.label_size)
            axes[3].grid(True, alpha=0.3)

        fig.suptitle("GRB Correlation Matrix", fontsize=self.title_size, y=0.995)
        fig.tight_layout()
        return fig

    def plot_luminosity_function(
        self,
        luminosities: np.ndarray,
        bins: int = 30,
        **kwargs
    ) -> plt.Figure:
        """
        Plot GRB luminosity function.

        Parameters
        ----------
        luminosities : np.ndarray
            Luminosity values (erg/s)
        bins : int
            Number of histogram bins
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Filter valid data
        valid_mask = np.isfinite(luminosities) & (luminosities > 0)
        lum_clean = luminosities[valid_mask]

        # Log scale histogram
        counts, edges, patches = ax.hist(np.log10(lum_clean), bins=bins,
                                         color="#9467bd", alpha=0.7, edgecolor="black",
                                         linewidth=1.2)

        # KDE overlay
        try:
            kde = gaussian_kde(np.log10(lum_clean))
            log_lum_range = np.linspace(np.log10(lum_clean).min(),
                                       np.log10(lum_clean).max(), 200)
            ax.plot(log_lum_range, kde(log_lum_range) * np.sum(counts) * (edges[1] - edges[0]),
                   color="#d62728", linewidth=2.5, label="KDE")
        except:
            pass

        ax.set_xlabel(r"$\log_{10}$(Luminosity [erg/s])", fontsize=self.label_size)
        ax.set_ylabel("Count", fontsize=self.label_size)
        ax.set_title("GRB Luminosity Function", fontsize=self.title_size)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=self.config.get("legend_size", 10))

        fig.tight_layout()
        return fig

    def plot_rate_vs_redshift(
        self,
        redshifts: np.ndarray,
        bins: int = 20,
        **kwargs
    ) -> plt.Figure:
        """
        Plot GRB rate as function of redshift.

        Parameters
        ----------
        redshifts : np.ndarray
            Redshift values
        bins : int
            Number of redshift bins
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Filter valid data
        valid_mask = np.isfinite(redshifts) & (redshifts >= 0)
        z_clean = redshifts[valid_mask]

        # Histogram of redshifts
        counts, edges, _ = ax.hist(z_clean, bins=bins, color="#1f77b4", alpha=0.7,
                                   edgecolor="black", linewidth=1.2, label="GRB Rate")

        # Bin centers
        z_centers = (edges[:-1] + edges[1:]) / 2

        # Overlay trend
        if len(z_centers) > 2:
            ax.plot(z_centers, counts, "r-", linewidth=2.5, alpha=0.7, label="Trend")

        # Mark redshift range
        ax.axvline(np.median(z_clean), color="green", linestyle="--", linewidth=2,
                  label=f"Median z: {np.median(z_clean):.2f}")

        ax.set_xlabel(r"Redshift ($z$)", fontsize=self.label_size)
        ax.set_ylabel("Number of GRBs", fontsize=self.label_size)
        ax.set_title("GRB Rate vs Redshift", fontsize=self.title_size)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=self.config.get("legend_size", 10))

        fig.tight_layout()
        return fig

    def plot_classification_pie(
        self,
        classifications: Dict[str, int],
        **kwargs
    ) -> plt.Figure:
        """
        Plot pie/donut chart of GRB classifications.

        Parameters
        ----------
        classifications : dict
            Dictionary with classification types as keys and counts as values
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=(8, 8), dpi=self.dpi)

        labels = list(classifications.keys())
        sizes = list(classifications.values())
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        colors = colors[:len(labels)]

        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct="%1.1f%%",
                                           colors=colors, startangle=90, textprops={
                                               "fontsize": self.label_size
                                           })

        # Draw circle for donut chart
        centre_circle = plt.Circle((0, 0), 0.70, fc="white")
        ax.add_artist(centre_circle)

        ax.set_title("GRB Classification Distribution", fontsize=self.title_size)

        fig.tight_layout()
        return fig

    def plot_mission_timeline(
        self,
        missions_data: Dict[str, Dict],
        **kwargs
    ) -> plt.Figure:
        """
        Plot timeline of GRB detections by mission.

        Parameters
        ----------
        missions_data : dict
            Dictionary with mission names as keys and dicts containing:
            - "start_year": mission start year
            - "end_year": mission end year
            - "count": number of GRBs detected
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=(14, 6), dpi=self.dpi)

        missions = list(missions_data.keys())
        colors = plt.cm.tab20(np.linspace(0, 1, len(missions)))

        for idx, mission in enumerate(missions):
            data = missions_data[mission]
            start = data.get("start_year", 2000)
            end = data.get("end_year", 2025)
            count = data.get("count", 0)

            # Draw horizontal bar
            ax.barh(mission, end - start, left=start, height=0.6, color=colors[idx],
                   edgecolor="black", linewidth=1)

            # Add count label
            mid_year = (start + end) / 2
            ax.text(mid_year, idx, str(count), va="center", ha="center",
                   fontweight="bold", fontsize=self.label_size)

        ax.set_xlabel("Year", fontsize=self.label_size)
        ax.set_title("GRB Detection Timeline by Mission", fontsize=self.title_size)
        ax.set_ylim(-0.5, len(missions) - 0.5)
        ax.set_xlim(1990, 2030)
        ax.grid(True, axis="x", alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_parameter_corner(
        self,
        data: pd.DataFrame,
        params: List[str],
        **kwargs
    ) -> plt.Figure:
        """
        Corner plot showing correlations between multiple parameters.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the parameters
        params : list
            List of parameter column names to plot
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        n_params = len(params)
        fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12), dpi=self.dpi)

        for i in range(n_params):
            for j in range(n_params):
                ax = axes[i, j]

                if i == j:
                    # Diagonal: histogram
                    valid = np.isfinite(data[params[i]])
                    ax.hist(data[valid][params[i]], bins=20, color="#1f77b4",
                           alpha=0.7, edgecolor="black")
                    ax.set_ylabel("Count", fontsize=9)
                elif i > j:
                    # Lower triangle: scatter plots
                    valid = (np.isfinite(data[params[i]]) &
                            np.isfinite(data[params[j]]))
                    ax.scatter(data[valid][params[j]], data[valid][params[i]],
                             s=30, color="#1f77b4", alpha=0.5, edgecolors="black",
                             linewidth=0.3)
                    ax.set_ylabel(params[i], fontsize=9)
                    ax.set_xlabel(params[j], fontsize=9)
                    ax.grid(True, alpha=0.3)
                else:
                    # Upper triangle: empty or correlation value
                    valid = (np.isfinite(data[params[i]]) &
                            np.isfinite(data[params[j]]))
                    if len(data[valid]) > 1:
                        corr = np.corrcoef(data[valid][params[j]],
                                          data[valid][params[i]])[0, 1]
                        ax.text(0.5, 0.5, f"{corr:.2f}", ha="center", va="center",
                               fontsize=12, fontweight="bold",
                               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
                    ax.set_xticks([])
                    ax.set_yticks([])

                # Remove labels for internal plots
                if j > 0 and i != j:
                    ax.set_ylabel("")
                if i < n_params - 1:
                    ax.set_xlabel("")

        fig.suptitle("Parameter Corner Plot", fontsize=self.title_size, y=0.995)
        fig.tight_layout()
        return fig

    def plot_afterglow_comparison(
        self,
        afterglows: List[Dict[str, np.ndarray]],
        **kwargs
    ) -> plt.Figure:
        """
        Plot multiple afterglow light curves overlaid, normalized to reference time.

        Parameters
        ----------
        afterglows : list
            List of dicts with keys:
            - "time": time array (seconds)
            - "flux": flux array
            - "flux_err": flux errors
            - "name": GRB name
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        colors = plt.cm.tab10(np.linspace(0, 1, len(afterglows)))

        for idx, ag in enumerate(afterglows):
            time = ag.get("time", np.array([]))
            flux = ag.get("flux", np.array([]))
            flux_err = ag.get("flux_err", np.array([]))
            name = ag.get("name", f"GRB {idx+1}")

            # Filter valid data
            valid = np.isfinite(time) & np.isfinite(flux) & (flux > 0)
            time_clean = time[valid]
            flux_clean = flux[valid]

            if len(time_clean) > 0:
                # Normalize to first time point
                t_ref = time_clean[0]
                t_normalized = time_clean - t_ref

                ax.loglog(t_normalized + 1e-2, flux_clean, color=colors[idx],
                         marker="o", markersize=5, linewidth=2, alpha=0.7, label=name)

                if len(flux_err) > 0:
                    flux_err_clean = flux_err[valid]
                    ax.loglog(t_normalized + 1e-2, flux_clean + flux_err_clean,
                             color=colors[idx], linestyle=":", alpha=0.3, linewidth=1)
                    ax.loglog(t_normalized + 1e-2, flux_clean - flux_err_clean,
                             color=colors[idx], linestyle=":", alpha=0.3, linewidth=1)

        ax.set_xlabel(r"Time since trigger (s)", fontsize=self.label_size)
        ax.set_ylabel(r"Flux (arbitrary units)", fontsize=self.label_size)
        ax.set_title("Afterglow Comparison", fontsize=self.title_size)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=self.config.get("legend_size", 10))

        fig.tight_layout()
        return fig
