"""Standard plotting utilities for GRB light curves and distributions."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from scipy import stats
from scipy.stats import gaussian_kde


class StandardPlotter:
    """Plotter for standard GRB analysis visualizations."""

    def __init__(self, config: Optional[Dict] = None, style: str = "publication"):
        """
        Initialize the StandardPlotter with configuration and style.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary for plot parameters
        style : str
            Style preset: "publication" or "presentation"
        """
        self.config = config or {}
        self.style = style

        # Set up matplotlib style
        self._setup_publication_style() if style == "publication" else self._setup_presentation_style()

        # Default figure parameters
        self.figsize = self.config.get("figsize", (10, 6))
        self.dpi = self.config.get("dpi", 300)
        self.font_size = self.config.get("font_size", 11)
        self.label_size = self.config.get("label_size", 12)
        self.title_size = self.config.get("title_size", 14)

    def _setup_publication_style(self):
        """Configure matplotlib rcParams for publication quality."""
        rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.2,
            "axes.labelsize": self.config.get("label_size", 12),
            "axes.titlesize": self.config.get("title_size", 14),
            "xtick.labelsize": self.config.get("tick_size", 11),
            "ytick.labelsize": self.config.get("tick_size", 11),
            "legend.fontsize": self.config.get("legend_size", 10),
            "font.size": self.config.get("font_size", 11),
            "font.family": "serif",
            "font.serif": ["Times", "DejaVu Serif"],
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "patch.linewidth": 0.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.loc": "best",
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        })

    def _setup_presentation_style(self):
        """Configure matplotlib rcParams for presentation quality."""
        rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.5,
            "axes.labelsize": 13,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "font.size": 12,
            "font.family": "sans-serif",
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            "patch.linewidth": 1,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
        })

    def plot_lightcurve(
        self,
        time: np.ndarray,
        rate: np.ndarray,
        rate_err: np.ndarray,
        energy_band: Optional[str] = None,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        t90: Optional[Tuple[float, float]] = None,
        background: Optional[float] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot a standard GRB light curve with error bars.

        Parameters
        ----------
        time : np.ndarray
            Time array (seconds)
        rate : np.ndarray
            Count rate array
        rate_err : np.ndarray
            Count rate errors
        energy_band : str, optional
            Energy band description (e.g., "15-350 keV")
        title : str, optional
            Plot title
        ax : plt.Axes, optional
            Existing axes to plot on
        t90 : tuple, optional
            (t90_start, t90_end) tuple to mark T90 interval
        background : float, optional
            Background rate level to display
        **kwargs
            Additional keyword arguments for plot customization

        Returns
        -------
        plt.Figure
            The figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = ax.get_figure()

        # Plot with error bars
        color = kwargs.get("color", "#1f77b4")
        ax.errorbar(time, rate, yerr=rate_err, fmt="o", color=color,
                   ecolor=color, alpha=0.7, markersize=4, elinewidth=1.5,
                   capsize=2, label="Data")

        # Mark background level if provided
        if background is not None:
            ax.axhline(background, color="red", linestyle="--", linewidth=1.5,
                      label=f"Background: {background:.2f} cts/s")

        # Mark T90 interval if provided
        if t90 is not None:
            t90_start, t90_end = t90
            ax.axvspan(t90_start, t90_end, alpha=0.2, color="yellow",
                      label=f"T90: {t90_end - t90_start:.2f} s")

        # Log-log scaling
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Labels and formatting
        ax.set_xlabel(r"Time (s)", fontsize=self.label_size)
        ax.set_ylabel(r"Count Rate (cts/s)", fontsize=self.label_size)

        if title:
            ax.set_title(title, fontsize=self.title_size)
        elif energy_band:
            ax.set_title(f"Light Curve: {energy_band}", fontsize=self.title_size)

        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=self.config.get("legend_size", 10))
        fig.tight_layout()

        return fig

    def plot_multi_band_lightcurve(
        self,
        lightcurves: Dict[str, Dict[str, np.ndarray]],
        title: Optional[str] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot multiple energy band light curves stacked vertically.

        Parameters
        ----------
        lightcurves : dict
            Dictionary with band names as keys and dicts of {time, rate, rate_err} as values
        title : str, optional
            Overall figure title
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        n_bands = len(lightcurves)
        fig, axes = plt.subplots(n_bands, 1, figsize=(self.figsize[0], 3*n_bands),
                                  dpi=self.dpi, sharex=True)

        if n_bands == 1:
            axes = [axes]

        # Color palette (colorblind-friendly)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for idx, (band_name, data) in enumerate(lightcurves.items()):
            ax = axes[idx]
            time = data.get("time", np.array([]))
            rate = data.get("rate", np.array([]))
            rate_err = data.get("rate_err", np.array([]))

            color = colors[idx % len(colors)]
            ax.errorbar(time, rate, yerr=rate_err, fmt="o", color=color,
                       ecolor=color, alpha=0.7, markersize=4, elinewidth=1.5,
                       capsize=2)

            ax.set_yscale("log")
            ax.set_ylabel(f"{band_name}\n(cts/s)", fontsize=self.label_size)
            ax.grid(True, which="both", alpha=0.3)

        axes[-1].set_xscale("log")
        axes[-1].set_xlabel(r"Time (s)", fontsize=self.label_size)

        if title:
            fig.suptitle(title, fontsize=self.title_size, y=0.995)
        else:
            fig.suptitle("Multi-Band Light Curves", fontsize=self.title_size, y=0.995)

        fig.tight_layout()
        return fig

    def plot_t90_distribution(
        self,
        t90_values: np.ndarray,
        t90_errors: Optional[np.ndarray] = None,
        bins: int = 50,
        **kwargs
    ) -> plt.Figure:
        """
        Plot T90 distribution histogram showing bimodal short/long GRB distribution.

        Parameters
        ----------
        t90_values : np.ndarray
            T90 durations in seconds
        t90_errors : np.ndarray, optional
            T90 uncertainties
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
        valid_mask = np.isfinite(t90_values) & (t90_values > 0)
        t90_clean = t90_values[valid_mask]

        # Histogram
        counts, edges, patches = ax.hist(np.log10(t90_clean), bins=bins,
                                         color="#1f77b4", alpha=0.7, edgecolor="black",
                                         linewidth=1.2)

        # KDE overlay
        try:
            kde = gaussian_kde(np.log10(t90_clean))
            log_t90_range = np.linspace(np.log10(t90_clean).min(),
                                        np.log10(t90_clean).max(), 200)
            ax.plot(log_t90_range, kde(log_t90_range) * np.sum(counts) * (edges[1] - edges[0]),
                   color="#d62728", linewidth=2.5, label="KDE")
        except:
            pass

        # Mark short/long boundary at T90 = 2s (log10(2) ≈ 0.301)
        boundary = np.log10(2)
        ax.axvline(boundary, color="red", linestyle="--", linewidth=2,
                  label="Short/Long Boundary (2 s)")

        ax.set_xlabel(r"$\log_{10}(T_{90})$ (s)", fontsize=self.label_size)
        ax.set_ylabel("Count", fontsize=self.label_size)
        ax.set_title("T90 Duration Distribution", fontsize=self.title_size)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=self.config.get("legend_size", 10))

        fig.tight_layout()
        return fig

    def plot_hardness_duration(
        self,
        t90_values: np.ndarray,
        hardness_ratios: np.ndarray,
        classifications: Optional[np.ndarray] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot T90 vs hardness ratio scatter plot with classification regions.

        Parameters
        ----------
        t90_values : np.ndarray
            T90 durations
        hardness_ratios : np.ndarray
            Hardness ratios (e.g., hard/soft count ratio)
        classifications : np.ndarray, optional
            Classification labels ("short" or "long")
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        fig = plt.figure(figsize=(10, 8), dpi=self.dpi)
        gs = fig.add_gridspec(3, 3, hspace=0.05, wspace=0.05)

        # Main scatter plot
        ax_main = fig.add_subplot(gs[1:, :-1])

        # Filter valid data
        valid_mask = np.isfinite(t90_values) & np.isfinite(hardness_ratios)
        t90_clean = t90_values[valid_mask]
        hardness_clean = hardness_ratios[valid_mask]

        if classifications is not None:
            classes_clean = classifications[valid_mask]
            short_mask = np.array([c.lower() == "short" for c in classes_clean])

            ax_main.scatter(t90_clean[short_mask], hardness_clean[short_mask],
                          color="#1f77b4", s=60, alpha=0.6, label="Short", marker="o")
            ax_main.scatter(t90_clean[~short_mask], hardness_clean[~short_mask],
                          color="#ff7f0e", s=60, alpha=0.6, label="Long", marker="s")
        else:
            ax_main.scatter(t90_clean, hardness_clean, color="#1f77b4", s=60,
                          alpha=0.6, marker="o")

        # Mark boundary region
        ax_main.axvline(2, color="red", linestyle="--", linewidth=2, alpha=0.7)
        ax_main.axhspan(ax_main.get_ylim()[0], ax_main.get_ylim()[1],
                       xmin=0, xmax=0.3, alpha=0.1, color="blue", label="Short GRB Region")
        ax_main.axhspan(ax_main.get_ylim()[0], ax_main.get_ylim()[1],
                       xmin=0.3, xmax=1, alpha=0.1, color="orange", label="Long GRB Region")

        ax_main.set_xscale("log")
        ax_main.set_xlabel(r"T$_{90}$ (s)", fontsize=self.label_size)
        ax_main.set_ylabel("Hardness Ratio", fontsize=self.label_size)
        ax_main.set_title("Hardness-Duration Diagram", fontsize=self.title_size)
        ax_main.grid(True, alpha=0.3, which="both")
        ax_main.legend(loc="best", fontsize=self.config.get("legend_size", 10))

        # Top histogram (T90)
        ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
        ax_top.hist(np.log10(t90_clean), bins=30, color="#1f77b4", alpha=0.7, edgecolor="black")
        ax_top.set_ylabel("Count", fontsize=10)
        ax_top.tick_params(labelbottom=False)
        ax_top.grid(True, alpha=0.3)

        # Right histogram (Hardness)
        ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)
        ax_right.hist(hardness_clean, bins=30, orientation="horizontal",
                     color="#1f77b4", alpha=0.7, edgecolor="black")
        ax_right.set_xlabel("Count", fontsize=10)
        ax_right.tick_params(labelleft=False)
        ax_right.grid(True, alpha=0.3)

        return fig

    def plot_sky_map(
        self,
        ra_list: np.ndarray,
        dec_list: np.ndarray,
        names: Optional[List[str]] = None,
        projection: str = "aitoff",
        property_values: Optional[np.ndarray] = None,
        property_label: Optional[str] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot all-sky GRB positions in Aitoff or Mollweide projection.

        Parameters
        ----------
        ra_list : np.ndarray
            Right ascension values (degrees)
        dec_list : np.ndarray
            Declination values (degrees)
        names : list, optional
            GRB names for annotation
        projection : str
            "aitoff" or "mollweide"
        property_values : np.ndarray, optional
            Values for coloring points (e.g., redshift, T90)
        property_label : str, optional
            Label for the color property
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        fig = plt.figure(figsize=(12, 8), dpi=self.dpi)
        ax = fig.add_subplot(111, projection=projection)

        # Convert to radians
        ra_rad = np.radians(ra_list)
        dec_rad = np.radians(dec_list)

        # Filter valid coordinates
        valid_mask = np.isfinite(ra_rad) & np.isfinite(dec_rad)
        ra_clean = ra_rad[valid_mask]
        dec_clean = dec_rad[valid_mask]

        # Plot points with optional coloring
        if property_values is not None:
            property_clean = property_values[valid_mask]
            scatter = ax.scatter(ra_clean, dec_clean, c=property_clean, s=100,
                               cmap="viridis", alpha=0.7, edgecolors="black", linewidth=0.5)
            cbar = plt.colorbar(scatter, ax=ax, orientation="vertical", pad=0.1)
            if property_label:
                cbar.set_label(property_label, fontsize=self.label_size)
        else:
            ax.scatter(ra_clean, dec_clean, s=100, color="#1f77b4", alpha=0.7,
                     edgecolors="black", linewidth=0.5)

        # Annotate with names if provided
        if names is not None:
            names_clean = [names[i] for i in range(len(names)) if valid_mask[i]]
            for ra, dec, name in zip(ra_clean, dec_clean, names_clean[:10]):  # Limit labels
                ax.annotate(name, (ra, dec), fontsize=8, alpha=0.7)

        ax.set_xlabel(r"Right Ascension (rad)", fontsize=self.label_size)
        ax.set_ylabel(r"Declination (rad)", fontsize=self.label_size)
        ax.set_title("All-Sky GRB Distribution", fontsize=self.title_size)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_redshift_distribution(
        self,
        redshifts: np.ndarray,
        bins: int = 30,
        **kwargs
    ) -> plt.Figure:
        """
        Plot redshift histogram with KDE overlay.

        Parameters
        ----------
        redshifts : np.ndarray
            Redshift values
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
        valid_mask = np.isfinite(redshifts) & (redshifts >= 0)
        z_clean = redshifts[valid_mask]

        # Histogram
        counts, edges, patches = ax.hist(z_clean, bins=bins, color="#1f77b4",
                                         alpha=0.7, edgecolor="black", linewidth=1.2)

        # KDE overlay
        try:
            kde = gaussian_kde(z_clean)
            z_range = np.linspace(z_clean.min(), z_clean.max(), 200)
            ax.plot(z_range, kde(z_range) * np.sum(counts) * (edges[1] - edges[0]),
                   color="#d62728", linewidth=2.5, label="KDE")
        except:
            pass

        # Mark statistics
        median_z = np.median(z_clean)
        mean_z = np.mean(z_clean)
        ax.axvline(median_z, color="green", linestyle="--", linewidth=2,
                  label=f"Median: {median_z:.2f}")
        ax.axvline(mean_z, color="orange", linestyle="--", linewidth=2,
                  label=f"Mean: {mean_z:.2f}")

        ax.set_xlabel(r"Redshift ($z$)", fontsize=self.label_size)
        ax.set_ylabel("Count", fontsize=self.label_size)
        ax.set_title("Redshift Distribution", fontsize=self.title_size)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=self.config.get("legend_size", 10))

        fig.tight_layout()
        return fig

    def plot_fluence_distribution(
        self,
        fluences: np.ndarray,
        bins: int = 30,
        **kwargs
    ) -> plt.Figure:
        """
        Plot fluence distribution (log-normal).

        Parameters
        ----------
        fluences : np.ndarray
            Fluence values (erg/cm^2)
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
        valid_mask = np.isfinite(fluences) & (fluences > 0)
        fluence_clean = fluences[valid_mask]

        # Log scale histogram
        counts, edges, patches = ax.hist(np.log10(fluence_clean), bins=bins,
                                         color="#2ca02c", alpha=0.7, edgecolor="black",
                                         linewidth=1.2)

        # KDE overlay
        try:
            kde = gaussian_kde(np.log10(fluence_clean))
            log_fluence_range = np.linspace(np.log10(fluence_clean).min(),
                                           np.log10(fluence_clean).max(), 200)
            ax.plot(log_fluence_range, kde(log_fluence_range) * np.sum(counts) * (edges[1] - edges[0]),
                   color="#d62728", linewidth=2.5, label="KDE")
        except:
            pass

        ax.set_xlabel(r"$\log_{10}$(Fluence [erg cm$^{-2}$])", fontsize=self.label_size)
        ax.set_ylabel("Count", fontsize=self.label_size)
        ax.set_title("Fluence Distribution", fontsize=self.title_size)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=self.config.get("legend_size", 10))

        fig.tight_layout()
        return fig

    def plot_peak_flux_distribution(
        self,
        peak_fluxes: np.ndarray,
        bins: int = 30,
        **kwargs
    ) -> plt.Figure:
        """
        Plot peak flux distribution.

        Parameters
        ----------
        peak_fluxes : np.ndarray
            Peak flux values (photons/cm^2/s)
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
        valid_mask = np.isfinite(peak_fluxes) & (peak_fluxes > 0)
        pf_clean = peak_fluxes[valid_mask]

        # Log scale histogram
        counts, edges, patches = ax.hist(np.log10(pf_clean), bins=bins,
                                         color="#d62728", alpha=0.7, edgecolor="black",
                                         linewidth=1.2)

        # KDE overlay
        try:
            kde = gaussian_kde(np.log10(pf_clean))
            log_pf_range = np.linspace(np.log10(pf_clean).min(),
                                      np.log10(pf_clean).max(), 200)
            ax.plot(log_pf_range, kde(log_pf_range) * np.sum(counts) * (edges[1] - edges[0]),
                   color="#1f77b4", linewidth=2.5, label="KDE")
        except:
            pass

        ax.set_xlabel(r"$\log_{10}$(Peak Flux [ph cm$^{-2}$ s$^{-1}$])",
                     fontsize=self.label_size)
        ax.set_ylabel("Count", fontsize=self.label_size)
        ax.set_title("Peak Flux Distribution", fontsize=self.title_size)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=self.config.get("legend_size", 10))

        fig.tight_layout()
        return fig

    def plot_prompt_emission_summary(
        self,
        grb_data: Dict,
        **kwargs
    ) -> plt.Figure:
        """
        Multi-panel summary: light curve, cumulative counts, hardness evolution, spectrum.

        Parameters
        ----------
        grb_data : dict
            Dictionary containing:
            - "time": time array
            - "rate": count rate array
            - "rate_err": count rate errors
            - "hardness": hardness ratio evolution
            - "hardness_err": hardness uncertainties
            - "energy": energy array for spectrum
            - "flux": flux array
            - "flux_err": flux errors
            - (optional) "t90": (t90_start, t90_end)
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        fig = plt.figure(figsize=(14, 10), dpi=self.dpi)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Light curve
        ax1 = fig.add_subplot(gs[0, 0])
        time = grb_data.get("time", np.array([]))
        rate = grb_data.get("rate", np.array([]))
        rate_err = grb_data.get("rate_err", np.array([]))

        ax1.errorbar(time, rate, yerr=rate_err, fmt="o", color="#1f77b4",
                    ecolor="#1f77b4", alpha=0.7, markersize=4, elinewidth=1.5, capsize=2)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel(r"Time (s)", fontsize=self.label_size)
        ax1.set_ylabel(r"Count Rate (cts/s)", fontsize=self.label_size)
        ax1.set_title("Light Curve", fontsize=self.label_size)
        ax1.grid(True, which="both", alpha=0.3)

        # Cumulative counts
        ax2 = fig.add_subplot(gs[0, 1])
        cumulative = np.cumsum(rate)
        ax2.plot(time, cumulative, color="#2ca02c", linewidth=2)
        t90 = grb_data.get("t90")
        if t90 is not None:
            t90_start, t90_end = t90
            ax2.axvline(t90_start, color="red", linestyle="--", alpha=0.7)
            ax2.axvline(t90_end, color="red", linestyle="--", alpha=0.7)
        ax2.set_xscale("log")
        ax2.set_xlabel(r"Time (s)", fontsize=self.label_size)
        ax2.set_ylabel("Cumulative Counts", fontsize=self.label_size)
        ax2.set_title("Cumulative Counts", fontsize=self.label_size)
        ax2.grid(True, alpha=0.3)

        # Hardness evolution
        ax3 = fig.add_subplot(gs[1, 0])
        hardness = grb_data.get("hardness")
        if hardness is not None:
            hardness_err = grb_data.get("hardness_err", np.zeros_like(hardness))
            ax3.errorbar(time, hardness, yerr=hardness_err, fmt="o", color="#ff7f0e",
                        ecolor="#ff7f0e", alpha=0.7, markersize=4, elinewidth=1.5, capsize=2)
        ax3.set_xscale("log")
        ax3.set_xlabel(r"Time (s)", fontsize=self.label_size)
        ax3.set_ylabel("Hardness Ratio", fontsize=self.label_size)
        ax3.set_title("Hardness Evolution", fontsize=self.label_size)
        ax3.grid(True, alpha=0.3)

        # Spectrum
        ax4 = fig.add_subplot(gs[1, 1])
        energy = grb_data.get("energy")
        flux = grb_data.get("flux")
        flux_err = grb_data.get("flux_err")
        if energy is not None and flux is not None:
            ax4.errorbar(energy, flux, yerr=flux_err, fmt="o", color="#d62728",
                        ecolor="#d62728", alpha=0.7, markersize=4, elinewidth=1.5, capsize=2)
            ax4.set_xscale("log")
            ax4.set_yscale("log")
        ax4.set_xlabel(r"Energy (keV)", fontsize=self.label_size)
        ax4.set_ylabel(r"nuFnu (erg cm$^{-2}$ s$^{-1}$)", fontsize=self.label_size)
        ax4.set_title("Energy Spectrum", fontsize=self.label_size)
        ax4.grid(True, which="both", alpha=0.3)

        fig.suptitle("Prompt Emission Summary", fontsize=self.title_size, y=0.995)
        return fig

    def save_figure(
        self,
        fig: plt.Figure,
        filename: Union[str, Path],
        formats: List[str] = ["png", "pdf"]
    ) -> List[str]:
        """
        Save figure in multiple formats.

        Parameters
        ----------
        fig : plt.Figure
            Figure to save
        filename : str or Path
            Output filename without extension
        formats : list
            List of formats to save ("png", "pdf", "eps", etc.)

        Returns
        -------
        list
            List of saved file paths
        """
        filename = Path(filename)
        saved_files = []

        for fmt in formats:
            output_path = filename.with_suffix(f".{fmt}")
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight",
                       facecolor="white", edgecolor="none")
            saved_files.append(str(output_path))

        return saved_files
