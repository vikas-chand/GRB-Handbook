"""Spectral analysis plotting utilities for GRB analysis."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Callable
from scipy.stats import gaussian_kde


class SpectralPlotter:
    """Plotter for GRB spectral analysis visualizations."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the SpectralPlotter.

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
        """Configure matplotlib rcParams for spectral plots."""
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

    def plot_spectrum(
        self,
        energy: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        model_func: Optional[Callable] = None,
        model_params: Optional[Dict] = None,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot nuFnu spectrum with optional model overlay and residuals panel.

        Parameters
        ----------
        energy : np.ndarray
            Energy array (keV)
        flux : np.ndarray
            Energy flux array (nuFnu in erg cm^-2 s^-1)
        flux_err : np.ndarray
            Flux errors
        model_func : callable, optional
            Function to compute model flux: model_func(energy, **model_params)
        model_params : dict, optional
            Parameters for model function
        title : str, optional
            Plot title
        ax : plt.Axes, optional
            Existing axes (uses new figure with subplots if None)
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        if ax is None:
            fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1]*1.3),
                                             dpi=self.dpi, gridspec_kw={"height_ratios": [3, 1]})
        else:
            fig = ax.get_figure()
            ax_res = None

        # Filter valid data
        valid_mask = np.isfinite(energy) & np.isfinite(flux) & np.isfinite(flux_err) & (flux > 0)
        energy_clean = energy[valid_mask]
        flux_clean = flux[valid_mask]
        flux_err_clean = flux_err[valid_mask]

        # Plot data
        ax.errorbar(energy_clean, flux_clean, yerr=flux_err_clean, fmt="o",
                   color="#1f77b4", ecolor="#1f77b4", alpha=0.7, markersize=6,
                   elinewidth=1.5, capsize=3, label="Data")

        # Plot model if provided
        model_flux = None
        if model_func is not None and model_params is not None:
            energy_model = np.logspace(np.log10(energy_clean.min()),
                                       np.log10(energy_clean.max()), 200)
            model_flux = model_func(energy_model, **model_params)
            ax.plot(energy_model, model_flux, color="#d62728", linewidth=2.5,
                   label="Model")

        # Residuals panel
        if ax_res is not None and model_flux is not None:
            model_at_data = model_func(energy_clean, **model_params)
            residuals = (flux_clean - model_at_data) / flux_err_clean
            ax_res.errorbar(energy_clean, residuals, yerr=np.ones_like(residuals),
                           fmt="o", color="#1f77b4", ecolor="#1f77b4", alpha=0.7,
                           markersize=5, elinewidth=1.5, capsize=2)
            ax_res.axhline(0, color="red", linestyle="--", linewidth=1.5)
            ax_res.axhline(2, color="gray", linestyle=":", alpha=0.5)
            ax_res.axhline(-2, color="gray", linestyle=":", alpha=0.5)
            ax_res.set_ylabel(r"$\sigma$ residuals", fontsize=self.label_size)
            ax_res.set_xscale("log")
            ax_res.grid(True, alpha=0.3)

        # Formatting
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Energy (keV)", fontsize=self.label_size)
        ax.set_ylabel(r"$\nu F_\nu$ (erg cm$^{-2}$ s$^{-1}$)", fontsize=self.label_size)

        if title:
            ax.set_title(title, fontsize=self.title_size)
        else:
            ax.set_title("Energy Spectrum", fontsize=self.title_size)

        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=self.config.get("legend_size", 10))

        if ax_res is not None:
            ax_res.set_xlabel(r"Energy (keV)", fontsize=self.label_size)

        fig.tight_layout()
        return fig

    def plot_spectral_fit(
        self,
        data: Dict[str, np.ndarray],
        fit_result: Dict,
        model_name: str = "Band",
        **kwargs
    ) -> plt.Figure:
        """
        Two-panel spectral fit plot with data, model, and residuals.

        Parameters
        ----------
        data : dict
            Dictionary with keys: "energy", "flux", "flux_err"
        fit_result : dict
            Dictionary with keys: "params", "errors", "chi2", "dof", "model_func"
        model_name : str
            Name of the spectral model
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1]*1.3),
                                         dpi=self.dpi, gridspec_kw={"height_ratios": [3, 1]})

        energy = data.get("energy", np.array([]))
        flux = data.get("flux", np.array([]))
        flux_err = data.get("flux_err", np.array([]))

        # Filter valid data
        valid_mask = np.isfinite(energy) & np.isfinite(flux) & np.isfinite(flux_err) & (flux > 0)
        energy_clean = energy[valid_mask]
        flux_clean = flux[valid_mask]
        flux_err_clean = flux_err[valid_mask]

        # Plot data
        ax.errorbar(energy_clean, flux_clean, yerr=flux_err_clean, fmt="o",
                   color="#1f77b4", ecolor="#1f77b4", alpha=0.7, markersize=6,
                   elinewidth=1.5, capsize=3, label="Data")

        # Plot model
        model_func = fit_result.get("model_func")
        params = fit_result.get("params", {})

        if model_func is not None:
            energy_model = np.logspace(np.log10(energy_clean.min()),
                                       np.log10(energy_clean.max()), 300)
            model_flux = model_func(energy_model, **params)
            ax.plot(energy_model, model_flux, color="#d62728", linewidth=2.5,
                   label=f"{model_name} Model")

            # Residuals
            model_at_data = model_func(energy_clean, **params)
            residuals = (flux_clean - model_at_data) / flux_err_clean

            ax_res.errorbar(energy_clean, residuals, yerr=np.ones_like(residuals),
                           fmt="o", color="#1f77b4", ecolor="#1f77b4", alpha=0.7,
                           markersize=5, elinewidth=1.5, capsize=2)
            ax_res.axhline(0, color="red", linestyle="--", linewidth=1.5)
            ax_res.axhline(2, color="gray", linestyle=":", alpha=0.5, linewidth=1)
            ax_res.axhline(-2, color="gray", linestyle=":", alpha=0.5, linewidth=1)

        # Text box with fit parameters
        textstr = f"{model_name} Fit Results:\n"
        for param_name, param_val in params.items():
            error = fit_result.get("errors", {}).get(param_name, 0)
            if isinstance(error, (int, float)) and error > 0:
                textstr += f"{param_name}: {param_val:.3f} ± {error:.3f}\n"
            else:
                textstr += f"{param_name}: {param_val:.3f}\n"

        chi2 = fit_result.get("chi2")
        dof = fit_result.get("dof")
        if chi2 is not None and dof is not None:
            textstr += f"\nχ²/dof: {chi2/dof:.2f}"

        ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        # Formatting
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"$\nu F_\nu$ (erg cm$^{-2}$ s$^{-1}$)", fontsize=self.label_size)
        ax.set_title(f"Spectral Fit: {model_name}", fontsize=self.title_size)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=self.config.get("legend_size", 10))

        ax_res.set_xscale("log")
        ax_res.set_xlabel(r"Energy (keV)", fontsize=self.label_size)
        ax_res.set_ylabel(r"$\sigma$ residuals", fontsize=self.label_size)
        ax_res.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_model_comparison(
        self,
        data: Dict[str, np.ndarray],
        fits: List[Dict],
        **kwargs
    ) -> plt.Figure:
        """
        Compare multiple spectral models on same data.

        Parameters
        ----------
        data : dict
            Dictionary with keys: "energy", "flux", "flux_err"
        fits : list
            List of fit result dictionaries, each with:
            - "model_name": name of model
            - "params": parameter dict
            - "model_func": function to compute flux
            - "aic" or "bic": information criterion
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        energy = data.get("energy", np.array([]))
        flux = data.get("flux", np.array([]))
        flux_err = data.get("flux_err", np.array([]))

        # Filter valid data
        valid_mask = np.isfinite(energy) & np.isfinite(flux) & np.isfinite(flux_err) & (flux > 0)
        energy_clean = energy[valid_mask]
        flux_clean = flux[valid_mask]
        flux_err_clean = flux_err[valid_mask]

        # Plot data
        ax.errorbar(energy_clean, flux_clean, yerr=flux_err_clean, fmt="o",
                   color="black", ecolor="black", alpha=0.5, markersize=6,
                   elinewidth=1.5, capsize=3, label="Data", zorder=10)

        # Color palette
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        # Plot each model
        energy_model = np.logspace(np.log10(energy_clean.min()),
                                  np.log10(energy_clean.max()), 300)

        for idx, fit in enumerate(fits):
            model_func = fit.get("model_func")
            params = fit.get("params", {})
            model_name = fit.get("model_name", f"Model {idx+1}")

            if model_func is not None:
                model_flux = model_func(energy_model, **params)
                color = colors[idx % len(colors)]

                # Add AIC/BIC info to label
                label = model_name
                if "aic" in fit:
                    label += f" (AIC: {fit['aic']:.1f})"
                elif "bic" in fit:
                    label += f" (BIC: {fit['bic']:.1f})"

                ax.plot(energy_model, model_flux, color=color, linewidth=2.5, label=label)

        # Formatting
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Energy (keV)", fontsize=self.label_size)
        ax.set_ylabel(r"$\nu F_\nu$ (erg cm$^{-2}$ s$^{-1}$)", fontsize=self.label_size)
        ax.set_title("Spectral Model Comparison", fontsize=self.title_size)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=self.config.get("legend_size", 10))

        fig.tight_layout()
        return fig

    def plot_epeak_evolution(
        self,
        times: np.ndarray,
        epeaks: np.ndarray,
        epeak_errs: np.ndarray,
        flux_levels: Optional[np.ndarray] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot time-resolved Epeak evolution.

        Parameters
        ----------
        times : np.ndarray
            Time array (seconds)
        epeaks : np.ndarray
            Peak energy values (keV)
        epeak_errs : np.ndarray
            Peak energy errors
        flux_levels : np.ndarray, optional
            Flux values for color coding
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Filter valid data
        valid_mask = np.isfinite(times) & np.isfinite(epeaks) & np.isfinite(epeak_errs)
        times_clean = times[valid_mask]
        epeaks_clean = epeaks[valid_mask]
        errs_clean = epeak_errs[valid_mask]

        if flux_levels is not None:
            flux_clean = flux_levels[valid_mask]
            scatter = ax.scatter(times_clean, epeaks_clean, c=flux_clean, s=100,
                               cmap="viridis", alpha=0.7, edgecolors="black", linewidth=0.5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(r"Flux", fontsize=self.label_size)
        else:
            ax.scatter(times_clean, epeaks_clean, s=100, color="#1f77b4", alpha=0.7,
                      edgecolors="black", linewidth=0.5)

        # Error bars
        ax.errorbar(times_clean, epeaks_clean, yerr=errs_clean, fmt="none",
                   ecolor="#1f77b4", elinewidth=1.5, capsize=3, alpha=0.5)

        # Connecting line
        sort_idx = np.argsort(times_clean)
        ax.plot(times_clean[sort_idx], epeaks_clean[sort_idx], color="gray",
               alpha=0.3, linewidth=1.5, zorder=0)

        ax.set_xscale("log")
        ax.set_xlabel(r"Time (s)", fontsize=self.label_size)
        ax.set_ylabel(r"$E_{\rm peak}$ (keV)", fontsize=self.label_size)
        ax.set_title(r"$E_{\rm peak}$ Evolution", fontsize=self.title_size)
        ax.grid(True, alpha=0.3, which="both")

        fig.tight_layout()
        return fig

    def plot_spectral_evolution(
        self,
        time_resolved_fits: List[Dict],
        **kwargs
    ) -> plt.Figure:
        """
        Plot stacked spectra at different time slices with color gradient.

        Parameters
        ----------
        time_resolved_fits : list
            List of dicts with keys:
            - "time_interval": (t_start, t_end) tuple
            - "energy": energy array
            - "flux": flux array
            - "flux_err": flux errors
            - (optional) "params": fit parameters
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        n_spectra = len(time_resolved_fits)
        colors = plt.cm.twilight(np.linspace(0, 1, n_spectra))

        for idx, fit_data in enumerate(time_resolved_fits):
            energy = fit_data.get("energy", np.array([]))
            flux = fit_data.get("flux", np.array([]))
            flux_err = fit_data.get("flux_err", np.array([]))
            time_interval = fit_data.get("time_interval", (0, 0))

            # Filter valid data
            valid_mask = np.isfinite(energy) & np.isfinite(flux) & (flux > 0)
            energy_clean = energy[valid_mask]
            flux_clean = flux[valid_mask]

            if len(energy_clean) > 0:
                label = f"T: {time_interval[0]:.2f}-{time_interval[1]:.2f} s"
                ax.plot(energy_clean, flux_clean, color=colors[idx], linewidth=2,
                       marker="o", markersize=4, alpha=0.8, label=label)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Energy (keV)", fontsize=self.label_size)
        ax.set_ylabel(r"$\nu F_\nu$ (erg cm$^{-2}$ s$^{-1}$)", fontsize=self.label_size)
        ax.set_title("Spectral Evolution", fontsize=self.title_size)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=self.config.get("legend_size", 9), ncol=2)

        fig.tight_layout()
        return fig

    def plot_sed(
        self,
        frequencies: np.ndarray,
        fluxes: np.ndarray,
        flux_errs: np.ndarray,
        instruments: Optional[List[str]] = None,
        time_label: Optional[str] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot multi-wavelength SED from radio to gamma-ray.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequency array (Hz)
        fluxes : np.ndarray
            Flux array (Jy)
        flux_errs : np.ndarray
            Flux errors
        instruments : list, optional
            Instrument names for each point
        time_label : str, optional
            Time label for the SED
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Filter valid data
        valid_mask = np.isfinite(frequencies) & np.isfinite(fluxes) & np.isfinite(flux_errs)
        freq_clean = frequencies[valid_mask]
        flux_clean = fluxes[valid_mask]
        err_clean = flux_errs[valid_mask]

        if instruments is not None:
            inst_clean = [instruments[i] for i in range(len(instruments)) if valid_mask[i]]
            # Different symbols per instrument
            unique_inst = list(set(inst_clean))
            markers = ["o", "s", "^", "v", "d", "*", "+", "x"]

            for inst in unique_inst:
                inst_mask = np.array([i == inst for i in inst_clean])
                marker = markers[unique_inst.index(inst) % len(markers)]
                ax.errorbar(freq_clean[inst_mask], flux_clean[inst_mask],
                           yerr=err_clean[inst_mask], fmt=marker, markersize=7,
                           elinewidth=1.5, capsize=3, label=inst, alpha=0.7)
        else:
            ax.errorbar(freq_clean, flux_clean, yerr=err_clean, fmt="o",
                       color="#1f77b4", ecolor="#1f77b4", markersize=7,
                       elinewidth=1.5, capsize=3, alpha=0.7, label="Data")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Frequency (Hz)", fontsize=self.label_size)
        ax.set_ylabel(r"Flux (Jy)", fontsize=self.label_size)

        title = "Spectral Energy Distribution"
        if time_label:
            title += f" ({time_label})"
        ax.set_title(title, fontsize=self.title_size)

        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=self.config.get("legend_size", 10))

        fig.tight_layout()
        return fig

    def plot_band_function_demo(
        self,
        alpha: float = -1.0,
        beta: float = -2.3,
        epeak: float = 300,
        **kwargs
    ) -> plt.Figure:
        """
        Visualize Band function with annotations.

        Parameters
        ----------
        alpha : float
            Low-energy spectral index
        beta : float
            High-energy spectral index
        epeak : float
            Peak energy (keV)
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Energy range
        energy = np.logspace(-1, 4, 1000)

        # Band function: N(E) = A * (E/100)^alpha * exp(-E/E0) for E < E_break
        #                     = A * (E_break/100)^(alpha-beta) * (E/100)^beta for E > E_break
        # E_break = (alpha - beta) * E_peak

        e_break = (alpha - beta) * epeak
        normalization = ((alpha - beta) / e_break) ** (alpha - beta) * np.exp(beta - alpha)

        n_e = np.zeros_like(energy)
        low_e = energy < e_break
        high_e = energy >= e_break

        n_e[low_e] = normalization * (energy[low_e] / 100) ** alpha * np.exp(-energy[low_e] / epeak)
        n_e[high_e] = (normalization * ((alpha - beta) / e_break) ** (alpha - beta) *
                       (energy[high_e] / 100) ** beta)

        # nuFnu = E * N(E)
        nufnu = energy * n_e

        ax.loglog(energy, nufnu, color="#1f77b4", linewidth=3, label="Band Function")

        # Mark Epeak
        epeak_val = epeak * ((2 + alpha) / (1 + alpha))
        ax.axvline(epeak, color="red", linestyle="--", linewidth=2, alpha=0.7,
                  label=f"$E_{{\\rm peak}}$ = {epeak:.0f} keV")
        ax.scatter([epeak], [epeak * normalization * (epeak / 100) ** alpha * np.exp(-1)],
                  color="red", s=100, zorder=5)

        # Mark break energy
        ax.axvline(e_break, color="green", linestyle="--", linewidth=2, alpha=0.7,
                  label=f"$E_{{\\rm break}}$ = {e_break:.0f} keV")

        # Add text annotations
        textstr = f"$\\alpha$ = {alpha:.2f}\n$\\beta$ = {beta:.2f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        ax.set_xlabel(r"Energy (keV)", fontsize=self.label_size)
        ax.set_ylabel(r"$\nu F_\nu$ (arbitrary units)", fontsize=self.label_size)
        ax.set_title("Band Function Spectral Model", fontsize=self.title_size)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=self.config.get("legend_size", 10))

        fig.tight_layout()
        return fig

    def plot_nufnu_spectrum(
        self,
        energy: np.ndarray,
        photon_flux: np.ndarray,
        photon_flux_err: Optional[np.ndarray] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot nuFnu = E^2 * N(E) representation.

        Parameters
        ----------
        energy : np.ndarray
            Energy array (keV)
        photon_flux : np.ndarray
            Photon flux array (ph cm^-2 s^-1 keV^-1)
        photon_flux_err : np.ndarray, optional
            Photon flux errors
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Filter valid data
        valid_mask = np.isfinite(energy) & np.isfinite(photon_flux) & (photon_flux > 0)
        energy_clean = energy[valid_mask]
        pf_clean = photon_flux[valid_mask]

        # Convert to nuFnu (E^2 * N(E))
        nufnu = energy_clean ** 2 * pf_clean

        if photon_flux_err is not None:
            err_clean = photon_flux_err[valid_mask]
            nufnu_err = energy_clean ** 2 * err_clean
            ax.errorbar(energy_clean, nufnu, yerr=nufnu_err, fmt="o",
                       color="#1f77b4", ecolor="#1f77b4", alpha=0.7, markersize=6,
                       elinewidth=1.5, capsize=3)
        else:
            ax.plot(energy_clean, nufnu, "o", color="#1f77b4", alpha=0.7, markersize=6)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Energy (keV)", fontsize=self.label_size)
        ax.set_ylabel(r"$\nu F_\nu$ (erg cm$^{-2}$ s$^{-1}$)", fontsize=self.label_size)
        ax.set_title(r"$\nu F_\nu$ Spectrum", fontsize=self.title_size)
        ax.grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_count_spectrum(
        self,
        channels: np.ndarray,
        counts: np.ndarray,
        model_counts: Optional[np.ndarray] = None,
        channel_width: Optional[np.ndarray] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot raw count spectrum (detector space).

        Parameters
        ----------
        channels : np.ndarray
            Channel array
        counts : np.ndarray
            Count array
        model_counts : np.ndarray, optional
            Model count array for overlay
        channel_width : np.ndarray, optional
            Channel widths for normalization
        **kwargs
            Additional customization options

        Returns
        -------
        plt.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Normalize by channel width if provided
        plot_counts = counts.copy()
        if channel_width is not None:
            plot_counts = counts / channel_width

        ax.step(channels, plot_counts, where="mid", color="#1f77b4", linewidth=2,
               label="Data")

        if model_counts is not None:
            plot_model = model_counts.copy()
            if channel_width is not None:
                plot_model = model_counts / channel_width
            ax.step(channels, plot_model, where="mid", color="#d62728", linewidth=2,
                   label="Model", alpha=0.7)

        ax.set_yscale("log")
        ax.set_xlabel("Channel", fontsize=self.label_size)
        ax.set_ylabel("Counts/s" + ("/keV" if channel_width is not None else ""),
                     fontsize=self.label_size)
        ax.set_title("Count Spectrum (Detector Space)", fontsize=self.title_size)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=self.config.get("legend_size", 10))

        fig.tight_layout()
        return fig
