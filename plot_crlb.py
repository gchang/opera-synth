import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from dolphin.phase_link.simulate import simulate_coh
from matplotlib.gridspec import GridSpec

from synth.crlb import compute_lower_bound_std

plt.style.use(["science", "no-latex"])

DEFAULT_GAMMA_INF = 0.1
DEFAULT_GAMMA0 = 0.999
DEFAULT_TAU0 = 50
DEFAULT_NUM_LOOKS = 100


def analyze_parameter_effects(
    num_acquisitions: int = 175,
    acq_interval: int = 6,
    default_num_looks: int = DEFAULT_NUM_LOOKS,
    default_gamma_inf: float = DEFAULT_GAMMA_INF,
    default_gamma0: float = DEFAULT_GAMMA0,
    default_tau0: float = DEFAULT_TAU0,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Create a comprehensive analysis of coherence parameter effects on CRLB.

    Parameters
    ----------
    num_acquisitions : int
        Number of acquisitions to simulate
    acq_interval : int
        Time between acquisitions in days
    num_looks : int
        Number of looks used in estimation when not varying
    gamma_inf : float
        Coherence value at infinite time, used when not varying
    gamma0 : float
        Coherence value at time 0, used when not varying
    tau0 : float
        Time decay constant (in days), used when not varying

    Returns
    -------
    fig : plt.Figure
        The main figure containing all subplots
    axes : list[plt.Axes]
        list of axes objects for potential further modification

    """
    days = np.arange(num_acquisitions) * acq_interval
    # Create figure with gridspec for custom layout
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig)
    axes = []

    # Plot 1: Effect of tau (decay constant)
    ax1 = fig.add_subplot(gs[0, 0])
    taus = [11, 25, 50, 100]
    for tau in taus:
        coh_matrix, _ = simulate_coh(
            num_acq=num_acquisitions,
            gamma_inf=default_gamma_inf,
            gamma0=default_gamma0,
            Tau0=tau,
            acq_interval=acq_interval,
        )
        lower_bound = compute_lower_bound_std(coh_matrix, default_num_looks)
        ax1.plot(days, lower_bound, label=r"$\tau = $" + f"{tau} days")

    ax1.set_title(r"Effect of Decay Constant ($\tau$)", fontsize=12)
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Standard Deviation (rad)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    axes.append(ax1)

    # Plot 2: Effect of gamma_inf (asymptotic coherence)
    ax2 = fig.add_subplot(gs[0, 1])
    gamma_infs = [0.05, 0.1, 0.2, 0.4]
    for gamma in gamma_infs:
        coh_matrix, _ = simulate_coh(
            num_acq=num_acquisitions,
            gamma_inf=gamma,
            gamma0=default_gamma0,
            Tau0=default_tau0,
            acq_interval=acq_interval,
        )
        lower_bound = compute_lower_bound_std(coh_matrix, default_num_looks)
        ax2.plot(days, lower_bound, label=r"$\gamma_{\infty} =$ " + str(gamma))

    ax2.set_title(r"Effect of Asymptotic Coherence ($\gamma_{\infty}$)", fontsize=12)
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Standard Deviation (rad)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    axes.append(ax2)

    # Plot 3: Effect of gamma0 (initial coherence)
    ax3 = fig.add_subplot(gs[1, 0])
    gamma0s = [0.7, 0.8, 0.9, 0.999]
    for gamma in gamma0s:
        coh_matrix, _ = simulate_coh(
            num_acq=num_acquisitions,
            gamma_inf=default_gamma_inf,
            gamma0=gamma,
            Tau0=default_tau0,
            acq_interval=acq_interval,
        )
        lower_bound = compute_lower_bound_std(coh_matrix, default_num_looks)
        ax3.plot(days, lower_bound, label=r"$\gamma_{0} =$ " + str(gamma))

    ax3.set_title(r"Effect of Initial Coherence ($\gamma_0$)", fontsize=12)
    ax3.set_xlabel("Time (days)")
    ax3.set_ylabel("Standard Deviation (rad)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    axes.append(ax3)

    # Plot 4: Effect of number of looks
    ax4 = fig.add_subplot(gs[1, 1])
    coh_matrix, _ = simulate_coh(
        num_acq=num_acquisitions,
        gamma_inf=default_gamma_inf,
        gamma0=default_gamma0,
        Tau0=default_tau0,
        acq_interval=acq_interval,
    )
    looks_values = [5, 20, 50, 100]
    for looks in looks_values:
        lower_bound = compute_lower_bound_std(coh_matrix, looks)
        ax4.plot(days, lower_bound, label=f"L = {looks}")

    ax4.set_title("Effect of Number of Looks (L)", fontsize=12)
    ax4.set_xlabel("Time (days)")
    ax4.set_ylabel("Standard Deviation (rad)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    axes.append(ax4)

    # Adjust layout and add super title
    plt.suptitle(
        "Cramer-Rao Lower Bound Analysis for Phase Linking", fontsize=14, y=1.02
    )
    plt.tight_layout()

    return fig, axes


# Example usage
if __name__ == "__main__":
    # Create the analysis plots
    fig, axes = analyze_parameter_effects()

    # Optional: Save the figure
    plt.savefig("crlb_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()
