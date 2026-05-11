"""
Visualization utilities for QAOA postprocessing analysis.

This module contains plotting functions for distributions, training diagnostics,
and backend topology visualization.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from qiskit.visualization import plot_gate_map

from .utils_general import DistributionRow, SummaryStats, ensure_dir


# ----------------------------
# Graph visualization
# ----------------------------

def draw_graph(g: nx.Graph, title="", outpath=None, show=True):
    """
    Draw a NetworkX graph with spring layout.
    
    Args:
        g: NetworkX graph to visualize
        title: Plot title
        outpath: Optional path to save the figure
        show: Whether to display the plot
    """
    pos = nx.spring_layout(g, seed=42)
    plt.figure(figsize=(5, 4))
    nx.draw(g, pos, with_labels=True)
    plt.title(title)
    if outpath:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()


def plot_backend_subset(backend, subset: List[int], show: bool = True, outpath: Optional[Path] = None):
    """
    Visualize a backend's qubit topology with a subset highlighted.
    
    Args:
        backend: Qiskit backend object
        subset: List of qubit indices to highlight
        show: Whether to display the plot
        outpath: Optional path to save the figure
    """
    qubit_color = []
    for i in range(backend.num_qubits):
        qubit_color.append("#8c00ff" if i in subset else "#6699ff")
    fig = plot_gate_map(backend, qubit_color=qubit_color)
    if outpath:
        ensure_dir(outpath.parent)
        fig.savefig(outpath, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ----------------------------
# Distribution plotting
# ----------------------------

def plot_topk_bars(
    rows: List[DistributionRow],
    summary: SummaryStats,
    *,
    top_k: int = 40,
    title: str = "",
    outpath: Optional[Path] = None,
    show: bool = True,
):
    """
    Plot top-K bitstrings as a bar chart colored by cut value.
    
    Args:
        rows: Distribution rows to plot
        summary: Summary statistics for the distribution
        top_k: Number of top bitstrings to display
        title: Plot title
        outpath: Optional path to save the figure
        show: Whether to display the plot
    """
    rows_sorted = sorted(rows, key=lambda r: r.prob, reverse=True)[:min(top_k, len(rows))]
    labels = [r.bitstring for r in rows_sorted]
    probs = [r.prob for r in rows_sorted]
    cuts  = [r.cut for r in rows_sorted]

    fig = plt.figure(figsize=(max(14, len(labels) * 0.45), 6))
    ax = plt.gca()
    bars = ax.bar(range(len(labels)), probs)

    # color by cut quality
    cmin, cmax = min(cuts), max(cuts)
    for b, c in zip(bars, cuts):
        if cmax > cmin:
            t = (c - cmin) / (cmax - cmin)
            b.set_color((0.2, 0.25 + 0.6 * t, 0.9 - 0.6 * t))
        else:
            b.set_color((0.2, 0.5, 0.9))

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Probability")
    ax.set_title(
        f"{title}\n"
        f"Mode cut={summary.mode_cut:.1f} (p={summary.mode_prob:.4f}) | "
        f"Best cut={summary.best_cut:.1f} | "
        f"E[cut]={summary.expected_cut:.2f} | "
        f"AR_exp={summary.ar_expected:.4f}, AR_best={summary.ar_best:.4f}"
    )
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()

    if outpath:
        ensure_dir(outpath.parent)
        plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_full_stem(
    rows: List[DistributionRow],
    summary: SummaryStats,
    *,
    title: str = "",
    outpath: Optional[Path] = None,
    show: bool = True,
):
    """
    Plot full distribution as a stem plot.
    
    Args:
        rows: Distribution rows to plot
        summary: Summary statistics for the distribution
        title: Plot title
        outpath: Optional path to save the figure
        show: Whether to display the plot
    """
    rows_sorted = sorted(rows, key=lambda r: r.loc_int)
    xs = np.array([r.loc_int for r in rows_sorted], dtype=float)
    ys = np.array([r.prob for r in rows_sorted], dtype=float)

    fig, ax = plt.subplots(figsize=(14, 5))
    markerline, stemlines, baseline = ax.stem(xs, ys)
    plt.setp(markerline, markersize=3)
    plt.setp(stemlines, linewidth=0.7, alpha=0.8)
    plt.setp(baseline, linewidth=0.5, alpha=0.3)

    ax.set_xlabel("Integer location (little-endian of decoded logical bits)")
    ax.set_ylabel("Probability")
    ax.set_title(f"{title}\nSupport={summary.num_support} | AR_exp={summary.ar_expected:.4f} | AR_best={summary.ar_best:.4f}")
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    if outpath:
        ensure_dir(outpath.parent)
        plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ----------------------------
# Training diagnostics
# ----------------------------

def _extract_beta_gamma(param_history):
    """
    Robustly parse parameter_history entries into beta, gamma arrays.

    Supports entries like:
      - (beta, gamma)
      - [beta, gamma]
      - {"beta": ..., "gamma": ...}
      - {"params": [beta, gamma]}
    """
    betas = []
    gammas = []

    for p in param_history:
        beta = np.nan
        gamma = np.nan

        if isinstance(p, dict):
            if "beta" in p and "gamma" in p:
                beta = float(p["beta"])
                gamma = float(p["gamma"])
            elif "params" in p and len(p["params"]) >= 2:
                beta = float(p["params"][0])
                gamma = float(p["params"][1])
            else:
                # fallback: take first two numeric values if possible
                vals = list(p.values())
                if len(vals) >= 2:
                    beta = float(vals[0])
                    gamma = float(vals[1])

        elif isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2:
            beta = float(p[0])
            gamma = float(p[1])

        betas.append(beta)
        gammas.append(gamma)

    return np.asarray(betas, dtype=float), np.asarray(gammas, dtype=float)


def plot_training_diagnostics(
    train_res,
    *,
    optimized_param_order=("beta", "gamma"),
    figsize=(14, 10),
    tol=1e-10,
):
    """
    Plot training diagnostics from a train_res dictionary.

    Expected keys:
      - energy_history
      - parameter_history
      - optimized_params
      - energy

    Args:
        train_res: Training result dictionary
        optimized_param_order: Order of parameters in optimized_params
        figsize: Figure size tuple
        tol: Tolerance for matching energies
        
    Returns:
        Dictionary with diagnostic summary
    """
    energy_history = np.asarray(train_res["energy_history"], dtype=float)
    parameter_history = train_res["parameter_history"]
    optimized_params = train_res["optimized_params"]
    reported_energy = float(train_res["energy"])

    n = len(energy_history)
    steps = np.arange(n)

    # Parse parameter history
    betas, gammas = _extract_beta_gamma(parameter_history)

    if len(betas) != n or len(gammas) != n:
        raise ValueError(
            f"Length mismatch: len(energy_history)={n}, "
            f"len(beta_history)={len(betas)}, len(gamma_history)={len(gammas)}"
        )

    # Optimized params
    beta_opt = float(optimized_params[0])
    gamma_opt = float(optimized_params[1])

    # Min/max energy points from actual history
    idx_min = int(np.argmin(energy_history))
    idx_max = int(np.argmax(energy_history))

    e_min = float(energy_history[idx_min])
    e_max = float(energy_history[idx_max])

    beta_min = float(betas[idx_min])
    gamma_min = float(gammas[idx_min])

    beta_max = float(betas[idx_max])
    gamma_max = float(gammas[idx_max])

    # Determine whether reported optimum corresponds to min or max
    dmin = abs(reported_energy - e_min)
    dmax = abs(reported_energy - e_max)

    if dmin <= tol and dmin <= dmax:
        inferred_mode = "minimizing"
        idx_opt = idx_min
    elif dmax <= tol and dmax < dmin:
        inferred_mode = "maximizing"
        idx_opt = idx_max
    else:
        # If no exact match, pick whichever is closer
        if dmin <= dmax:
            inferred_mode = "likely minimizing (closest to min)"
            idx_opt = idx_min
        else:
            inferred_mode = "likely maximizing (closest to max)"
            idx_opt = idx_max

    # Also find nearest step to optimized params
    param_dist = (betas - beta_opt) ** 2 + (gammas - gamma_opt) ** 2
    idx_param_nearest = int(np.nanargmin(param_dist))
    e_at_opt_params = float(energy_history[idx_param_nearest])

    # Cumulative best traces
    cum_min = np.minimum.accumulate(energy_history)
    cum_max = np.maximum.accumulate(energy_history)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # (1) Energy vs step
    ax = axes[0, 0]
    ax.plot(steps, energy_history, marker="o", ms=3, lw=1.2, label="Energy history")
    ax.scatter(idx_min, e_min, color="tab:blue", s=80, marker="v", label=f"Min @ step {idx_min}")
    ax.scatter(idx_max, e_max, color="tab:red", s=80, marker="^", label=f"Max @ step {idx_max}")
    ax.axhline(reported_energy, color="black", linestyle="--", alpha=0.8, label="Reported optimum energy")
    ax.scatter(idx_param_nearest, e_at_opt_params, color="gold", edgecolor="black", s=90, zorder=5,
               label="Nearest step to optimized_params")
    ax.set_xlabel("Step / Evaluation index")
    ax.set_ylabel("Energy")
    ax.set_title(f"Energy vs Step\nInferred trainer behavior: {inferred_mode}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (2) Cumulative min/max
    ax = axes[0, 1]
    ax.plot(steps, cum_min, label="Cumulative minimum", color="tab:blue")
    ax.plot(steps, cum_max, label="Cumulative maximum", color="tab:red")
    ax.plot(steps, energy_history, alpha=0.25, color="gray", label="Raw energy")
    ax.set_xlabel("Step / Evaluation index")
    ax.set_ylabel("Energy")
    ax.set_title("Running best-so-far energy")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (3) Beta / Gamma vs step
    ax = axes[1, 0]
    ax.plot(steps, betas, marker="o", ms=3, lw=1.0, label="beta")
    ax.plot(steps, gammas, marker="o", ms=3, lw=1.0, label="gamma")
    ax.scatter(idx_min, beta_min, color="tab:blue", s=60, marker="v")
    ax.scatter(idx_min, gamma_min, color="tab:blue", s=60, marker="v")
    ax.scatter(idx_max, beta_max, color="tab:red", s=60, marker="^")
    ax.scatter(idx_max, gamma_max, color="tab:red", s=60, marker="^")
    ax.scatter(idx_param_nearest, beta_opt, color="gold", edgecolor="black", s=80, marker="o")
    ax.scatter(idx_param_nearest, gamma_opt, color="gold", edgecolor="black", s=80, marker="o")
    ax.set_xlabel("Step / Evaluation index")
    ax.set_ylabel("Parameter value")
    ax.set_title("Parameter trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (4) Beta-Gamma plane colored by energy
    ax = axes[1, 1]
    sc = ax.scatter(gammas, betas, c=energy_history, cmap="viridis", s=40)
    ax.scatter(gamma_min, beta_min, color="tab:blue", s=120, marker="v", label="Min energy point")
    ax.scatter(gamma_max, beta_max, color="tab:red", s=120, marker="^", label="Max energy point")
    ax.scatter(gamma_opt, beta_opt, color="gold", edgecolor="black", s=140, marker="*", label="optimized_params")
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Beta")
    ax.set_title("Parameter points colored by energy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Energy")

    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Printed summary
    # -----------------------------
    print("========== Training diagnostic summary ==========")
    print(f"Number of evaluations: {n}")
    print(f"Reported optimized_params ({optimized_param_order[0]}, {optimized_param_order[1]}): "
          f"({beta_opt:.12g}, {gamma_opt:.12g})")
    print(f"Reported trainer energy: {reported_energy:.12g}")
    print()
    print(f"Minimum energy in history: {e_min:.12g} at step {idx_min}")
    print(f"  params at minimum: beta={beta_min:.12g}, gamma={gamma_min:.12g}")
    print()
    print(f"Maximum energy in history: {e_max:.12g} at step {idx_max}")
    print(f"  params at maximum: beta={beta_max:.12g}, gamma={gamma_max:.12g}")
    print()
    print(f"Nearest sampled step to optimized_params: step {idx_param_nearest}")
    print(f"  sampled params there: beta={betas[idx_param_nearest]:.12g}, gamma={gammas[idx_param_nearest]:.12g}")
    print(f"  energy there: {e_at_opt_params:.12g}")
    print()
    print(f"Inferred trainer behavior: {inferred_mode}")
    print("===============================================")

    return {
        "inferred_mode": inferred_mode,
        "idx_min": idx_min,
        "idx_max": idx_max,
        "min_energy": e_min,
        "max_energy": e_max,
        "beta_min": beta_min,
        "gamma_min": gamma_min,
        "beta_max": beta_max,
        "gamma_max": gamma_max,
        "beta_opt": beta_opt,
        "gamma_opt": gamma_opt,
        "reported_energy": reported_energy,
        "idx_param_nearest": idx_param_nearest,
        "energy_at_nearest_opt_params": e_at_opt_params,
    }
