"""
End-to-end analysis driver for QAOA postprocessing.

This module contains the main analyze_and_report function that orchestrates
the complete analysis workflow including distribution analysis, postselection,
and postprocessing.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import networkx as nx

from qiskit import QuantumCircuit

from .utils_general import (
    VariantArtifacts,
    DistributionRow,
    SummaryStats,
    ensure_dir,
    run_tag,
    analyze_variant,
    filter_rows,
    renormalize_rows,
    summarize_rows,
    circuit_metrics,
    save_tables,
    dump_analysis_json,
)
from .visualization import plot_full_stem, plot_topk_bars
from .postprocessor_mqc import run_postprocessing_suite


def analyze_and_report(
    *,
    outdir: Path,
    config: dict,
    graph: nx.Graph,
    backend_name: str,
    circuits: Dict[str, QuantumCircuit],
    artifacts: Dict[str, VariantArtifacts],
    classical_opt: float,
    sat_map: Optional[Dict[int, int]] = None,
    sat_dir: Optional[str] = None,
    postselect: Optional[dict] = None,
    postprocessor: Optional[object] = None,  # kept for backward compatibility
):
    """
    Main analysis entry point.

    Produces:
      - per-variant distribution analysis (raw)
      - post-selection analysis (high-probability subset)
      - optional postprocessing suite on selected distributions
      - tables + JSON dumps

    Args:
        outdir: Output directory for results
        config: Configuration dictionary
        graph: NetworkX graph defining the problem
        backend_name: Name of the backend used
        circuits: Dictionary of quantum circuits
        artifacts: Dictionary of measurement results
        classical_opt: Classical optimal solution value
        sat_map: Optional SAT mapping dictionary
        sat_dir: Optional SAT mapping direction
        postselect: Optional postselection policy
        postprocessor: Optional single postprocessor (deprecated, use config["POSTPROCESSORS"])
        
    Returns:
        Dictionary containing analysis results and dataframes
    """

    ensure_dir(outdir)
    plots_dir = outdir / "plots"
    tables_dir = outdir / "tables"
    data_dir = outdir / "data"
    ensure_dir(plots_dir)
    ensure_dir(tables_dir)
    ensure_dir(data_dir)

    tag = run_tag(config, graph, backend_name, active_phys=None)

    # -----------------------------------------
    # 1) Raw analysis for each available variant
    # -----------------------------------------
    summaries: List[SummaryStats] = []
    rows_store: Dict[str, List[DistributionRow]] = {}

    for var_name, art in artifacts.items():
        if var_name not in circuits:
            print(f"[WARN] artifacts has variant='{var_name}' but circuits does not. Skipping.")
            continue

        dist, rows, summary = analyze_variant(
            graph,
            art.counts,
            circuits[var_name],
            classical_opt,
            is_counts=True,
            sat_map=sat_map,
            sat_dir=sat_dir,
            variant_name=var_name,
        )

        art.probs = dist
        summaries.append(summary)
        rows_store[var_name] = rows

        # plots (before selection)
        plot_full_stem(
            rows,
            summary,
            title=f"{var_name.upper()} FULL distribution (before selection)",
            outpath=plots_dir / f"dist_full_{var_name}_{tag}.png",
            show=True,
        )
        plot_topk_bars(
            rows,
            summary,
            top_k=int(config.get("PLOT_TOPK", 40)),
            title=f"{var_name.upper()} Top-K (before selection)",
            outpath=plots_dir / f"dist_top_{var_name}_{tag}.png",
            show=True,
        )

        print(
            f"\n[{var_name}] Mode cut={summary.mode_cut:.1f} (p={summary.mode_prob:.6f}) "
            f"vs classical={classical_opt:.1f} | AR_mode={summary.mode_cut/classical_opt:.4f}"
        )
        print(
            f"[{var_name}] Best cut={summary.best_cut:.1f} | AR_best={summary.ar_best:.4f} | "
            f"E[cut]={summary.expected_cut:.2f} | AR_exp={summary.ar_expected:.4f} | "
            f"support={summary.num_support}"
        )

    # -----------------------------------------
    # 2) Post-selection (high-probability subset) per variant
    # -----------------------------------------
    postselect = postselect or {
        "method": "mass",
        "mass_keep": 0.90,
        "prob_threshold": None,
        "k_max": None
    }

    selected_summaries: List[SummaryStats] = []
    selected_rows_store: Dict[str, List[DistributionRow]] = {}

    for var_name, rows in rows_store.items():
        rows_sorted = sorted(rows, key=lambda r: r.prob, reverse=True)
        kept, mass = filter_rows(
            rows_sorted,
            method=postselect.get("method", "mass"),
            mass_keep=float(postselect.get("mass_keep", 0.90)),
            prob_threshold=postselect.get("prob_threshold", None),
            k_max=postselect.get("k_max", None),
        )
        kept_norm = renormalize_rows(kept, mass)
        selected_rows_store[var_name] = kept_norm

        summ_sel = summarize_rows(kept_norm, classical_opt, variant=f"{var_name}_selected")
        summ_sel.kept_policy = {"mass_raw": mass, **postselect}
        selected_summaries.append(summ_sel)

        # plots (after selection)
        plot_full_stem(
            kept_norm,
            summ_sel,
            title=f"{var_name.upper()} FULL (after selection) policy={postselect}",
            outpath=plots_dir / f"dist_full_{var_name}_selected_{tag}.png",
            show=True,
        )
        plot_topk_bars(
            kept_norm,
            summ_sel,
            top_k=int(config.get("PLOT_TOPK", 40)),
            title=f"{var_name.upper()} Top-K (after selection)",
            outpath=plots_dir / f"dist_top_{var_name}_selected_{tag}.png",
            show=True,
        )

    # -----------------------------------------
    # 3) Circuit metrics + summary tables
    # -----------------------------------------
    metrics_rows = []
    for k, c in circuits.items():
        m = circuit_metrics(c)
        metrics_rows.append({"variant": k, **m})
    metrics_df = pd.DataFrame(metrics_rows)

    summaries_df = pd.DataFrame(
        [s.__dict__ for s in summaries] + [s.__dict__ for s in selected_summaries]
    )

    # Save base tables + summary JSON
    save_tables(tables_dir, tag, metrics_df, summaries_df, post_df=None)
    dump_analysis_json(
        data_dir, tag,
        {"summaries": summaries_df.to_dict(orient="records")},
        "analysis_summary"
    )

    # -----------------------------------------
    # 4) Optional postprocessing suite on selected distributions
    # -----------------------------------------
    def _sanitize(s: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)[:120]

    processors = config.get("POSTPROCESSORS", None)
    if processors is None:
        processors = [postprocessor] if postprocessor is not None else []
    processors = [p for p in processors if p is not None]

    post_suite_df = pd.DataFrame()
    post_rows_store: Dict[Tuple[str, str], List[DistributionRow]] = {}

    if processors:
        all_post_tables = []

        for var_name, rows_sel in selected_rows_store.items():
            if not rows_sel:
                print(f"[WARN] No selected rows for variant '{var_name}'; skipping postprocessing for this variant.")
                continue

            suite_df, post_rows_by_method = run_postprocessing_suite(
                rows_selected=rows_sel,
                graph=graph,
                classical_opt=classical_opt,
                variant_label=var_name,
                processors=processors,
            )
            all_post_tables.append(suite_df)

            for method_name, pp_rows in post_rows_by_method.items():
                if not pp_rows:
                    continue

                pseudo = summarize_rows(
                    pp_rows,
                    classical_opt,
                    variant=f"{var_name} | {method_name}"
                )
                mtag = _sanitize(method_name)

                plot_full_stem(
                    pp_rows,
                    pseudo,
                    title=f"{var_name.upper()} POSTPROCESSED — {method_name}",
                    outpath=plots_dir / f"dist_full_{var_name}_post_{mtag}_{tag}.png",
                    show=True,
                )
                plot_topk_bars(
                    pp_rows,
                    pseudo,
                    top_k=int(config.get("PLOT_TOPK", 40)),
                    title=f"{var_name.upper()} POSTPROCESSED Top-K — {method_name}",
                    outpath=plots_dir / f"dist_top_{var_name}_post_{mtag}_{tag}.png",
                    show=True,
                )

                post_rows_store[(var_name, method_name)] = pp_rows

        if all_post_tables:
            post_suite_df = pd.concat(all_post_tables, ignore_index=True)
        else:
            post_suite_df = pd.DataFrame()

        # Save postprocessing tables/JSON only if there is something to save
        if not post_suite_df.empty:
            save_tables(tables_dir, tag, metrics_df, summaries_df, post_suite_df)
            dump_analysis_json(
                data_dir,
                tag,
                {"postprocessing_suite": post_suite_df.to_dict(orient="records")},
                "postprocessed_suite",
            )
        else:
            print("[INFO] No postprocessing rows produced; skipping post-analysis export.")

    return {
        "tag": tag,
        "metrics_df": metrics_df,
        "summaries_df": summaries_df,
        "post_suite_df": post_suite_df,
        "rows_store": rows_store,
        "selected_rows_store": selected_rows_store,
        "post_rows_store": post_rows_store,
    }

