"""
QAOA Postprocessing Utilities

A modular toolkit for analyzing and postprocessing Quantum Approximate Optimization 
Algorithm (QAOA) results, including distribution analysis, visualization, and 
error mitigation techniques.

Main components:
- utils_general: Core utilities, data structures, and analysis functions
- postprocessor_greedy: Greedy one-flip postprocessor
- postprocessor_mqc: MQC-style multi-qubit correction postprocessor
- visualization: Plotting and visualization functions
- analysis: End-to-end analysis workflow
"""

# Core data structures
from .utils_general import (
    VariantArtifacts,
    DistributionRow,
    SummaryStats,
)

# Utility functions
from .utils_general import (
    # I/O
    ensure_dir,
    save_json,
    save_circuit_png,
    run_tag,
    # Circuit metrics
    circuit_metrics,
    print_metrics,
    # Graph operations
    generate_d_regular_graph,
    cut_value,
    # Classical baselines
    maxcut_fast_heuristic,
    maxcut_exact_ilp,
    classical_baseline_bits,
    # Backend
    get_backend,
    # SAT mapping
    apply_sat_mapping,
    sat_bits_to_original_bits,
    # Circuit building
    build_bound_executable_circuits,
    # Sampler
    run_sampler_circuits,
    extract_counts_from_sampler_v2,
    # Distribution analysis
    probs_from_counts,
    rows_from_distribution,
    summarize_rows,
    analyze_variant,
    # Filtering
    filter_rows,
    renormalize_rows,
    select_reference_rows,
    # Tables and JSON
    save_tables,
    dump_raw_artifacts_json,
    dump_analysis_json,
    # Bit operations
    bits_to_int_le,
    decode_original_bits,
)

# Postprocessors
from .postprocessor_greedy import (
    DistributionPostProcessor,
    HighProbGreedy1Flip,
)

from .postprocessor_mqc import (
    MQCStyleMultiQubitCorrection,
    run_postprocessing_suite,
)

# Visualization
from .visualization import (
    draw_graph,
    plot_backend_subset,
    plot_topk_bars,
    plot_full_stem,
    plot_training_diagnostics,
)

# Main analysis driver
from .analysis import analyze_and_report

__version__ = "1.0.0"

__all__ = [
    # Data structures
    "VariantArtifacts",
    "DistributionRow",
    "SummaryStats",
    # Core utilities
    "ensure_dir",
    "save_json",
    "save_circuit_png",
    "run_tag",
    "circuit_metrics",
    "print_metrics",
    # Graph
    "generate_d_regular_graph",
    "cut_value",
    # Classical
    "maxcut_fast_heuristic",
    "maxcut_exact_ilp",
    "classical_baseline_bits",
    # Backend
    "get_backend",
    # SAT
    "apply_sat_mapping",
    "sat_bits_to_original_bits",
    # Circuits
    "build_bound_executable_circuits",
    "run_sampler_circuits",
    "extract_counts_from_sampler_v2",
    # Analysis
    "probs_from_counts",
    "rows_from_distribution",
    "summarize_rows",
    "analyze_variant",
    "filter_rows",
    "renormalize_rows",
    "select_reference_rows",
    # I/O
    "save_tables",
    "dump_raw_artifacts_json",
    "dump_analysis_json",
    # Bits
    "bits_to_int_le",
    "decode_original_bits",
    # Postprocessors
    "DistributionPostProcessor",
    "HighProbGreedy1Flip",
    "MQCStyleMultiQubitCorrection",
    "run_postprocessing_suite",
    # Visualization
    "draw_graph",
    "plot_backend_subset",
    "plot_topk_bars",
    "plot_full_stem",
    "plot_training_diagnostics",
    # Main driver
    "analyze_and_report",
]

