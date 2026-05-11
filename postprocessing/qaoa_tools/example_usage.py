"""
Example usage of the QAOA postprocessing utilities.

This script demonstrates how to use the qaoa_tools package for QAOA analysis.
It can be converted to a Jupyter notebook or run as a standalone script.
"""

import os
from pathlib import Path
import datetime

# Import the qaoa_tools package
from qaoa_tools import *

# For training (if available)
try:
    from qaoa_training_pipeline.training import DepthOneScanTrainer
    from qaoa_training_pipeline.evaluation import EfficientDepthOneEvaluator
    from qaoa_training_pipeline.utils.graph_utils import graph_to_operator
    HAS_TRAINING = True
except ImportError:
    HAS_TRAINING = False
    print("[WARN] qaoa_training_pipeline not available. Using fallback methods.")


def main():
    """Main execution function."""
    
    # ============================================================
    # 1. CONFIGURATION
    # ============================================================
    
    CONFIG = {
        # Backend mode
        "USE_REAL_BACKEND": True,
        "RUN_MODE": "hardware",          # "hardware" or "simulator"
        "BACKEND_NAME": "ibm_marrakesh",
        "SIMULATOR_IDEAL": False,
        
        # Problem
        "GRAPH_N": 25,
        "GRAPH_D": 4,
        "GRAPH_SEED": 7,
        
        # QAOA
        "REPS": 1,
        "SEED": 7,
        "SHOTS": 20000,
        "OPTIMIZATION_LEVEL": 1,
        
        # Classical baseline
        "CLASSICAL_METHOD": "fast_heuristic",
        "HEUR_RESTARTS": 50,
        "HEUR_SWEEPS": 400,
        
        # SAT mapping + subset selection
        "USE_SAT_MAPPING": True,
        "SAT_TIMEOUT": 10,
        "USE_CUSTOM_FIDELITY_EVALUATOR": True,
        
        # Plot controls
        "PLOT_TOPK": 40,
        "SAVE_DIR": "qaoa_example_runs",
    }
    
    # ============================================================
    # 2. SETUP OUTPUT DIRECTORY
    # ============================================================
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTDIR = Path(CONFIG["SAVE_DIR"]) / f"run_{timestamp}"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTDIR}")
    
    # ============================================================
    # 3. GENERATE PROBLEM GRAPH
    # ============================================================
    
    print("\n" + "="*60)
    print("STEP 1: Generate Problem Graph")
    print("="*60)
    
    graph = generate_d_regular_graph(
        CONFIG["GRAPH_N"],
        CONFIG["GRAPH_D"],
        CONFIG["GRAPH_SEED"],
        auto_fix=True
    )
    
    draw_graph(
        graph,
        title=f"{CONFIG['GRAPH_D']}-regular MaxCut graph",
        outpath=OUTDIR / "plots" / "graph.png",
        show=True
    )
    
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # ============================================================
    # 4. CLASSICAL BASELINE
    # ============================================================
    
    print("\n" + "="*60)
    print("STEP 2: Compute Classical Baseline")
    print("="*60)
    
    classical_opt, classical_bits = maxcut_exact_ilp(graph)
    print(f"Classical optimal cut: {classical_opt}")
    print(f"Classical optimal bits: {classical_bits}")
    
    # ============================================================
    # 5. QAOA TRAINING (if available)
    # ============================================================
    
    print("\n" + "="*60)
    print("STEP 3: QAOA Training")
    print("="*60)
    
    if HAS_TRAINING:
        cost_op = graph_to_operator(graph, pre_factor=-0.5)
        print(f"Cost operator qubits: {cost_op.num_qubits}")
        
        trainer = DepthOneScanTrainer(EfficientDepthOneEvaluator())
        train_res = trainer.train(cost_op, num_points=60)
        
        beta_train, gamma_train = train_res["optimized_params"]
        print(f"Trained (beta, gamma): ({beta_train:.6f}, {gamma_train:.6f})")
        print(f"Trainer energy: {train_res['energy']:.6f}")
        
        # Plot training diagnostics
        summary = plot_training_diagnostics(train_res)
        print(f"Inferred mode: {summary['inferred_mode']}")
    else:
        # Fallback: use default parameters
        print("[INFO] Using default QAOA parameters")
        from qaoa_training_pipeline.utils.graph_utils import graph_to_operator
        cost_op = graph_to_operator(graph, pre_factor=-0.5)
        beta_train, gamma_train = 0.5, 0.5
        train_res = {
            "optimized_qaoa_angles": [beta_train, gamma_train],
            "optimized_params": [beta_train, gamma_train],
        }
    
    # ============================================================
    # 6. BACKEND SETUP
    # ============================================================
    
    print("\n" + "="*60)
    print("STEP 4: Backend Setup")
    print("="*60)
    
    real_backend, exec_backend, backend_name = get_backend(CONFIG)
    print(f"Backend: {backend_name}")
    
    # ============================================================
    # 7. SAT MAPPING
    # ============================================================
    
    print("\n" + "="*60)
    print("STEP 5: SAT Mapping")
    print("="*60)
    
    sat_cost_op, sat_map, sat_dir, min_sat_layers = apply_sat_mapping(cost_op, CONFIG)
    
    # ============================================================
    # 8. QUBIT SELECTION (if qopt available)
    # ============================================================
    
    print("\n" + "="*60)
    print("STEP 6: Qubit Selection")
    print("="*60)
    
    try:
        from qopt_best_practices.qubit_selection import BackendEvaluator
        path_finder = BackendEvaluator(real_backend)
        path, fidelity, num_subsets = path_finder.evaluate(sat_cost_op.num_qubits)
        print(f"Best path: {path}")
        print(f"Best path fidelity: {fidelity:.6f}")
        print(f"Num. evaluated paths: {num_subsets}")
        
        plot_backend_subset(
            real_backend,
            path,
            outpath=OUTDIR / "plots" / "backend_subset.png",
            show=True
        )
    except ImportError:
        print("[WARN] qopt_best_practices not available, using linear path")
        path = list(range(sat_cost_op.num_qubits))
    
    # ============================================================
    # 9. BUILD AND TRANSPILE CIRCUITS
    # ============================================================
    
    print("\n" + "="*60)
    print("STEP 7: Build Circuits")
    print("="*60)
    
    circuits = build_bound_executable_circuits(
        sat_cost_op,
        train_res["optimized_qaoa_angles"],
        real_backend,
        CONFIG,
        active_path=path,
    )
    
    for name, circ in circuits.items():
        print_metrics(name, circ)
        save_circuit_png(circ, name, OUTDIR / "plots")
    
    # ============================================================
    # 10. EXECUTE CIRCUITS
    # ============================================================
    
    print("\n" + "="*60)
    print("STEP 8: Execute Circuits")
    print("="*60)
    
    artifacts, job_id = run_sampler_circuits(
        exec_backend,
        circuits,
        CONFIG,
    )
    
    print(f"Job ID: {job_id}")
    for k, a in artifacts.items():
        print(f"{k} shots: {sum(a.counts.values())}")
    
    # ============================================================
    # 11. CONFIGURE POSTPROCESSING
    # ============================================================
    
    print("\n" + "="*60)
    print("STEP 9: Configure Postprocessing")
    print("="*60)
    
    CONFIG["REFERENCE_SELECTION"] = "hybrid"  # "prob", "objective", or "hybrid"
    
    CONFIG["POSTPROCESSORS"] = [
        HighProbGreedy1Flip(),
        MQCStyleMultiQubitCorrection(
            top_m=500,
            max_passes=3,
            reference_mode=CONFIG["REFERENCE_SELECTION"],
        ),
    ]
    
    postselect_policy = {
        "method": "mass",
        "mass_keep": 0.90,
        "prob_threshold": None,
        "k_max": 2000
    }
    
    print("Postprocessors configured:")
    for proc in CONFIG["POSTPROCESSORS"]:
        print(f"  - {proc.display_name}")
    
    # ============================================================
    # 12. RUN COMPREHENSIVE ANALYSIS
    # ============================================================
    
    print("\n" + "="*60)
    print("STEP 10: Run Analysis")
    print("="*60)
    
    report = analyze_and_report(
        outdir=OUTDIR,
        config=CONFIG,
        graph=graph,
        backend_name=backend_name,
        circuits=circuits,
        artifacts=artifacts,
        classical_opt=classical_opt,
        sat_map=sat_map,
        sat_dir=sat_dir,
        postselect=postselect_policy,
        postprocessor=None,
    )
    
    # ============================================================
    # 13. DISPLAY RESULTS
    # ============================================================
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print("\nCircuit Metrics:")
    print(report["metrics_df"].to_string())
    
    print("\n\nDistribution Summaries:")
    print(report["summaries_df"].to_string())
    
    if not report["post_suite_df"].empty:
        print("\n\nPostprocessing Results:")
        print(report["post_suite_df"].to_string())
    else:
        print("\n[INFO] No postprocessing results available")
    
    print(f"\n\nAll results saved to: {OUTDIR}")
    print("="*60)


if __name__ == "__main__":
    # Check for required environment variables
    if "QISKIT_IBM_TOKEN" not in os.environ:
        print("[WARN] QISKIT_IBM_TOKEN not set. Hardware execution will fail.")
        print("Set it with: export QISKIT_IBM_TOKEN='your_token'")
    
    if "QISKIT_IBM_INSTANCE" not in os.environ:
        print("[WARN] QISKIT_IBM_INSTANCE not set. Hardware execution will fail.")
        print("Set it with: export QISKIT_IBM_INSTANCE='your_instance'")
    
    # Run main function
    main()

