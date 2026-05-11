"""
General utilities for QAOA postprocessing.

This module contains core data structures, graph operations, backend management,
SAT mapping, circuit building, and distribution analysis functions.
"""

from __future__ import annotations

import os
import json
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
import networkx as nx

from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer

from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2

# Optional Aer
try:
    from qiskit_aer import AerSimulator
    HAS_AER = True
except Exception:
    HAS_AER = False

# Optional Gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except Exception:
    HAS_GUROBI = False

# qopt-best-practices
try:
    from qopt_best_practices.qubit_selection import BackendEvaluator
    from qopt_best_practices.sat_mapping import SATMapper
    from qopt_best_practices.circuit_library import annotated_qaoa_ansatz
    from qopt_best_practices.transpilation import generate_preset_qaoa_pass_manager
    HAS_QOPT = True
except Exception:
    HAS_QOPT = False

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import CouplingMap


# ============================================================
# Data structures
# ============================================================

@dataclass
class VariantArtifacts:
    name: str
    counts: Dict[str, int]
    probs: Optional[Dict[str, float]] = None
    circuit_name: Optional[str] = None
    register_name: Optional[str] = None
    job_id: Optional[str] = None


@dataclass
class DistributionRow:
    bitstring: str
    prob: float
    bits: List[int]
    loc_int: int
    cut: float


@dataclass
class SummaryStats:
    variant: str
    n: int
    classical_opt: float
    mode_bitstring: str
    mode_prob: float
    mode_cut: float
    best_bitstring: str
    best_prob: float
    best_cut: float
    expected_cut: float
    ar_best: float
    ar_expected: float
    num_support: int
    kept_policy: Optional[dict] = None


# ============================================================
# Helper functions
# ============================================================

def select_reference_rows(rows, *, top_m, mode="hybrid"):
    """
    Select reference rows for MQC-style postprocessing.

    mode:
      - "prob":      rank by probability only
      - "objective": rank by cut value only
      - "hybrid":    rank by (cut, probability)
    """
    if mode == "prob":
        key = lambda r: r.prob
    elif mode == "objective":
        key = lambda r: r.cut
    elif mode == "hybrid":
        key = lambda r: (r.cut, r.prob)
    else:
        raise ValueError(f"Unknown reference selection mode: {mode}")

    return sorted(rows, key=key, reverse=True)[:min(top_m, len(rows))]


# ----------------------------
# I/O helpers
# ----------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_circuit_png(circuit: QuantumCircuit, label: str, outdir: Path):
    ensure_dir(outdir)
    try:
        circuit_drawer(circuit, output="mpl", filename=outdir / f"circuit_{label}.png")
    except Exception as e:
        print(f"[WARN] Could not save circuit diagram '{label}': {e}")


def run_tag(config: dict, graph: nx.Graph, backend_name: str, active_phys: Optional[List[int]] = None) -> str:
    mode = config.get("RUN_MODE", "hardware")
    n_problem = graph.number_of_nodes()
    n_active = len(active_phys) if active_phys else n_problem
    shots = config.get("SHOTS", None)
    return f"{backend_name}_{mode}_n{n_problem}_act{n_active}_shots{shots}"


# ----------------------------
# Circuit metrics
# ----------------------------

def circuit_metrics(circuit: QuantumCircuit) -> Dict[str, Any]:
    ops = dict(circuit.count_ops())
    one_q_set = {"x", "sx", "rz", "rx", "ry", "y", "id"}
    two_q_set = {"cx", "cz", "ecr", "rzz", "swap"}
    return {
        "depth": circuit.depth(),
        "width": circuit.num_qubits,
        "ops": ops,
        "n_1q": int(sum(v for k, v in ops.items() if k in one_q_set)),
        "n_2q": int(sum(v for k, v in ops.items() if k in two_q_set)),
        "n_swap": int(ops.get("swap", 0)),
        "n_delay": int(ops.get("delay", 0)),
        "n_meas": ops.get("measure", 0),
        "n_barrier": ops.get("barrier", 0),
    }


def print_metrics(label: str, circuit: QuantumCircuit) -> Dict[str, Any]:
    m = circuit_metrics(circuit)
    print(f"[{label}] depth={m['depth']} n_2q={m['n_2q']} swap={m['n_swap']} n_1q={m['n_1q']} delay={m['n_delay']}")
    return m


# ============================================================
# Graph + MaxCut
# ============================================================

def generate_d_regular_graph(n: int, d: int, seed: int, auto_fix: bool = False) -> nx.Graph:
    if (n * d) % 2 != 0:
        if auto_fix:
            n += 1
        else:
            raise ValueError("n*d must be even")
    g = nx.random_regular_graph(d, n, seed=seed)
    for u, v in g.edges():
        g[u][v]["weight"] = 1.0
    return g


def cut_value(bits: Sequence[int], g: nx.Graph) -> float:
    return sum(
        g[u][v]["weight"]
        for u, v in g.edges()
        if bits[u] != bits[v]
    )


# ============================================================
# Classical baselines
# ============================================================

def maxcut_fast_heuristic(g: nx.Graph, seed=0, restarts=50, sweeps=400):
    rng = np.random.default_rng(seed)
    n = g.number_of_nodes()
    best_val, best_bits = -1, None

    adj = [[(j, g[i][j]["weight"]) for j in g.neighbors(i)] for i in range(n)]

    for _ in range(restarts):
        bits = rng.integers(0, 2, size=n).tolist()
        gain = np.zeros(n)

        for i in range(n):
            gain[i] = sum(w if bits[i] == bits[j] else -w for j, w in adj[i])

        for _ in range(sweeps):
            i = int(np.argmax(gain))
            if gain[i] <= 0:
                break
            bits[i] ^= 1
            gain[i] = -gain[i]
            for j, w in adj[i]:
                gain[j] += 2 * w if bits[i] == bits[j] else -2 * w

        val = cut_value(bits, g)
        if val > best_val:
            best_val, best_bits = val, bits.copy()

    return float(best_val), best_bits


def maxcut_exact_ilp(g: nx.Graph):
    if not HAS_GUROBI:
        raise RuntimeError("Gurobi not available")

    n = g.number_of_nodes()
    edges = list(g.edges())

    m = gp.Model("maxcut")
    m.Params.OutputFlag = 0

    x = m.addVars(n, vtype=GRB.BINARY)
    y = m.addVars(len(edges), vtype=GRB.BINARY)

    for k, (u, v) in enumerate(edges):
        m.addConstr(y[k] >= x[u] - x[v])
        m.addConstr(y[k] >= x[v] - x[u])
        m.addConstr(y[k] <= x[u] + x[v])
        m.addConstr(y[k] <= 2 - x[u] - x[v])

    m.setObjective(
        gp.quicksum(g[u][v]["weight"] * y[k] for k, (u, v) in enumerate(edges)),
        GRB.MAXIMIZE,
    )
    m.optimize()

    bits = [int(x[i].X) for i in range(n)]
    return float(m.ObjVal), bits


def classical_baseline_bits(graph: nx.Graph, config: dict):
    if HAS_GUROBI and config.get("CLASSICAL_METHOD") == "exact_ilp":
        return maxcut_exact_ilp(graph)
    return maxcut_fast_heuristic(
        graph,
        seed=config.get("SEED", 0),
        restarts=config.get("HEUR_RESTARTS", 50),
        sweeps=config.get("HEUR_SWEEPS", 400),
    )


# ----------------------------
# Backend selection
# ----------------------------

def get_backend(config: dict):
    """
    Returns (real_backend, exec_backend, backend_name_str)
    - In hardware mode: exec_backend = real_backend
    - In simulator mode: exec_backend = AerSimulator(.from_backend(real_backend)) or ideal AerSimulator()
    """
    use_real = config.get("USE_REAL_BACKEND", True)
    run_mode = config.get("RUN_MODE", "hardware")

    if use_real:
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=os.environ["QISKIT_IBM_TOKEN"],
            instance=os.environ["QISKIT_IBM_INSTANCE"],
        )
        real_backend = service.backend(config.get("BACKEND_NAME", "ibm_marrakish"))
        backend_name = real_backend.name
    else:
        # Local debug backend
        from qiskit.providers.fake_provider import GenericBackendV2
        cmap = CouplingMap.from_heavy_hex(distance=3)
        real_backend = GenericBackendV2(
            num_qubits=19,
            coupling_map=cmap,
            basis_gates=["x", "sx", "cz", "id", "rz"],
            seed=0,
        )
        backend_name = "GenericBackendV2"

    if run_mode == "hardware":
        exec_backend = real_backend
    else:
        if not HAS_AER:
            raise RuntimeError("RUN_MODE=simulator but qiskit-aer is not installed.")
        if config.get("SIMULATOR_IDEAL", False):
            exec_backend = AerSimulator()
        else:
            exec_backend = AerSimulator.from_backend(real_backend)

    return real_backend, exec_backend, backend_name


# ----------------------------
# SAT mapping + direction inference
# ----------------------------

def pauli_expectation_on_bits(op: SparsePauliOp, bits: List[int]) -> float:
    bits = list(bits)
    value = 0.0
    labels = op.paulis.to_labels()
    for label, coeff in zip(labels, op.coeffs):
        term = 1.0
        for i, p in enumerate(reversed(label)):
            if p == "I":
                continue
            if p == "Z":
                term *= (1.0 if bits[i] == 0 else -1.0)
            else:
                term = 0.0
                break
        value += float(np.real(coeff)) * term
    return float(value)


def infer_sat_map_direction(
    original_op: SparsePauliOp,
    mapped_op: SparsePauliOp,
    mapping: Dict[int, int],
    n_qubits: int,
    trials: int = 30,
    seed: int = 123,
) -> Tuple[str, float, float]:
    rng = np.random.default_rng(seed)
    score_A = 0.0
    score_B = 0.0

    for _ in range(trials):
        old_bits = rng.integers(0, 2, size=n_qubits).tolist()
        e_old = pauli_expectation_on_bits(original_op, old_bits)

        # A) mapping[old] = new
        new_bits_A = [0] * n_qubits
        for old_idx, new_idx in mapping.items():
            new_bits_A[new_idx] = old_bits[old_idx]
        e_A = pauli_expectation_on_bits(mapped_op, new_bits_A)
        score_A += abs(e_old - e_A)

        # B) mapping[new] = old
        new_bits_B = [0] * n_qubits
        for new_idx, old_idx in mapping.items():
            new_bits_B[new_idx] = old_bits[old_idx]
        e_B = pauli_expectation_on_bits(mapped_op, new_bits_B)
        score_B += abs(e_old - e_B)

    if score_A <= score_B:
        return "old_to_new", score_A, score_B
    return "new_to_old", score_A, score_B


def sat_bits_to_original_bits(bits_sat: List[int], sat_map: Dict[int, int], direction: str) -> List[int]:
    n = len(bits_sat)
    old_bits = [0] * n
    if direction == "old_to_new":
        for old_idx, new_idx in sat_map.items():
            old_bits[old_idx] = bits_sat[new_idx]
    elif direction == "new_to_old":
        for new_idx, old_idx in sat_map.items():
            old_bits[old_idx] = bits_sat[new_idx]
    else:
        raise ValueError(f"Unknown direction: {direction}")
    return old_bits


def apply_sat_mapping(cost_op: SparsePauliOp, config: dict) -> Tuple[SparsePauliOp, Optional[Dict[int, int]], Optional[str], Optional[int]]:
    """
    Returns: (sat_cost_op, sat_map, sat_dir, min_layers)
    """
    if not config.get("USE_SAT_MAPPING", True):
        return cost_op, None, None, None
    if not HAS_QOPT:
        raise RuntimeError("SAT mapping requires qopt_best_practices.SATMapper.")
    timeout = config.get("SAT_TIMEOUT", 10)
    from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy
    swap_strat = SwapStrategy.from_line(range(cost_op.num_qubits))
    sat_cost_op, sat_map, min_layers = SATMapper(timeout=timeout).remap_graph_with_sat(cost_op, swap_strat)
    # infer direction using energy preservation test
    sat_dir, scoreA, scoreB = infer_sat_map_direction(cost_op, sat_cost_op, sat_map, cost_op.num_qubits)
    print(f"[SAT] direction={sat_dir} scoreA={scoreA:.4g} scoreB={scoreB:.4g} min_layers={min_layers}")
    return sat_cost_op, sat_map, sat_dir, min_layers


# ----------------------------
# Circuit creation + transpilation
# ----------------------------

from qiskit.circuit.library import QAOAAnsatz


def build_bound_executable_circuits(
    cost_op: SparsePauliOp,
    qaoa_angles: Sequence[float],
    backend,
    config: dict,
    active_path: List[int],
) -> Dict[str, QuantumCircuit]:
    """
    Build a bound QAOA circuit using optimized_qaoa_angles, but preserve
    the qopt_best_practices compilation path when available.
    """
    reps = int(config.get("REPS", 1))

    if HAS_QOPT:
        circ = annotated_qaoa_ansatz(cost_op, reps=reps)
    else:
        circ = QAOAAnsatz(cost_op, reps=reps, flatten=True)

    circ.remove_final_measurements()
    circ.assign_parameters(qaoa_angles, inplace=True)
    circ.measure_all()

    if HAS_QOPT:
        from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy
        from qiskit.transpiler import Layout

        swap_strat = SwapStrategy.from_line(range(cost_op.num_qubits))
        edge_coloring = {(i, i + 1): i % 2 for i in range(cost_op.num_qubits - 1)}
        initial_layout = Layout.from_intlist(active_path, circ.qregs[0])

        pm = generate_preset_qaoa_pass_manager(
            backend,
            swap_strat,
            initial_layout=initial_layout,
            edge_coloring=edge_coloring,
        )
        baseline = pm.run(circ)
    else:
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=int(config.get("OPTIMIZATION_LEVEL", 1)),
            seed_transpiler=int(config.get("SEED", 0)),
            initial_layout=active_path,
        )
        baseline = pm.run(circ)

    return {"baseline": baseline}


# ----------------------------
# Sampler execution & extraction
# ----------------------------

def extract_counts_from_sampler_v2(pub_result) -> Tuple[Dict[str, int], str]:
    data = pub_result.data
    for attr in dir(data):
        if attr.startswith("_"):
            continue
        reg = getattr(data, attr)
        if hasattr(reg, "get_counts"):
            return reg.get_counts(), attr
    raise RuntimeError("No register with get_counts() found in SamplerV2 result.")


def run_sampler_circuits(
    exec_backend,
    circuits: Dict[str, QuantumCircuit],
    config: dict,
) -> Tuple[Dict[str, VariantArtifacts], Optional[str]]:
    """
    Run already-bound circuits with SamplerV2.
    """
    shots = int(config.get("SHOTS", 20000))
    run_mode = config.get("RUN_MODE", "hardware")

    pubs = []
    order = []

    for name, circ in circuits.items():
        pubs.append((circ,))
        order.append(name)

    job_id = None

    if run_mode == "hardware":
        with Session(backend=exec_backend) as session:
            sampler = SamplerV2(mode=session)
            job = sampler.run(pubs, shots=shots)
            job_id = job.job_id()
            prim_result = job.result()
    else:
        sampler = SamplerV2(mode=exec_backend)
        job = sampler.run(pubs, shots=shots)
        prim_result = job.result()

    artifacts: Dict[str, VariantArtifacts] = {}
    for idx, name in enumerate(order):
        counts, reg = extract_counts_from_sampler_v2(prim_result[idx])
        artifacts[name] = VariantArtifacts(
            name=name,
            counts=counts,
            probs=None,
            circuit_name=name,
            register_name=reg,
            job_id=job_id,
        )

    return artifacts, job_id


# ----------------------------
# Layout + measurement decoding
# ----------------------------

def bit_at(bitstring: str, i: int) -> int:
    # little-endian bit index: bit 0 is rightmost in classical string
    return int(bitstring[::-1][i])


def measured_phys_qubits_from_measures(circuit: QuantumCircuit) -> List[int]:
    clbit_to_phys = {}
    for inst, qargs, cargs in circuit.data:
        if inst.name == "measure":
            phys_index = circuit.qubits.index(qargs[0])
            cl_index = circuit.clbits.index(cargs[0])
            clbit_to_phys[cl_index] = phys_index
    if not clbit_to_phys:
        raise RuntimeError("No measure instructions found.")
    max_c = max(clbit_to_phys)
    missing = [i for i in range(max_c + 1) if i not in clbit_to_phys]
    if missing:
        raise RuntimeError(f"Missing measurement mapping for classical bits: {missing}")
    return [clbit_to_phys[i] for i in range(max_c + 1)]


def get_virt_to_phys_list(circuit: QuantumCircuit, n_virtual: int) -> List[int]:
    layout = getattr(circuit, "layout", None)
    if layout is None or not hasattr(layout, "final_index_layout"):
        return list(range(n_virtual))
    try:
        v2p = layout.final_index_layout(filter_ancillas=True)
    except TypeError:
        v2p = layout.final_index_layout()
    return [int(v2p[v]) for v in range(n_virtual)]


def virt_bits_from_bitstring(bitstring: str, circuit: QuantumCircuit, n_virtual: int) -> List[int]:
    measured_phys = measured_phys_qubits_from_measures(circuit)
    phys_to_cl = {phys: cl for cl, phys in enumerate(measured_phys)}
    virt_to_phys = get_virt_to_phys_list(circuit, n_virtual)

    out = []
    for v in range(n_virtual):
        p = int(virt_to_phys[v])
        if p in phys_to_cl:
            out.append(bit_at(bitstring, phys_to_cl[p]))
        else:
            out.append(bit_at(bitstring, v))
    return out


def decode_original_bits(
    bitstring: str,
    circuit: QuantumCircuit,
    n_original: int,
    sat_map: Optional[Dict[int, int]] = None,
    sat_dir: Optional[str] = None,
) -> List[int]:
    bits = virt_bits_from_bitstring(bitstring, circuit, n_original)
    if sat_map is not None and sat_dir is not None:
        bits = sat_bits_to_original_bits(bits, sat_map, sat_dir)
    return bits


def bits_to_int_le(bits: Sequence[int]) -> int:
    return int(sum((int(b) & 1) << i for i, b in enumerate(bits)))


# ----------------------------
# Distribution analysis
# ----------------------------

def probs_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    shots = float(sum(counts.values()))
    if shots <= 0:
        return {}
    return {k: v / shots for k, v in counts.items()}


def rows_from_distribution(
    graph: nx.Graph,
    dist: Dict[str, float],
    circuit: QuantumCircuit,
    n_original: int,
    sat_map: Optional[Dict[int, int]] = None,
    sat_dir: Optional[str] = None,
) -> List[DistributionRow]:
    rows: List[DistributionRow] = []
    for bs, p in dist.items():
        bits = decode_original_bits(bs, circuit, n_original, sat_map=sat_map, sat_dir=sat_dir)
        loc = bits_to_int_le(bits)
        c = cut_value(bits, graph)
        rows.append(DistributionRow(bitstring=bs, prob=float(p), bits=bits, loc_int=int(loc), cut=float(c)))
    return rows


def summarize_rows(rows: List[DistributionRow], classical_opt: float, variant: str) -> SummaryStats:
    n = len(rows[0].bits) if rows else 0
    if classical_opt <= 0:
        classical_opt = 1.0

    # most probable
    mode = max(rows, key=lambda r: r.prob)
    # best by cut (tie by prob)
    best = max(rows, key=lambda r: (r.cut, r.prob))
    expected_cut = sum(r.prob * r.cut for r in rows)
    ar_best = best.cut / classical_opt
    ar_expected = expected_cut / classical_opt

    return SummaryStats(
        variant=variant,
        n=n,
        classical_opt=float(classical_opt),
        mode_bitstring=mode.bitstring,
        mode_prob=float(mode.prob),
        mode_cut=float(mode.cut),
        best_bitstring=best.bitstring,
        best_prob=float(best.prob),
        best_cut=float(best.cut),
        expected_cut=float(expected_cut),
        ar_best=float(ar_best),
        ar_expected=float(ar_expected),
        num_support=len(rows),
    )


def analyze_variant(
    graph: nx.Graph,
    counts_or_dist: Dict[str, Any],
    circuit: QuantumCircuit,
    classical_opt: float,
    *,
    is_counts: bool,
    sat_map: Optional[Dict[int, int]] = None,
    sat_dir: Optional[str] = None,
    variant_name: str = "raw",
) -> Tuple[Dict[str, float], List[DistributionRow], SummaryStats]:
    n = graph.number_of_nodes()
    dist = probs_from_counts(counts_or_dist) if is_counts else dict(counts_or_dist)
    rows = rows_from_distribution(graph, dist, circuit, n, sat_map=sat_map, sat_dir=sat_dir)
    summary = summarize_rows(rows, classical_opt, variant_name)
    return dist, rows, summary


# ----------------------------
# Tables + JSON dumps
# ----------------------------

def save_tables(
    outdir: Path,
    tag: str,
    metrics_df: pd.DataFrame,
    summaries_df: pd.DataFrame,
    post_df: Optional[pd.DataFrame] = None,
):
    ensure_dir(outdir)
    metrics_csv = outdir / f"metrics_{tag}.csv"
    summaries_csv = outdir / f"summaries_{tag}.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    summaries_df.to_csv(summaries_csv, index=False)

    # HTML (wide, readable)
    style = (
        "<style>"
        "table {border-collapse: collapse; width: 100%;}"
        "th, td {border: 1px solid #ddd; padding: 6px; white-space: nowrap;}"
        "th {background: #f5f5f5;}"
        "</style>"
    )
    (outdir / f"metrics_{tag}.html").write_text(style + metrics_df.to_html(index=False), encoding="utf-8")
    (outdir / f"summaries_{tag}.html").write_text(style + summaries_df.to_html(index=False), encoding="utf-8")

    if post_df is not None:
        post_csv = outdir / f"post_{tag}.csv"
        post_df.to_csv(post_csv, index=False)
        (outdir / f"post_{tag}.html").write_text(style + post_df.to_html(index=False), encoding="utf-8")


def dump_raw_artifacts_json(
    outdir: Path,
    config: dict,
    graph: nx.Graph,
    backend_name: str,
    circuits: Dict[str, QuantumCircuit],
    artifacts: Dict[str, VariantArtifacts],
    *,
    sat_map: Optional[Dict[int, int]] = None,
    sat_dir: Optional[str] = None,
    min_sat_layers: Optional[int] = None,
    job_id: Optional[str] = None,
):
    ensure_dir(outdir)
    n = graph.number_of_nodes()

    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "backend_name": backend_name,
            "run_mode": config.get("RUN_MODE", "hardware"),
            "shots": int(config.get("SHOTS", 0)),
            "seed": int(config.get("SEED", 0)),
            "reps": int(config.get("REPS", 1)),
            "graph_n": n,
            "graph_edges": graph.number_of_edges(),
            "use_sat_mapping": bool(config.get("USE_SAT_MAPPING", False)),
            "sat_map": sat_map,
            "sat_dir": sat_dir,
            "sat_min_layers": min_sat_layers,
            "job_id": job_id,
        },
        "mapping": {},
        "variants": {}
    }

    for name, circ in circuits.items():
        payload["mapping"][name] = {
            "measured_phys_qubits": measured_phys_qubits_from_measures(circ),
            "virt_to_phys": get_virt_to_phys_list(circ, n),
        }

    for name, art in artifacts.items():
        payload["variants"][name] = {"counts": art.counts}

    tag = run_tag(config, graph, backend_name, active_phys=None)
    path = outdir / f"raw_artifacts_{tag}.json"
    save_json(payload, path)
    print("✅ Saved raw artifacts:", path)
    return path


def dump_analysis_json(outdir: Path, tag: str, obj: dict, name: str):
    path = outdir / f"{name}_{tag}.json"
    save_json(obj, path)
    print(f"✅ Saved {name}:", path)
    return path


def filter_rows(
    rows_sorted_by_prob: List[DistributionRow],
    *,
    method: str = "mass",
    mass_keep: float = 0.90,
    prob_threshold: Optional[float] = None,
    k_max: Optional[int] = None,
) -> Tuple[List[DistributionRow], float]:
    kept: List[DistributionRow] = []
    cum = 0.0
    if method == "mass":
        for r in rows_sorted_by_prob:
            kept.append(r)
            cum += r.prob
            if cum >= mass_keep:
                break
    elif method == "threshold":
        if prob_threshold is None:
            raise ValueError("prob_threshold must be set when method='threshold'")
        for r in rows_sorted_by_prob:
            if r.prob >= prob_threshold:
                kept.append(r)
            else:
                break
        cum = sum(r.prob for r in kept)
    else:
        raise ValueError(f"Unknown method: {method}")

    if k_max is not None:
        kept = kept[:int(k_max)]
        cum = sum(r.prob for r in kept)

    return kept, float(cum)


def renormalize_rows(rows: List[DistributionRow], total_mass: float) -> List[DistributionRow]:
    if total_mass <= 0:
        return []
    out = []
    for r in rows:
        out.append(DistributionRow(
            bitstring=r.bitstring, prob=r.prob / total_mass,
            bits=r.bits, loc_int=r.loc_int, cut=r.cut
        ))
    return out
