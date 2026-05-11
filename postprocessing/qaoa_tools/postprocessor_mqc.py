"""
MQC-style multi-qubit correction postprocessor for QAOA distributions.

This module implements the MQCStyleMultiQubitCorrection postprocessor that uses
reference solutions to identify and correct groups of qubits that may be flipped
together to improve the objective value.

Reference: Ayanzadeh et al. Sci Rep 11, 16119 (2021)
"""

from typing import Dict, List, Tuple
import networkx as nx

from .utils_general import DistributionRow, cut_value, select_reference_rows
from .postprocessor_greedy import (
    DistributionPostProcessor,
    _aggregate_rows_from_bits_prob,
    _renormalize_prob_dict,
    _greedy_bitflip_locked,
)


class MQCStyleMultiQubitCorrection(DistributionPostProcessor):
    """
    MQC-inspired multi-qubit correction + greedy polish.
    
    This postprocessor selects high-quality reference solutions and uses them
    to identify groups of qubits (connected components in the difference graph)
    that can be flipped together to improve the objective value. After multi-qubit
    corrections, a final greedy polish is applied.
    
    Reference: Ayanzadeh et al. Sci Rep 11, 16119 (2021)
    """
    display_name = "MQC-style multi-qubit correction + polish"
    ref = "Ayanzadeh et al. Sci Rep 11, 16119 (2021)"

    def __init__(
        self,
        top_m: int = 200,
        max_passes: int = 3,
        reference_mode: str = "hybrid",
    ):
        """
        Initialize MQC-style postprocessor.
        
        Args:
            top_m: Number of top reference solutions to use
            max_passes: Maximum number of correction passes
            reference_mode: How to select references ("prob", "objective", or "hybrid")
        """
        self.top_m = int(top_m)
        self.max_passes = int(max_passes)
        self.reference_mode = reference_mode

    def _try_group_flip(
        self,
        bits: List[int],
        group: List[int],
        graph: nx.Graph,
    ) -> Tuple[bool, List[int], float]:
        """
        Try flipping a group of qubits and check if it improves the cut value.
        
        Args:
            bits: Current bitstring
            group: Indices of qubits to flip together
            graph: NetworkX graph defining the MaxCut problem
            
        Returns:
            Tuple of (improved, new_bits, new_value)
        """
        cur_val = cut_value(bits, graph)
        trial = bits.copy()
        for i in group:
            trial[i] ^= 1
        v = cut_value(trial, graph)
        return (v > cur_val), trial, v

    def transform(self, rows: List[DistributionRow], graph: nx.Graph) -> List[DistributionRow]:
        """
        Transform distribution using MQC-style multi-qubit correction.
        
        Args:
            rows: Input distribution as list of DistributionRow objects
            graph: NetworkX graph defining the MaxCut problem
            
        Returns:
            Transformed distribution with improved solutions
        """
        if not rows:
            return rows

        n = len(rows[0].bits)
        ref_rows = select_reference_rows(rows, top_m=self.top_m, mode=self.reference_mode)
        ref_bits = [r.bits for r in ref_rows]
        G = graph

        def components_of_diff(diff_idx: List[int]) -> List[List[int]]:
            """Find connected components in the subgraph of differing qubits."""
            if not diff_idx:
                return []
            sub = G.subgraph(diff_idx)
            return [list(comp) for comp in nx.connected_components(sub)]

        agg: Dict[Tuple[int, ...], float] = {}

        for seed_row in ref_rows:
            bits = seed_row.bits.copy()

            for _ in range(self.max_passes):
                improved_any = False
                for other in ref_bits:
                    diff = [i for i in range(n) if bits[i] != other[i]]
                    if not diff:
                        continue

                    groups = components_of_diff(diff)
                    if not groups:
                        continue

                    best_trial = None
                    best_val = cut_value(bits, G)

                    for grp in groups:
                        ok, trial, val = self._try_group_flip(bits, grp, G)
                        if ok and val > best_val:
                            best_val = val
                            best_trial = trial

                    if best_trial is not None:
                        bits = best_trial
                        improved_any = True

                if not improved_any:
                    break

            # Final greedy polish
            bits = _greedy_bitflip_locked(bits, G)
            key = tuple(int(b) for b in bits)
            agg[key] = agg.get(key, 0.0) + float(seed_row.prob)

        agg = _renormalize_prob_dict(agg)
        return _aggregate_rows_from_bits_prob(agg, graph)


def run_postprocessing_suite(
    rows_selected: List[DistributionRow],
    graph: nx.Graph,
    classical_opt: float,
    variant_label: str,
    processors: List[DistributionPostProcessor],
):
    """
    Apply multiple postprocessing methods to the same selected distribution.
    
    Args:
        rows_selected: Selected high-probability distribution rows
        graph: NetworkX graph defining the MaxCut problem
        classical_opt: Classical optimal cut value for comparison
        variant_label: Label for this variant (e.g., "baseline")
        processors: List of postprocessor instances to apply
        
    Returns:
        Tuple of (summary_dataframe, dict_of_postprocessed_rows)
    """
    import pandas as pd
    from .utils_general import summarize_rows, DistributionRow
    from typing import Any
    
    out_rows: Dict[str, List[DistributionRow]] = {}
    rows_summary: List[Dict[str, Any]] = []

    for proc in processors:
        pp_rows = proc.transform(rows_selected, graph)

        # renormalize just in case
        total = sum(r.prob for r in pp_rows)
        if total > 0:
            pp_rows = [
                DistributionRow(
                    bitstring=r.bitstring,
                    prob=float(r.prob) / total,
                    bits=r.bits,
                    loc_int=r.loc_int,
                    cut=r.cut,
                )
                for r in pp_rows
            ]

        summ = summarize_rows(pp_rows, classical_opt, variant=f"{variant_label} | {proc.display_name}")

        rows_summary.append({
            "variant": variant_label,
            "method": proc.display_name,
            "reference": proc.ref,
            "expected_cut": summ.expected_cut,
            "best_cut": summ.best_cut,
            "ar_expected": summ.ar_expected,
            "ar_best": summ.ar_best,
            "num_support": summ.num_support,
        })
        out_rows[proc.display_name] = pp_rows

    return pd.DataFrame(rows_summary), out_rows

