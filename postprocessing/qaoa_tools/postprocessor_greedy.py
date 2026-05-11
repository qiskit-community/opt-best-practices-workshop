"""
Greedy one-flip postprocessor for QAOA distributions.

This module implements the HighProbGreedy1Flip postprocessor that applies
greedy local search to improve quantum measurement outcomes.

Reference: Chancellor 2017 (hybrid local search with annealers)
"""

from typing import Dict, List, Optional, Tuple
import networkx as nx

from .utils_general import DistributionRow, cut_value, bits_to_int_le


class DistributionPostProcessor:
    """
    Base class for postprocessors that transform a distribution
    (list of DistributionRow) into a new distribution.
    """
    display_name: str = "BaseProcessor"
    ref: str = ""

    def transform(self, rows: List[DistributionRow], graph: nx.Graph) -> List[DistributionRow]:
        raise NotImplementedError


def _aggregate_rows_from_bits_prob(
    bitprob: Dict[Tuple[int, ...], float],
    graph: nx.Graph
) -> List[DistributionRow]:
    """
    Convert a dictionary of (bits_tuple -> probability) into DistributionRow objects.
    """
    out_rows: List[DistributionRow] = []
    for key, p in bitprob.items():
        bits = list(key)
        loc = bits_to_int_le(bits)
        c = cut_value(bits, graph)
        bs = "".join(str(b) for b in bits[::-1])  # little-endian display
        out_rows.append(
            DistributionRow(
                bitstring=bs,
                prob=float(p),
                bits=bits,
                loc_int=int(loc),
                cut=float(c),
            )
        )
    return out_rows


def _renormalize_prob_dict(bitprob: Dict[Tuple[int, ...], float]) -> Dict[Tuple[int, ...], float]:
    """Renormalize probability dictionary to sum to 1."""
    s = sum(bitprob.values())
    if s <= 0:
        return bitprob
    return {k: v / s for k, v in bitprob.items()}


def _greedy_bitflip_locked(bits: List[int], graph: nx.Graph, locked: Optional[List[bool]] = None) -> List[int]:
    """
    Greedy 1-bit flip hill-climb, optionally preventing flips for locked variables.
    
    Args:
        bits: Initial bitstring
        graph: NetworkX graph defining the MaxCut problem
        locked: Optional list indicating which bits cannot be flipped
        
    Returns:
        Improved bitstring after greedy local search
    """
    n = len(bits)
    locked = locked or [False] * n
    cur = bits.copy()
    cur_val = cut_value(cur, graph)

    improved = True
    while improved:
        improved = False
        for i in range(n):
            if locked[i]:
                continue
            trial = cur.copy()
            trial[i] ^= 1
            v = cut_value(trial, graph)
            if v > cur_val:
                cur, cur_val = trial, v
                improved = True
    return cur


class HighProbGreedy1Flip(DistributionPostProcessor):
    """
    Apply greedy local search to each sample and aggregate probability mass.
    
    This postprocessor takes each bitstring in the distribution, applies a greedy
    1-bit flip local search to improve the cut value, and aggregates the probability
    mass of bitstrings that converge to the same local optimum.
    
    Reference: Chancellor 2017 (hybrid local search with annealers)
    """
    display_name = "HighProb + Greedy(1-flip) polish"
    ref = "Chancellor 2017 (hybrid local search with annealers)"

    def transform(self, rows: List[DistributionRow], graph: nx.Graph) -> List[DistributionRow]:
        """
        Transform distribution by applying greedy local search to each bitstring.
        
        Args:
            rows: Input distribution as list of DistributionRow objects
            graph: NetworkX graph defining the MaxCut problem
            
        Returns:
            Transformed distribution with aggregated probability mass
        """
        agg: Dict[Tuple[int, ...], float] = {}
        for r in rows:
            new_bits = _greedy_bitflip_locked(r.bits, graph)
            key = tuple(int(b) for b in new_bits)
            agg[key] = agg.get(key, 0.0) + float(r.prob)
        agg = _renormalize_prob_dict(agg)
        return _aggregate_rows_from_bits_prob(agg, graph)

