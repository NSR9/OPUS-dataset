# scripts/golden_allocator.py
"""
Allocation engine for the OPUS golden dataset builder.

Distributes N total samples across domains and datasets.
Rules:
  1. Every dataset gets >= 1 sample
  2. Domain budgets determined by weights (default: uniform 1/num_domains)
  3. Within-domain: samples split evenly across datasets
  4. Rounding remainders distributed round-robin
"""
from __future__ import annotations
from typing import Dict, List
import math


def compute_allocation(
    total: int,
    domain_names: List[str],
    domain_weights: Dict[str, float],
    datasets_per_domain: Dict[str, List[str]],
) -> Dict[str, int]:
    """Compute per-dataset sample allocation.

    Args:
        total: Total number of samples (e.g., 128)
        domain_names: Ordered list of domain names
        domain_weights: Domain name -> weight (should sum to ~1.0)
        datasets_per_domain: Domain name -> list of dataset IDs

    Returns:
        Dict mapping dataset_id -> sample count

    Raises:
        ValueError: If total < number of datasets
    """
    all_datasets = []
    for domain in domain_names:
        all_datasets.extend(datasets_per_domain[domain])
    num_datasets = len(all_datasets)

    if total < num_datasets:
        raise ValueError(
            f"Cannot allocate: need at least {num_datasets} samples "
            f"for {num_datasets} datasets, got {total}"
        )

    # --- Step 1: Domain-level allocation ---
    # Raw allocation by weight
    weight_sum = sum(domain_weights[d] for d in domain_names)
    raw_domain_budgets = {}
    for domain in domain_names:
        raw = (domain_weights[domain] / weight_sum) * total
        raw_domain_budgets[domain] = raw

    # Floor each, clamp to at least num_datasets_in_domain
    domain_budgets = {}
    for domain in domain_names:
        nd = len(datasets_per_domain[domain])
        domain_budgets[domain] = max(nd, math.floor(raw_domain_budgets[domain]))

    # Distribute remainder to hit total
    allocated = sum(domain_budgets.values())
    remainder = total - allocated
    if remainder > 0:
        # Sort domains by fractional part (descending) to distribute fairly
        frac_parts = []
        for domain in domain_names:
            frac = raw_domain_budgets[domain] - math.floor(raw_domain_budgets[domain])
            frac_parts.append((frac, domain))
        frac_parts.sort(reverse=True)
        for i in range(remainder):
            domain_budgets[frac_parts[i % len(frac_parts)][1]] += 1
    elif remainder < 0:
        # Over-allocated due to clamping. Steal from domains with most headroom.
        overshoot = -remainder
        headroom = []
        for domain in domain_names:
            nd = len(datasets_per_domain[domain])
            spare = domain_budgets[domain] - nd
            headroom.append((spare, domain))
        headroom.sort(reverse=True)
        for i in range(overshoot):
            d = headroom[i % len(headroom)][1]
            nd = len(datasets_per_domain[d])
            if domain_budgets[d] > nd:
                domain_budgets[d] -= 1

    # --- Step 2: Within-domain distribution ---
    allocation = {}
    for domain in domain_names:
        ds_list = datasets_per_domain[domain]
        budget = domain_budgets[domain]
        nd = len(ds_list)
        base = budget // nd
        extra = budget % nd
        for i, ds_id in enumerate(ds_list):
            allocation[ds_id] = base + (1 if i < extra else 0)

    return allocation


def validate_allocation(
    allocation: Dict[str, int],
    total: int,
    all_dataset_ids: List[str],
) -> None:
    """Assert allocation is valid."""
    assert sum(allocation.values()) == total, (
        f"Allocation sums to {sum(allocation.values())}, expected {total}"
    )
    for ds_id in all_dataset_ids:
        assert ds_id in allocation, f"Dataset {ds_id} missing from allocation"
        assert allocation[ds_id] >= 1, f"Dataset {ds_id} has 0 samples"
