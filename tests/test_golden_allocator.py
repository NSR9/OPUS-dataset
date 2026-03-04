# tests/test_golden_allocator.py
"""Tests for the golden dataset allocation engine."""
import pytest
from scripts.golden_allocator import compute_allocation, validate_allocation


# --- Domain/dataset fixtures ---

SAMPLE_REGISTRY = [
    ("ds_a1", "DomainA"), ("ds_a2", "DomainA"), ("ds_a3", "DomainA"),
    ("ds_b1", "DomainB"), ("ds_b2", "DomainB"),
    ("ds_c1", "DomainC"),
]
SAMPLE_DOMAINS = ["DomainA", "DomainB", "DomainC"]


class TestComputeAllocation:
    def test_total_equals_n(self):
        """Sum of all allocations must equal N."""
        alloc = compute_allocation(
            total=60,
            domain_names=SAMPLE_DOMAINS,
            domain_weights={"DomainA": 1/3, "DomainB": 1/3, "DomainC": 1/3},
            datasets_per_domain={"DomainA": ["ds_a1","ds_a2","ds_a3"],
                                 "DomainB": ["ds_b1","ds_b2"],
                                 "DomainC": ["ds_c1"]},
        )
        assert sum(alloc.values()) == 60

    def test_every_dataset_at_least_one(self):
        """Every dataset must get >= 1 sample."""
        alloc = compute_allocation(
            total=6,  # exactly num_datasets, minimum possible
            domain_names=SAMPLE_DOMAINS,
            domain_weights={"DomainA": 1/3, "DomainB": 1/3, "DomainC": 1/3},
            datasets_per_domain={"DomainA": ["ds_a1","ds_a2","ds_a3"],
                                 "DomainB": ["ds_b1","ds_b2"],
                                 "DomainC": ["ds_c1"]},
        )
        for ds_id, count in alloc.items():
            assert count >= 1, f"{ds_id} got {count} samples"

    def test_too_few_samples_raises(self):
        """N < num_datasets must raise ValueError."""
        with pytest.raises(ValueError, match="at least 6"):
            compute_allocation(
                total=5,
                domain_names=SAMPLE_DOMAINS,
                domain_weights={"DomainA": 1/3, "DomainB": 1/3, "DomainC": 1/3},
                datasets_per_domain={"DomainA": ["ds_a1","ds_a2","ds_a3"],
                                     "DomainB": ["ds_b1","ds_b2"],
                                     "DomainC": ["ds_c1"]},
            )

    def test_uniform_distribution(self):
        """With uniform weights, domains get roughly equal budgets."""
        alloc = compute_allocation(
            total=120,
            domain_names=SAMPLE_DOMAINS,
            domain_weights={"DomainA": 1/3, "DomainB": 1/3, "DomainC": 1/3},
            datasets_per_domain={"DomainA": ["ds_a1","ds_a2","ds_a3"],
                                 "DomainB": ["ds_b1","ds_b2"],
                                 "DomainC": ["ds_c1"]},
        )
        domain_a_total = alloc["ds_a1"] + alloc["ds_a2"] + alloc["ds_a3"]
        domain_b_total = alloc["ds_b1"] + alloc["ds_b2"]
        domain_c_total = alloc["ds_c1"]
        assert domain_a_total == 40
        assert domain_b_total == 40
        assert domain_c_total == 40

    def test_domain_clamped_to_num_datasets(self):
        """Domain with many datasets can't get fewer samples than datasets."""
        alloc = compute_allocation(
            total=7,  # 6 datasets + 1 extra
            domain_names=SAMPLE_DOMAINS,
            domain_weights={"DomainA": 0.01, "DomainB": 0.01, "DomainC": 0.98},
            datasets_per_domain={"DomainA": ["ds_a1","ds_a2","ds_a3"],
                                 "DomainB": ["ds_b1","ds_b2"],
                                 "DomainC": ["ds_c1"]},
        )
        # DomainA has 3 datasets, must get at least 3 even with tiny weight
        domain_a_total = alloc["ds_a1"] + alloc["ds_a2"] + alloc["ds_a3"]
        assert domain_a_total >= 3

    def test_within_domain_even_split(self):
        """Within a domain, samples split evenly across datasets."""
        alloc = compute_allocation(
            total=120,
            domain_names=SAMPLE_DOMAINS,
            domain_weights={"DomainA": 1/3, "DomainB": 1/3, "DomainC": 1/3},
            datasets_per_domain={"DomainA": ["ds_a1","ds_a2","ds_a3"],
                                 "DomainB": ["ds_b1","ds_b2"],
                                 "DomainC": ["ds_c1"]},
        )
        # DomainA gets 40, split across 3 => 14, 13, 13 or 13, 13, 14
        assert alloc["ds_a1"] in (13, 14)
        assert alloc["ds_a2"] in (13, 14)
        assert alloc["ds_a3"] in (13, 14)


class TestValidateAllocation:
    def test_valid_passes(self):
        """Valid allocation passes validation."""
        alloc = {"ds1": 3, "ds2": 2, "ds3": 1}
        validate_allocation(alloc, total=6, all_dataset_ids=["ds1","ds2","ds3"])

    def test_wrong_total_raises(self):
        alloc = {"ds1": 3, "ds2": 2, "ds3": 1}
        with pytest.raises(AssertionError):
            validate_allocation(alloc, total=7, all_dataset_ids=["ds1","ds2","ds3"])

    def test_missing_dataset_raises(self):
        alloc = {"ds1": 3, "ds2": 3}  # ds3 missing
        with pytest.raises(AssertionError):
            validate_allocation(alloc, total=6, all_dataset_ids=["ds1","ds2","ds3"])

    def test_zero_allocation_raises(self):
        alloc = {"ds1": 6, "ds2": 0, "ds3": 0}
        with pytest.raises(AssertionError):
            validate_allocation(alloc, total=6, all_dataset_ids=["ds1","ds2","ds3"])
