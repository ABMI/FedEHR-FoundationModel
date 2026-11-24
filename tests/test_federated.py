"""Site sampling, FedAvg averaging, client parsing and eval-folder links on synthetic data."""

import polars as pl
import pytest
import torch

from ausom_ehr import federated


def synthetic_profile(n=3_000):
    ids = list(range(1, n + 1))
    frame = pl.DataFrame({
        "person_id": ids,
        "reference": [i <= 200 for i in ids],
        "gender": [federated.FEMALE if i % 2 else federated.MALE for i in ids],
        "year_of_birth": [1940 if i % 5 == 0 else 2010 if i % 5 == 1 else 1990 for i in ids],
        "end_year": [9999 if i % 7 == 0 else 2020 for i in ids],
        "visits": [0 if i % 9 == 0 else 3 for i in ids],
        "inpatient_visits": [1 if i % 4 == 0 else 0 for i in ids],
        "er_visits": [1 if i % 3 == 0 else 0 for i in ids],
        "outpatient_visits": [2 for _ in ids],
        **{f"dx_{name}": [i % 11 == 0 for i in ids] for name in federated.DISEASE_GROUPS},
    }).with_columns(
        (pl.min_horizontal(pl.col("end_year"), pl.lit(federated.DATA_END_YEAR)) - pl.col("year_of_birth")).alias("age_last")
    )
    return federated.add_axes(frame)


def test_sites_hit_targets_and_never_overlap_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(federated, "SITE_SIZE", 500)
    profile = synthetic_profile()
    summary = federated.make_sites(profile, tmp_path, seed=1)
    reference_ids = profile.filter(pl.col("reference"))["person_id"]
    for name in federated.SITES:
        site = pl.read_parquet(tmp_path / f"{name}.parquet")
        assert site.height == 500 and site["person_id"].n_unique() == 500
        assert not site["person_id"].is_in(reference_ids.implode()).any()
        assert site["split"].value_counts().sort("split")["count"].to_list() == [75, 350, 75]
    assert summary["sites"]["D1_elderly"]["age_group_%"]["65+"] == 80.0
    assert summary["sites"]["D4_female"]["female_%"] == 80.0
    assert summary["sites"]["D2_inpatient"]["ever_inpatient_%"] == 80.0
    # the cancer site takes every eligible patient
    pool_cancer = profile.filter(~pl.col("reference") & pl.col("dx_cancer")).height
    assert round(summary["sites"]["D3_cancer"]["disease_%"]["cancer"] * 5) == pool_cancer
    # filler end year 9999 is capped at DATA_END_YEAR for age
    assert profile.filter(pl.col("end_year") == 9999)["age_last"].max() <= federated.DATA_END_YEAR - 1940


def test_disjoint_sites_hit_targets_and_share_no_patients(tmp_path, monkeypatch):
    monkeypatch.setattr(federated, "SITE_SIZE", 200)
    profile = synthetic_profile()
    pool_ids = profile.filter(~pl.col("reference"))["person_id"]
    first = federated.assign_splits(pool_ids.sample(200, seed=0).sort(), federated.np.random.default_rng(0))
    first_path = tmp_path / "first" / "D0_random.parquet"
    first_path.parent.mkdir()
    first.write_parquet(first_path)
    summary = federated.make_disjoint_sites(profile, first_path, tmp_path, seed=1)
    names = ["D0_random", *federated.DISJOINT_SITES]
    sites = {name: pl.read_parquet(tmp_path / f"{name}.parquet") for name in names}
    assert sites["D0_random"].equals(first)
    every = pl.concat([site["person_id"] for site in sites.values()])
    assert every.len() == 200 * len(names) and every.n_unique() == every.len()
    assert not every.is_in(profile.filter(pl.col("reference"))["person_id"].implode()).any()
    for name in federated.DISJOINT_SITES:
        assert sites[name]["split"].value_counts().sort("split")["count"].to_list() == [30, 140, 30]
    assert summary["sites"]["D1_pediatric"]["age_group_%"]["0-18"] == 80.0
    assert summary["sites"]["D2_er"]["ever_er_%"] == 80.0
    assert summary["sites"]["D3_cancer"]["disease_%"]["cancer"] == 25.0
    assert summary["sites"]["D4_female"]["female_%"] == 80.0
    assert summary["sites"]["D5_elderly"]["age_group_%"]["65+"] == 60.0


def test_fedavg_weights_by_tokens_and_keeps_integer_buffers():
    a = {"w": torch.tensor([1.0, 3.0]), "count": torch.tensor([7])}
    b = {"w": torch.tensor([3.0, 7.0]), "count": torch.tensor([9])}
    merged = federated.average([a, b], [1, 3])
    assert torch.allclose(merged["w"], torch.tensor([2.5, 6.0]))
    assert merged["count"].item() == 7


def test_parse_clients_reads_name_config_and_gpu():
    clients = federated.parse_clients(["A=configs/a.toml@3", "D1=configs/site@x.toml@4"])
    assert clients == {"A": ("configs/a.toml", "3"), "D1": ("configs/site@x.toml", "4")}


def test_link_is_idempotent_and_refuses_other_targets(tmp_path):
    target = tmp_path / "model" / "state.pt"
    target.parent.mkdir()
    target.write_bytes(b"weights")
    link_path = tmp_path / "eval" / "training" / "continued" / "best" / "state.pt"
    federated.link(target, link_path)
    federated.link(target, link_path)
    assert link_path.resolve() == target.resolve()

    other = tmp_path / "other.pt"
    other.write_bytes(b"other")
    with pytest.raises(ValueError, match="linked to a different input"):
        federated.link(other, link_path)
    real_file = tmp_path / "eval" / "real.pt"
    real_file.write_bytes(b"real")
    with pytest.raises(FileExistsError):
        federated.link(target, real_file)
