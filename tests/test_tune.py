import copy
import json

from suppmatch import config
from suppmatch.candidates import build_candidates
from suppmatch.decide import MATCH, decide
from suppmatch.tune import tune, write_outputs


def test_tuning_reads_only_dev_rows(built, records, fake_embedder):
    plan = built["plan"]
    result, rows, _, _ = tune(records, plan, fake_embedder, "fake")
    assert rows and all(plan["vendors"][k]["split"] == "dev" for k in rows)
    assert result.dev_rows == sum(v["split"] == "dev" for v in plan["vendors"].values())


def test_heldout_and_redteam_labels_do_not_change_thresholds(built, records, fake_embedder):
    plan = built["plan"]
    result, _, _, _ = tune(records, plan, fake_embedder, "fake")
    poisoned = copy.deepcopy(plan)
    for v in poisoned["vendors"].values():
        if v["split"] != "dev":
            v.update(expected_outcome="MATCH", allowed_outcomes=["MATCH"], expected_fusion_vendor_id=1)
    again, _, _, _ = tune(records, poisoned, fake_embedder, "fake")
    assert (again.t_low, again.t_high) == (result.t_low, result.t_high)


def test_chosen_t_low_sends_no_dev_duplicate_to_new(built, records, fake_embedder):
    plan = built["plan"]
    result, rows, _, _ = tune(records, plan, fake_embedder, "fake")
    assert result.duplicates_below_t_low == 0
    dups = [k for k in rows if plan["vendors"][k]["expected_outcome"] == MATCH]
    cands = build_candidates({k: records.sap[k] for k in dups}, records.fusion, fake_embedder)
    for k in dups:
        assert decide(cands[k], result.t_low, result.t_high).rule != "SURE_NEW", k


def test_chosen_t_high_meets_precision_target(built, records, fake_embedder):
    result, _, _, _ = tune(records, built["plan"], fake_embedder, "fake")
    assert result.target_met and result.precision >= 0.98
    assert result.t_low < result.t_high


def test_config_reads_thresholds_file_for_that_embedder(built, records, fake_embedder, tmp_path):
    assert config.thresholds("fake", tmp_path) == (config.T_LOW, config.T_HIGH, "placeholder (config.py)")
    result, _, high, low = tune(records, built["plan"], fake_embedder, "fake")
    json_path, md_path = write_outputs(result, high, low, reports_dir=tmp_path, data_dir=tmp_path)
    assert json_path.name == "thresholds_fake.json" and md_path.name == "thresholds_fake.md"
    assert json.loads(json_path.read_text())["tuned_on"] == "dev"
    t_low, t_high, source = config.thresholds("fake", tmp_path)
    assert (t_low, t_high, source) == (result.t_low, result.t_high, "thresholds_fake.json")
    # a different embedder does not pick up the fake file
    assert config.thresholds("local", tmp_path)[2].startswith("placeholder")
