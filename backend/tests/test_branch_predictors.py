"""Tests for the branch prediction engine (Stage 1).

Two layers of verification:
1. Hand-computed fixtures (fixtures/branch_predictor.json) — every expected
   number was worked out on paper from the documented model, asserted by BOTH
   the Python and TypeScript suites.
2. Focused unit tests for the tricky internals (saturating counter dynamics,
   gshare index folding, FNV-1a test vectors, input parsing, validation).
"""

import json
from pathlib import Path

import pytest

from app.core.branch_predictors import (
    MAX_SEQUENCE_LENGTH,
    BranchConfig,
    BranchEvent,
    GsharePredictor,
    fnv1a32,
    parse_sequence,
    simulate,
)

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "branch_predictor.json"


def _load_cases():
    data = json.loads(FIXTURE_PATH.read_text())
    return data["cases"]


def _config_from(fixture_config: dict) -> BranchConfig:
    defaults = BranchConfig()
    return BranchConfig(
        predictor=fixture_config.get("predictor", defaults.predictor),
        initial_prediction=fixture_config.get("initial_prediction", defaults.initial_prediction),
        initial_counter=fixture_config.get("initial_counter", defaults.initial_counter),
        initial_ghr=fixture_config.get("initial_ghr", defaults.initial_ghr),
        history_bits=fixture_config.get("history_bits", defaults.history_bits),
        misprediction_penalty=fixture_config.get(
            "misprediction_penalty", defaults.misprediction_penalty
        ),
    )


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["name"])
def test_fixture_cases(case):
    events = [BranchEvent(label=e["label"], outcome=e["outcome"]) for e in case["input"]["sequence"]]
    result = simulate(events, _config_from(case["input"]["config"]))
    expected = case["expected"]

    assert result.total == expected["total"]
    assert result.correct == expected["correct"]
    assert result.mispredicted == expected["mispredicted"]
    assert result.accuracy == pytest.approx(expected["accuracy"], abs=1e-4)
    assert result.penalty_cycles == expected["penalty_cycles"]
    assert [s.prediction for s in result.steps] == expected["predictions"]

    if "state_before" in expected:
        assert [s.state_before for s in result.steps] == expected["state_before"]
    if "state_after" in expected:
        assert [s.state_after for s in result.steps] == expected["state_after"]
    if "label_stats" in expected:
        got = {ls.label: (ls.total, ls.correct) for ls in result.label_stats}
        want = {ls["label"]: (ls["total"], ls["correct"]) for ls in expected["label_stats"]}
        assert got == want


def test_fnv1a32_standard_vectors():
    # Published FNV-1a 32-bit test vectors.
    assert fnv1a32("") == 0x811C9DC5
    assert fnv1a32("a") == 0xE40C292C
    assert fnv1a32("foobar") == 0xBF9CF968


def test_two_bit_saturates_and_counts_hands():
    seq = [BranchEvent("B1", o) for o in (True, True, True, True, True)]
    result = simulate(seq, BranchConfig(predictor="two_bit", initial_counter=3))
    # Starts Strongly Taken: five correct predictions, counter never leaves 3.
    assert result.correct == 5
    assert result.steps[-1].state_after == "3 ST"


def test_one_bit_unseen_label_uses_initial_state():
    seq = [BranchEvent("brand_new", True)]
    result = simulate(seq, BranchConfig(predictor="one_bit", initial_prediction="not_taken"))
    assert result.steps[0].prediction is False
    assert result.steps[0].correct is False
    assert result.steps[0].state_before == "init(N)"


def test_ghr_register_evolution():
    p = GsharePredictor(history_bits=3, initial_counter=2, initial_ghr=0)
    for outcome in (True, False, True):
        p.update("B1", outcome)
    # History shifts left; newest outcome lands in the low bit: T,N,T -> 0b101.
    assert p.ghr == 0b101


def test_ghr_index_folds_label_and_history():
    p = GsharePredictor(history_bits=4, initial_counter=2, initial_ghr=0b1011)
    expected_index = (fnv1a32("loop") % p.table_size) ^ 0b1011
    assert p._index("loop") == expected_index


def test_ghr_prediction_uses_index_before_history_shifts():
    # Verify predict() and update() agree on the table index within one event.
    p = GsharePredictor(history_bits=2, initial_counter=0, initial_ghr=0)
    idx_before = p._index("B1")
    p.predict("B1")
    p.update("B1", True)
    assert p.table[idx_before] == 1  # 0 -> 1 (weakly taken) at the pre-shift index


def test_parse_sequence_plain_and_named():
    events = parse_sequence("T n 1 0 taken not")
    assert [e.outcome for e in events] == [True, False, True, False, True, False]
    assert all(e.label == "B1" for e in events)

    named = parse_sequence("loop:T if:N loop:N")
    assert [(e.label, e.outcome) for e in named] == [
        ("loop", True),
        ("if", False),
        ("loop", False),
    ]


def test_parse_sequence_rejects_garbage():
    with pytest.raises(ValueError):
        parse_sequence("T N maybe")
    with pytest.raises(ValueError):
        parse_sequence("bad-label:T")


def test_empty_sequence_is_safe():
    result = simulate([], BranchConfig())
    assert (result.total, result.correct, result.mispredicted) == (0, 0, 0)
    assert result.accuracy == 0.0
    assert result.penalty_cycles == 0
    assert result.steps == []


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        simulate([], BranchConfig(predictor="magic"))
    with pytest.raises(ValueError):
        simulate([], BranchConfig(initial_counter=9))
    with pytest.raises(ValueError):
        simulate([], BranchConfig(history_bits=0))


def test_sequence_length_cap():
    seq = [BranchEvent("B1", True)] * (MAX_SEQUENCE_LENGTH + 1)
    with pytest.raises(ValueError):
        simulate(seq, BranchConfig())


def test_misprediction_penalty_is_configurable():
    seq = [BranchEvent("B1", False), BranchEvent("B1", False)]
    a = simulate(seq, BranchConfig(predictor="always_taken", misprediction_penalty=1))
    b = simulate(seq, BranchConfig(predictor="always_taken", misprediction_penalty=5))
    assert a.penalty_cycles == 2
    assert b.penalty_cycles == 10
