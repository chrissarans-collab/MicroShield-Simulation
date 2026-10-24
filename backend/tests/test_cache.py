"""Tests for the cache simulator engine (Stage 3).

Layered verification:
1. fixtures/cache.json — hand-computed set/tag mapping, eviction order and
   3C classification, shared with the TypeScript suite.
2. Geometry validation (powers of two, exact set division, policy names).
3. Access-text parser (R/W tokens, hex, error token indices).
"""

import json
from pathlib import Path

import pytest

from app.core.cache import (
    CacheConfig,
    CacheParseError,
    parse_accesses,
    simulate_cache,
    validate_cache_config,
)

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "cache.json"


def _load_cases():
    return json.loads(FIXTURE_PATH.read_text())["cases"]


def _config(raw: dict) -> CacheConfig:
    return CacheConfig(
        capacity=raw.get("capacity", 256),
        line_size=raw.get("line_size", 16),
        associativity=raw.get("associativity", 4),
        replacement=raw.get("replacement", "lru"),
    )


def _assert_case(case: dict) -> None:
    accesses = parse_accesses(" ".join(case["accesses"])) if case["accesses"] else []
    result = simulate_cache(accesses, _config(case["config"]))
    expected = case["expected"]

    assert result.sets == expected["sets"]
    assert result.accesses == expected["accesses"]
    if expected.get("reads") is not None:
        assert result.reads == expected["reads"]
    if expected.get("writes") is not None:
        assert result.writes == expected["writes"]
    assert result.hits == expected["hits"]
    assert result.misses == expected["misses"]
    if expected.get("hit_rate") is None:
        assert result.hit_rate is None
    else:
        assert result.hit_rate == pytest.approx(expected["hit_rate"], abs=1e-9)
    if expected.get("read_hits") is not None:
        assert result.read_hits == expected["read_hits"]
    if expected.get("write_hits") is not None:
        assert result.write_hits == expected["write_hits"]
    assert result.evictions == expected["evictions"]
    assert result.misses_by_type == expected["misses_by_type"]
    assert result.final_state == expected["final_state"]

    if "steps" in expected:
        assert len(result.steps) == len(case["accesses"])
        for want in expected["steps"]:
            got = result.steps[want["index"]]
            for key, value in want.items():
                assert got[key] == value


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["name"])
def test_cache_fixture_cases(case):
    _assert_case(case)


def test_fully_associative_cache_never_reports_conflict_misses():
    config = CacheConfig(capacity=64, line_size=16, associativity=4)  # 1 set
    accesses = parse_accesses("R0 R16 R32 R48 R0 R64 R16")
    result = simulate_cache(accesses, config)
    assert result.sets == 1
    assert result.misses_by_type["conflict"] == 0
    # 5 distinct blocks > 4 ways: R16 comes back after being evicted by R64
    assert result.misses_by_type["capacity"] == 1


def test_direct_mapped_misses_are_compulsory_or_conflict():
    config = CacheConfig(capacity=16, line_size=4, associativity=1)
    result = simulate_cache(parse_accesses("R0 R16 R0 R4 R0"), config)
    assert result.misses_by_type["capacity"] == 0
    assert result.misses_by_type["conflict"] >= 1


def test_step_state_after_is_a_snapshot():
    result = simulate_cache(parse_accesses("R0 R4"), CacheConfig(capacity=8, line_size=4, associativity=1))
    assert result.steps[0]["state_after"] == [[0], []]
    assert result.steps[1]["state_after"] == [[0], [4]]


def test_geometry_validation_rejects_bad_configs():
    assert validate_cache_config(CacheConfig(capacity=48, line_size=16, associativity=1))
    assert validate_cache_config(CacheConfig(capacity=64, line_size=8, associativity=3))
    assert validate_cache_config(CacheConfig(capacity=64, line_size=12, associativity=2))
    assert validate_cache_config(CacheConfig(capacity=64, line_size=16, associativity=8))
    assert validate_cache_config(CacheConfig(capacity=64, line_size=16, associativity=4, replacement="mru"))
    assert not validate_cache_config(CacheConfig(capacity=64, line_size=16, associativity=4))
    with pytest.raises(ValueError):
        simulate_cache([("R", 0)], CacheConfig(capacity=48, line_size=16, associativity=1))


def test_parse_accesses_accepts_bases_and_reports_token_indices():
    accesses = parse_accesses("R16, W0x20\nr4 w0b100")
    assert accesses == [("R", 16), ("W", 32), ("R", 4), ("W", 4)]

    with pytest.raises(CacheParseError) as exc:
        parse_accesses("R16 X32")
    assert exc.value.token == 1

    with pytest.raises(CacheParseError):
        parse_accesses("R")
    with pytest.raises(CacheParseError):
        parse_accesses("R-4")
    with pytest.raises(CacheParseError):
        parse_accesses("R" + str(2**33))
