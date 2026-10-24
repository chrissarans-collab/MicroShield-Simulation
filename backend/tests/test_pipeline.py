"""Tests for the 5-stage pipeline engine (Stage 2).

Layered verification:
1. fixtures/pipeline.json — cycle counts, stalls, flushes and traces derived
   on paper from the documented model, shared with the TypeScript suite.
2. Parser edge cases (labels, comments, hex, error messages with line numbers).
3. Termination: empty program, cycle-limit on infinite loops, zero-retired CPI.
"""

import json
from pathlib import Path

import pytest

from app.core.pipeline import (
    PipelineConfig,
    PipelineParseError,
    parse_program,
    simulate_pipeline,
)

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "pipeline.json"


def _load_cases():
    return json.loads(FIXTURE_PATH.read_text())["cases"]


def _config(raw: dict) -> PipelineConfig:
    return PipelineConfig(
        forwarding=raw.get("forwarding", True),
        branch_policy=raw.get("branch_policy", "static_not_taken"),
        memory_latency=raw.get("memory_latency", 1),
        initial_counter=raw.get("initial_counter", 2),
        initial_prediction=raw.get("initial_prediction", "taken"),
        history_bits=raw.get("history_bits", 4),
        initial_ghr=raw.get("initial_ghr", 0),
        max_cycles=raw.get("max_cycles", 5000),
    )


def _assert_case(case: dict) -> None:
    result = simulate_pipeline(case["program"], _config(case["config"]))
    expected = case["expected"]

    assert result.total_cycles == expected["total_cycles"]
    assert result.retired == expected["retired"]
    if "cpi" in expected:
        if expected["cpi"] is None:
            assert result.cpi is None
        else:
            assert result.cpi == pytest.approx(expected["cpi"], abs=1e-4)
    if "hazard_stall_cycles" in expected:
        assert result.hazard_stall_cycles == expected["hazard_stall_cycles"]
    if "mem_stall_cycles" in expected:
        assert result.mem_stall_cycles == expected["mem_stall_cycles"]
    if "flush_cycles" in expected:
        assert result.flush_cycles == expected["flush_cycles"]
    if "mispredicts" in expected:
        assert result.mispredicts == expected["mispredicts"]
    if "branches" in expected:
        assert result.branches == expected["branches"]
    assert result.termination == expected["termination"]

    if "trace_texts" in expected:
        got = [
            [row[stage]["text"] if row[stage]["kind"] == "instr" else "" for stage in ("IF", "ID", "EX", "MEM", "WB")]
            for row in result.trace
        ]
        assert got == expected["trace_texts"]

    if "final_registers" in expected:
        assert result.final_registers == expected["final_registers"]

    if "branch_events" in expected:
        assert len(result.branch_events) == len(expected["branch_events"])
        for got_event, want in zip(result.branch_events, expected["branch_events"]):
            for key, value in want.items():
                assert got_event[key] == value

    if "flush_cells" in expected:
        for cell in expected["flush_cells"]:
            row = result.trace[cell["cycle"] - 1]
            assert row[cell["stage"]]["state"] == "flush"

    if "retired_texts" in expected:
        assert [r["text"] for r in result.retired_instructions] == expected["retired_texts"]

    if "memory_events" in expected:
        assert len(result.memory_events) == len(expected["memory_events"])
        for want, got in zip(expected["memory_events"], result.memory_events):
            for key, value in want.items():
                assert got[key] == value

    if "memory_writes" in expected:
        got_writes = [
            {"address": w["address"], "value": w["value"]}
            for w in result.memory_writes
        ]
        assert got_writes == expected["memory_writes"]


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["name"])
def test_pipeline_fixture_cases(case):
    _assert_case(case)


def test_cpi_formula_and_zero_retired():
    result = simulate_pipeline("")
    assert result.retired == 0
    assert result.cpi is None  # never divide by zero
    single = simulate_pipeline("ADDI x1, x0, 5")
    assert single.cpi == pytest.approx(single.total_cycles / single.retired)


def test_infinite_loop_hits_cycle_limit():
    # A self-branch with static not-taken never retires past the loop.
    result = simulate_pipeline(
        "loop: BEQ x0, x0, loop", PipelineConfig(max_cycles=50)
    )
    assert result.termination == "cycle_limit"
    assert result.total_cycles == 50
    assert result.mispredicts >= 1


def test_parser_accepts_comments_and_hex():
    program = parse_program(
        """
        # leading comment
        ADDI x1, x0, 0x10   ; trailing comment
        start: ADD x2, x1, x1
        BEQ x1, x1, start   # backward branch
        HALT
        """
    )
    assert [i.mnemonic for i in program] == ["ADDI", "ADD", "BEQ", "HALT"]
    assert program[0].imm == 16
    assert program[2].target_pc == 4
    assert program[2].imm == 4 - 8  # pc-relative displacement


def test_parser_rejects_bad_input_with_line_numbers():
    with pytest.raises(PipelineParseError) as exc:
        parse_program("NOP\nFOO x1, x2, x3")
    assert exc.value.line == 2

    with pytest.raises(PipelineParseError):
        parse_program("ADD x1, x99, x2")
    with pytest.raises(PipelineParseError):
        parse_program("BEQ x1, x2, nowhere")
    with pytest.raises(PipelineParseError):
        parse_program("dup: NOP\ndup: HALT")
    with pytest.raises(PipelineParseError):
        parse_program("LW x1, x2")  # missing memory operand syntax


def test_trailing_label_points_at_implicit_halt():
    program = parse_program("ADDI x1, x0, 1\nend:")
    assert program[-1].is_halt


def test_initial_registers_are_used():
    result = simulate_pipeline(
        "ADD x1, x2, x0",
        PipelineConfig(),
        initial_registers={2: 21},
    )
    assert result.final_registers[1] == 21


def test_config_validation_rejects_bad_values():
    with pytest.raises(ValueError):
        simulate_pipeline("HALT", PipelineConfig(memory_latency=0))
    with pytest.raises(ValueError):
        simulate_pipeline("HALT", PipelineConfig(branch_policy="psychic"))
    with pytest.raises(ValueError):
        simulate_pipeline("HALT", PipelineConfig(max_cycles=5))


def test_dynamic_branch_policy_reduces_flushes_on_loop():
    loop = """
    ADDI x1, x0, 3
    loop: ADDI x1, x1, -1
    BEQ x1, x0, done
    BEQ x0, x0, loop
    done: HALT
    """
    static = simulate_pipeline(loop, PipelineConfig(branch_policy="static_not_taken"))
    two_bit = simulate_pipeline(loop, PipelineConfig(branch_policy="two_bit"))
    # Both must finish; the dynamic predictor should not mispredict more.
    assert static.termination == "completed"
    assert two_bit.termination == "completed"
    assert two_bit.mispredicts <= static.mispredicts


def test_memory_events_record_latency_and_addresses():
    result = simulate_pipeline(
        "ADDI x1, x0, 4\nSW x1, 4(x0)\nLW x2, 4(x0)\nHALT",
        PipelineConfig(memory_latency=2),
    )
    ops = [(e["op"], e["address"], e["latency"]) for e in result.memory_events]
    assert ops == [("SW", 4, 2), ("LW", 4, 2)]
    assert result.mem_stall_cycles > 0  # latency 2 freezes the front end
    assert result.final_registers[2] == 4
