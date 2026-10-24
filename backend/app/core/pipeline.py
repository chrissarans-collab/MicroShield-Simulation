"""Five-stage in-order pipeline simulator — MicroShield Stage 2.

CANONICAL MODEL
===============
Mirrored 1:1 in TypeScript at ``frontend/src/lib/engines/pipeline.ts`` and
validated against ``fixtures/pipeline.json`` (hand-computed cases).

Pipeline: IF -> ID -> EX -> MEM -> WB, single issue, in order, one
instruction per stage per cycle.

TOY INSTRUCTION SET (16 registers x0..x15, x0 hardwired to 0, byte addresses,
4 bytes per instruction, values wrap as signed 32-bit):

    ADD  rd, rs1, rs2     register-register ALU
    SUB  rd, rs1, rs2
    AND  rd, rs1, rs2
    OR   rd, rs1, rs2
    XOR  rd, rs1, rs2
    SLT  rd, rs1, rs2     rd = 1 if rs1 < rs2 (signed) else 0
    ADDI rd, rs1, imm     immediate ALU
    LW   rd, imm(rs1)     word load   (occupies MEM)
    SW   rd, imm(rs1)     word store  (rd holds the data register)
    BEQ  rs1, rs2, label  branch if equal
    BNE  rs1, rs2, label  branch if not equal
    NOP                   alias for ADDI x0, x0, 0
    HALT                  fetched once, drains the pipeline, then fetch stops

Labels: ``name:`` on its own line or before an instruction.

HAZARD MODEL (documented; see docs/MODEL_SPEC.md)
-------------------------------------------------
Register file: WB writes in the FIRST half of the cycle, ID reads in the
SECOND half (write-first / read-late). Operand *values* are latched in ID;
forwarding (when enabled) overrides them at the start of EX:

  * MEM->EX forward: producer one ahead supplies its ALU result / load.
  * WB->EX forward:  producer two ahead (or a load after its stall) supplies
    its value from the WB latch.

Stall decision re-evaluated every cycle the consumer sits in ID:

  forwarding ON : load-use (LW directly feeding the next instruction) costs
                  1 stall; every ALU dependency costs 0.
  forwarding OFF: producer 1 ahead -> waits 2 cycles; producer 2 ahead ->
                  waits 1 cycle (producer must be *in WB* during the
                  consumer's ID cycle thanks to write-first/read-late).

WAW and WAR hazards cannot occur (in-order single issue, one write each).

BRANCH MODEL (documented)
-------------------------
* Branches decode in ID; the PC-relative target is known there.
* A prediction is looked up in ID. Predicting taken redirects fetch at the
  end of that cycle, so the sequentially fetched instruction becomes
  wrong-path if the branch is taken (1 flush even when the guess is right).
* Branches resolve at the END of EX. On any misprediction every younger
  instruction (the ID latch and the instruction fetched this cycle) is
  flushed and fetch restarts at the correct target: a 2-instruction flush.
* Policies: static_not_taken (default), static_taken, or a dynamic predictor
  (one_bit / two_bit / ghr) reusing the Stage-1 predictor module, trained
  online when the branch resolves in EX.

CYCLE-COUNT CONVENTION
----------------------
total_cycles counts every cycle from the first IF until the last instruction
completes WB (fill and drain included). Stalls and flushes are inside that
count and are also reported separately:

    CPI = total_cycles / retired      (None when retired == 0)

``memory_latency`` models a blocking cache: while a load/store occupies MEM
the front end is frozen; those cycles count as ``mem_stall_cycles``. A
``memory_access`` hook (op, address) -> (latency, hit) lets the cache model
drive per-access latency instead of one fixed value (used by the
Architecture Comparison lab).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from app.core.branch_predictors import (
    AlwaysNotTaken,
    AlwaysTaken,
    BasePredictor,
    BranchConfig,
    GsharePredictor,
    OneBitPredictor,
    TwoBitPredictor,
)

BRANCH_POLICIES = (
    "static_not_taken",
    "static_taken",
    "one_bit",
    "two_bit",
    "ghr",
)

MAX_PROGRAM_LENGTH = 400
DEFAULT_MAX_CYCLES = 5000
NUM_REGISTERS = 16

ALU_OPS = {"ADD", "SUB", "AND", "OR", "XOR", "SLT"}

# memory_access hook: (op, address) -> (latency_cycles, hit_or_None)
MemoryAccessHook = Callable[[str, int], Tuple[int, Optional[bool]]]


class PipelineParseError(ValueError):
    """Invalid toy-assembly input, carrying the offending line number."""

    def __init__(self, line: int, message: str) -> None:
        super().__init__(f"line {line}: {message}")
        self.line = line
        self.message = message


def _wrap32(value: int) -> int:
    return ((value + 2**31) % 2**32) - 2**31


def _normalize_reg(token: str) -> str:
    stripped = token.strip()
    if stripped[:1].lower() == "x" and stripped[1:].isdigit():
        return f"x{int(stripped[1:])}"
    return stripped


# --------------------------------------------------------------------------- #
# Instruction representation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Instruction:
    index: int
    pc: int
    mnemonic: str
    text: str
    rd: Optional[int] = None
    rs1: Optional[int] = None
    rs2: Optional[int] = None
    imm: int = 0
    target_pc: Optional[int] = None
    is_branch: bool = False
    is_load: bool = False
    is_store: bool = False
    is_halt: bool = False

    @property
    def writes_register(self) -> bool:
        return self.rd is not None and self.rd != 0

    @property
    def sources(self) -> List[int]:
        return [r for r in (self.rs1, self.rs2) if r is not None]

    @property
    def uses_memory(self) -> bool:
        return self.is_load or self.is_store

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "pc": self.pc,
            "mnemonic": self.mnemonic,
            "text": self.text,
            "rd": self.rd,
            "rs1": self.rs1,
            "rs2": self.rs2,
            "imm": self.imm,
            "target_pc": self.target_pc,
            "is_branch": self.is_branch,
            "is_load": self.is_load,
            "is_store": self.is_store,
            "is_halt": self.is_halt,
        }


@dataclass
class PipelineConfig:
    forwarding: bool = True
    branch_policy: str = "static_not_taken"
    memory_latency: int = 1
    # dynamic-predictor options (branch_policy one_bit / two_bit / ghr)
    initial_prediction: str = "taken"
    initial_counter: int = 2
    history_bits: int = 4
    initial_ghr: int = 0
    max_cycles: int = DEFAULT_MAX_CYCLES


def validate_pipeline_config(config: PipelineConfig) -> List[str]:
    errors: List[str] = []
    if config.branch_policy not in BRANCH_POLICIES:
        errors.append(
            f"Unknown branch policy '{config.branch_policy}'. "
            f"Choose one of: {', '.join(BRANCH_POLICIES)}."
        )
    if config.initial_prediction not in ("taken", "not_taken"):
        errors.append("initial_prediction must be 'taken' or 'not_taken'.")
    if not (0 <= config.initial_counter <= 3):
        errors.append("initial_counter must be between 0 and 3.")
    if not (1 <= config.history_bits <= 8):
        errors.append("history_bits must be between 1 and 8.")
    if config.initial_ghr < 0:
        errors.append("initial_ghr must be >= 0.")
    if not (1 <= config.memory_latency <= 8):
        errors.append("memory_latency must be between 1 and 8 cycles.")
    if not (10 <= config.max_cycles <= 100_000):
        errors.append("max_cycles must be between 10 and 100000.")
    return errors


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def _parse_register(token: str) -> int:
    low = token.strip().lower()
    if not low.startswith("x") or not low[1:].isdigit():
        raise PipelineParseError(0, f"bad register '{token}' (expected x0..x15)")
    number = int(low[1:])
    if not (0 <= number < NUM_REGISTERS):
        raise PipelineParseError(0, f"register '{token}' out of range (x0..x15)")
    return number


def _parse_imm(token: str) -> int:
    token = token.strip()
    try:
        value = int(token, 0)
    except ValueError:
        raise PipelineParseError(0, f"bad immediate '{token}'") from None
    if not (-999999 <= value <= 999999):
        raise PipelineParseError(0, f"immediate '{token}' out of range")
    return value


def _parse_mem_operand(token: str) -> Tuple[int, int]:
    """Parse ``8(x2)`` -> (offset, base_register)."""
    token = token.strip()
    if "(" not in token or not token.endswith(")"):
        raise PipelineParseError(0, f"bad memory operand '{token}' (expected imm(reg))")
    offset_part, _, reg_part = token.partition("(")
    offset = _parse_imm(offset_part) if offset_part.strip() else 0
    return offset, _parse_register(reg_part[:-1])


def _canonical_text(mnemonic: str, args: Sequence[str]) -> str:
    regs = [_normalize_reg(a) if a[:1].lower() == "x" and a[1:].isdigit() else a for a in args]
    if not args:
        return mnemonic
    if mnemonic in ("LW", "SW"):
        return f"{mnemonic} {regs[0]}, {''.join(args[1:])}"
    if mnemonic in ("BEQ", "BNE"):
        return f"{mnemonic} {regs[0]}, {regs[1]}, {args[2]}"
    if mnemonic == "ADDI":
        return f"{mnemonic} {regs[0]}, {regs[1]}, {args[2]}"
    if mnemonic in ALU_OPS:
        return f"{mnemonic} {regs[0]}, {regs[1]}, {regs[2]}"
    return mnemonic


def parse_program(source: str) -> List[Instruction]:
    """Parse toy assembly into instructions.

    Raises PipelineParseError (with line numbers) for invalid input.
    """
    # Pass 1: strip comments, buffer labels, collect statement bodies.
    entries: List[Tuple[int, str]] = []  # (line_no, body)
    label_targets: Dict[str, int] = {}  # label -> entry position
    label_lines: Dict[str, int] = {}
    pending: List[Tuple[int, str]] = []  # labels waiting for a statement

    for line_no, raw_line in enumerate(source.splitlines(), start=1):
        line = raw_line.split("#")[0].split(";")[0].strip()
        if not line:
            continue
        # one or more "name:" prefixes may precede the statement
        collected = list(pending)
        pending = []
        while True:
            head, sep, rest = line.partition(":")
            if not sep:
                break
            name = head.strip()
            if not name or name[0].isdigit() or not all(c.isalnum() or c == "_" for c in name):
                raise PipelineParseError(line_no, f"invalid label '{name}'")
            collected.append((line_no, name))
            line = rest.strip()
        if not line:
            # label with no statement yet — attach to the NEXT statement
            pending = collected
            continue
        position = len(entries)
        for ln, name in collected:
            if name in label_targets:
                raise PipelineParseError(ln, f"duplicate label '{name}'")
            label_targets[name] = position
            label_lines[name] = ln
        entries.append((line_no, line))

    # Trailing labels point at an implicit HALT after the last statement.
    if pending:
        position = len(entries)
        for ln, name in pending:
            if name in label_targets:
                raise PipelineParseError(ln, f"duplicate label '{name}'")
            label_targets[name] = position
            label_lines[name] = ln
        entries.append((pending[-1][0], "HALT"))

    if len(entries) > MAX_PROGRAM_LENGTH:
        raise PipelineParseError(1, f"program longer than {MAX_PROGRAM_LENGTH} instructions")

    # Pass 2: parse each statement.
    instructions: List[Instruction] = []
    for position, (line_no, body) in enumerate(entries):
        try:
            instructions.append(_parse_statement(position, body, label_targets, len(entries)))
        except PipelineParseError as exc:
            if exc.line == 0:
                raise PipelineParseError(line_no, exc.message) from None
            raise
    return instructions


def _parse_statement(
    position: int,
    body: str,
    label_targets: Dict[str, int],
    size: int,
) -> Instruction:
    tokens = body.replace(",", " ").split()
    mnemonic = tokens[0].upper()
    args = tokens[1:]
    pc = position * 4

    def target_pc_of(label: str) -> int:
        if label not in label_targets:
            raise PipelineParseError(0, f"unknown label '{label}'")
        target = label_targets[label]
        if target >= size:
            raise PipelineParseError(0, f"label '{label}' has no instruction")
        return target * 4

    def finish(**kwargs: object) -> Instruction:
        base: Dict[str, object] = {
            "index": position,
            "pc": pc,
            "mnemonic": mnemonic,
            "text": _canonical_text(mnemonic, args),
        }
        base.update(kwargs)
        return Instruction(**base)  # type: ignore[arg-type]

    if mnemonic == "NOP":
        return finish(mnemonic="NOP", text="NOP", rd=0, rs1=0, rs2=0, imm=0)
    if mnemonic == "HALT":
        if args:
            raise PipelineParseError(0, "HALT takes no operands")
        return finish(text="HALT", is_halt=True)
    if mnemonic in ALU_OPS:
        if len(args) != 3:
            raise PipelineParseError(0, f"{mnemonic} expects rd, rs1, rs2")
        rd, rs1, rs2 = (_parse_register(a) for a in args)
        return finish(rd=rd, rs1=rs1, rs2=rs2)
    if mnemonic == "ADDI":
        if len(args) != 3:
            raise PipelineParseError(0, "ADDI expects rd, rs1, imm")
        return finish(
            rd=_parse_register(args[0]),
            rs1=_parse_register(args[1]),
            imm=_parse_imm(args[2]),
        )
    if mnemonic in ("LW", "SW"):
        if len(args) != 2:
            raise PipelineParseError(0, f"{mnemonic} expects rd, imm(reg)")
        first = _parse_register(args[0])
        offset, base = _parse_mem_operand(args[1])
        if mnemonic == "LW":
            return finish(rd=first, rs1=base, imm=offset, is_load=True)
        return finish(rd=None, rs1=base, rs2=first, imm=offset, is_store=True)
    if mnemonic in ("BEQ", "BNE"):
        if len(args) != 3:
            raise PipelineParseError(0, f"{mnemonic} expects rs1, rs2, label")
        rs1, rs2 = _parse_register(args[0]), _parse_register(args[1])
        target_pc = target_pc_of(args[2])
        return finish(rs1=rs1, rs2=rs2, imm=target_pc - pc, target_pc=target_pc, is_branch=True)
    raise PipelineParseError(0, f"unknown instruction '{tokens[0]}'")


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #


@dataclass
class _Latch:
    """An instruction occupying a pipeline stage for one or more cycles."""

    inst: Instruction
    seq: int  # dynamic instance id, assigned when latched into IF/ID
    operands: Dict[int, int] = field(default_factory=dict)  # values read in ID
    result: int = 0  # ALU result / loaded value
    store_value: int = 0
    address: int = 0
    mem_left: int = 1
    latency: int = 1
    hit: Optional[bool] = None
    predicted: Optional[bool] = None


@dataclass
class PipelineResult:
    instructions: List[Instruction]
    total_cycles: int
    retired: int
    cpi: Optional[float]
    hazard_stall_cycles: int
    mem_stall_cycles: int
    stall_cycles: int
    flush_cycles: int
    branches: int
    mispredicts: int
    forwarded_operands: int
    trace: List[Dict[str, object]]
    branch_events: List[Dict[str, object]]
    memory_events: List[Dict[str, object]]
    retired_instructions: List[Dict[str, object]]
    final_registers: List[int]
    memory_writes: List[Dict[str, int]]
    termination: str
    notes: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "instructions": [i.to_dict() for i in self.instructions],
            "total_cycles": self.total_cycles,
            "retired": self.retired,
            "cpi": round(self.cpi, 6) if self.cpi is not None else None,
            "hazard_stall_cycles": self.hazard_stall_cycles,
            "mem_stall_cycles": self.mem_stall_cycles,
            "stall_cycles": self.stall_cycles,
            "flush_cycles": self.flush_cycles,
            "branches": self.branches,
            "mispredicts": self.mispredicts,
            "forwarded_operands": self.forwarded_operands,
            "trace": self.trace,
            "branch_events": self.branch_events,
            "memory_events": self.memory_events,
            "retired_instructions": self.retired_instructions,
            "final_registers": self.final_registers,
            "memory_writes": self.memory_writes,
            "termination": self.termination,
            "notes": self.notes,
        }


def _bubble() -> Dict[str, object]:
    return {"kind": "bubble", "seq": None, "text": "", "state": "bubble", "note": ""}


def _cell(latch: _Latch, state: str = "ok", note: str = "") -> Dict[str, object]:
    return {
        "kind": "instr",
        "seq": latch.seq,
        "text": latch.inst.text,
        "state": state,
        "note": note,
    }


def _make_branch_predictor(config: PipelineConfig) -> Optional[BasePredictor]:
    if config.branch_policy == "static_not_taken":
        return AlwaysNotTaken()
    if config.branch_policy == "static_taken":
        return AlwaysTaken()
    if config.branch_policy == "one_bit":
        return OneBitPredictor(config.initial_prediction)
    if config.branch_policy == "two_bit":
        return TwoBitPredictor(config.initial_counter)
    if config.branch_policy == "ghr":
        return GsharePredictor(config.history_bits, config.initial_counter, config.initial_ghr)
    return None


def _model_notes(config: PipelineConfig) -> List[str]:
    notes = [
        "Cycle count includes pipeline fill and drain.",
        "Branches resolve at the end of EX; a misprediction flushes every "
        "younger instruction (ID latch + current fetch = up to 2 flushed).",
        "Register file: WB writes first half of the cycle, ID reads second half.",
    ]
    if config.branch_policy == "static_not_taken":
        notes.append("Branch policy: static predict-not-taken (no prediction state).")
    elif config.branch_policy == "static_taken":
        notes.append("Branch policy: static predict-taken (no prediction state).")
    else:
        notes.append(f"Branch policy: dynamic '{config.branch_policy}' predictor trained in EX.")
    notes.append(
        "Forwarding enabled: only load-use hazards cost 1 stall cycle."
        if config.forwarding
        else "Forwarding disabled: a distance-1 dependency costs 2 stall cycles, distance-2 costs 1."
    )
    return notes


def simulate_pipeline(
    source: str,
    config: Optional[PipelineConfig] = None,
    initial_registers: Optional[Dict[int, int]] = None,
    memory_access: Optional[MemoryAccessHook] = None,
) -> PipelineResult:
    """Run the toy CPU to completion and return every observable."""
    config = config or PipelineConfig()
    errors = validate_pipeline_config(config)
    if errors:
        raise ValueError("; ".join(errors))

    program = parse_program(source)
    notes = _model_notes(config)

    if not program:
        return PipelineResult(
            instructions=[],
            total_cycles=0,
            retired=0,
            cpi=None,
            hazard_stall_cycles=0,
            mem_stall_cycles=0,
            stall_cycles=0,
            flush_cycles=0,
            branches=0,
            mispredicts=0,
            forwarded_operands=0,
            trace=[],
            branch_events=[],
            memory_events=[],
            retired_instructions=[],
            final_registers=[0] * NUM_REGISTERS,
            memory_writes=[],
            termination="completed",
            notes=notes,
        )

    regs = [0] * NUM_REGISTERS
    if initial_registers:
        for raw_index, raw_value in initial_registers.items():
            index = int(raw_index)
            if 0 < index < NUM_REGISTERS:  # x0 is hardwired to zero
                regs[index] = _wrap32(int(raw_value))
    memory: Dict[int, int] = {}
    memory_writes: List[Dict[str, int]] = []
    predictor = _make_branch_predictor(config)

    program_end = len(program) * 4
    pc = 0
    fetch_enabled = True
    if_id: Optional[_Latch] = None
    id_ex: Optional[_Latch] = None
    ex_mem: Optional[_Latch] = None
    mem_wb: Optional[_Latch] = None

    cycle = 0
    retired = hazard_stalls = mem_stalls = 0
    flushes = branches = mispredicts = forwards = 0
    next_seq = 0
    trace: List[Dict[str, object]] = []
    branch_events: List[Dict[str, object]] = []
    memory_events: List[Dict[str, object]] = []
    retired_log: List[Dict[str, object]] = []
    termination = "completed"

    while cycle < config.max_cycles:
        # ---- IF (fetch happens first: it uses the PC latched last cycle) ----
        fetched: Optional[Instruction] = None
        if fetch_enabled and 0 <= pc < program_end:
            fetched = program[pc // 4]

        cycle += 1
        annotations: List[str] = []
        cells: Dict[str, Dict[str, object]] = {
            "IF": (
                {"kind": "instr", "seq": None, "text": fetched.text, "state": "ok", "note": ""}
                if fetched is not None
                else _bubble()
            ),
            "ID": _cell(if_id) if if_id else _bubble(),
            "EX": _cell(id_ex) if id_ex else _bubble(),
            "MEM": _cell(ex_mem) if ex_mem else _bubble(),
            "WB": _cell(mem_wb) if mem_wb else _bubble(),
        }

        # next-state registers for the end of this cycle
        mem_wb_next: Optional[_Latch] = None
        ex_mem_next: Optional[_Latch] = None
        id_ex_next: Optional[_Latch] = None
        if_id_next: Optional[_Latch] = None
        redirect: Optional[int] = None
        fetch_enabled_next = fetch_enabled

        # ---- 1. WB: always completes; register write happens first half ----
        if mem_wb is not None:
            if mem_wb.inst.writes_register:
                regs[mem_wb.inst.rd] = _wrap32(mem_wb.result)
            retired += 1
            retired_log.append(
                {"seq": mem_wb.seq, "pc": mem_wb.inst.pc, "text": mem_wb.inst.text, "cycle": cycle}
            )
            cells["WB"] = _cell(mem_wb, "ok", "retired")

        # ---- 2. MEM: the stage is occupied for `latency` cycles ----
        frozen = False
        mem_done = False
        if ex_mem is not None:
            ex_mem.mem_left -= 1
            if ex_mem.mem_left > 0:
                frozen = True
                mem_stalls += 1
                cells["MEM"] = _cell(ex_mem, "mem_stall", "memory access in progress")
                for stage in ("EX", "ID"):
                    occupant = id_ex if stage == "EX" else if_id
                    cells[stage] = (
                        _cell(occupant, "stall", "front end frozen by MEM") if occupant else _bubble()
                    )
                cells["IF"] = (
                    {
                        "kind": "instr",
                        "seq": None,
                        "text": fetched.text,
                        "state": "stall",
                        "note": "refetch after memory freeze",
                    }
                    if fetched is not None
                    else _bubble()
                )
                annotations.append(
                    f"MEM busy: {ex_mem.inst.text} waits {ex_mem.latency} cycles for memory "
                    "(blocking) — front end frozen"
                )
            else:
                mem_done = True
                inst = ex_mem.inst
                if inst.is_load:
                    ex_mem.result = _wrap32(memory.get(ex_mem.address, 0))
                elif inst.is_store:
                    value = _wrap32(ex_mem.store_value)
                    memory[ex_mem.address] = value
                    memory_writes.append(
                        {"address": ex_mem.address, "value": value, "cycle": cycle}
                    )
                if inst.uses_memory:
                    memory_events.append(
                        {
                            "seq": ex_mem.seq,
                            "op": "LW" if inst.is_load else "SW",
                            "address": ex_mem.address,
                            "value": ex_mem.result if inst.is_load else _wrap32(ex_mem.store_value),
                            "latency": ex_mem.latency,
                            "hit": ex_mem.hit,
                            "cycle": cycle,
                        }
                    )
                mem_wb_next = ex_mem

        if frozen:
            # id_ex, if_id and pc all hold; the consumed WB slot stays empty.
            ex_mem_next = ex_mem
            id_ex_next = id_ex
            if_id_next = if_id
            trace.append(_trace_row(cycle, cells, annotations))
            mem_wb, ex_mem, id_ex, if_id = mem_wb_next, ex_mem_next, id_ex_next, if_id_next
            continue

        # ---- 3. EX: executes when the MEM slot is free; branches resolve ----
        if id_ex is not None:
            inst = id_ex.inst
            if config.forwarding:
                for reg, value in _forwarded_operands(id_ex, ex_mem, mem_wb).items():
                    if id_ex.operands.get(reg) != value:
                        forwards += 1
                    id_ex.operands[reg] = value

            a = id_ex.operands.get(inst.rs1, 0) if inst.rs1 is not None else 0
            b = id_ex.operands.get(inst.rs2, 0) if inst.rs2 is not None else 0

            if inst.mnemonic == "ADD":
                id_ex.result = _wrap32(a + b)
            elif inst.mnemonic == "SUB":
                id_ex.result = _wrap32(a - b)
            elif inst.mnemonic == "AND":
                id_ex.result = _wrap32(a & b)
            elif inst.mnemonic == "OR":
                id_ex.result = _wrap32(a | b)
            elif inst.mnemonic == "XOR":
                id_ex.result = _wrap32(a ^ b)
            elif inst.mnemonic == "SLT":
                id_ex.result = 1 if a < b else 0
            elif inst.mnemonic == "ADDI":
                id_ex.result = _wrap32(a + inst.imm)
            elif inst.uses_memory:
                id_ex.address = _wrap32(a + inst.imm)
                if inst.is_store:
                    # data register value; forwarded like any other operand
                    id_ex.store_value = b
                id_ex.latency = config.memory_latency
                id_ex.hit = None
                if memory_access is not None and inst.uses_memory:
                    latency, hit = memory_access("LW" if inst.is_load else "SW", id_ex.address)
                    id_ex.latency = max(1, int(latency))
                    id_ex.hit = hit
                id_ex.result = id_ex.address
                id_ex.mem_left = id_ex.latency
            elif inst.is_branch:
                branches += 1
                actual = (a == b) if inst.mnemonic == "BEQ" else (a != b)
                predicted = id_ex.predicted if id_ex.predicted is not None else False
                correct = predicted == actual
                flush_count = 0

                if not correct:
                    mispredicts += 1
                    if if_id is not None:
                        cells["ID"] = _cell(if_id, "flush", "flushed: branch misprediction")
                        flush_count += 1
                        if_id = None
                    if fetched is not None:
                        cells["IF"] = {
                            "kind": "instr",
                            "seq": None,
                            "text": fetched.text,
                            "state": "flush",
                            "note": "flushed: branch misprediction",
                        }
                        flush_count += 1
                        fetched = None
                    redirect = inst.target_pc if actual else inst.pc + 4
                    fetch_enabled_next = True
                    annotations.append(
                        f"MISPREDICT {inst.text} → actually {'taken' if actual else 'not taken'}, "
                        f"predicted {'taken' if predicted else 'not taken'}: "
                        f"flushed {flush_count}, redirect to 0x{redirect:x}"
                    )
                elif actual and if_id is not None:
                    # correct "taken" prediction: ID redirected fetch earlier,
                    # so the sequential instruction sitting in ID is wrong-path.
                    cells["ID"] = _cell(if_id, "flush", "flushed: wrong-path after taken branch")
                    flush_count += 1
                    if_id = None
                    annotations.append(
                        f"{inst.text} taken (predicted correctly): flushed 1 wrong-path instruction"
                    )

                flushes += flush_count
                if config.branch_policy not in ("static_not_taken", "static_taken"):
                    if predictor is not None:
                        predictor.update(f"B@{inst.pc}", actual)
                branch_events.append(
                    {
                        "seq": id_ex.seq,
                        "pc": inst.pc,
                        "text": inst.text,
                        "predicted": predicted,
                        "actual": actual,
                        "correct": correct,
                        "flushes": flush_count,
                        "cycle": cycle,
                    }
                )
                cells["EX"] = _cell(id_ex, "ok", f"resolve {'taken' if actual else 'not taken'}")

            ex_mem_next = id_ex
            id_ex = None

        # ---- 4. ID: hazard check, operand read, branch prediction ----
        id_freed = True
        if if_id is not None:
            if _needs_stall(if_id, ex_mem_next, ex_mem, mem_wb, config):
                hazard_stalls += 1
                producer = _producer_text(if_id, ex_mem_next, ex_mem, mem_wb)
                source = _first_source(if_id)
                cells["ID"] = _cell(
                    if_id, "stall", f"RAW hazard: waits for {producer} (reads x{source})"
                )
                annotations.append(
                    f"RAW hazard: {if_id.inst.text} waits for {producer} — stall, bubble into EX"
                )
                id_freed = False
                if_id_next = if_id
            else:
                for reg in if_id.inst.sources:
                    if_id.operands[reg] = regs[reg]
                if if_id.inst.is_branch:
                    predicted = False
                    if config.branch_policy == "static_taken":
                        predicted = True
                    elif config.branch_policy == "static_not_taken":
                        predicted = False
                    elif predictor is not None:
                        predicted = predictor.predict(f"B@{if_id.inst.pc}")
                    if_id.predicted = predicted
                    if predicted:
                        redirect = if_id.inst.target_pc
                        fetch_enabled_next = True
                        annotations.append(
                            f"{if_id.inst.text} predicted TAKEN in ID — "
                            f"fetch redirected to 0x{redirect:x}"
                        )
                id_ex_next = if_id
                if_id = None

        # ---- 5. IF: latch the fetched instruction if the ID slot is free ----
        latched = False
        if if_id is None and id_freed and fetched is not None:
            if fetched.is_halt:
                fetch_enabled_next = False
            if_id_next = _Latch(inst=fetched, seq=next_seq, mem_left=1, latency=1)
            next_seq += 1
            latched = True
        elif not id_freed and fetched is not None:
            cells["IF"] = {
                "kind": "instr",
                "seq": None,
                "text": fetched.text,
                "state": "stall",
                "note": "ID stalled — fetch not latched",
            }

        # Commit next state.
        # PC rules: a redirect always wins (and re-enables fetch, in case a
        # wrong-path HALT had disabled it); otherwise the PC only advances
        # when this cycle's fetch was actually latched — a stalled ID holds
        # the PC so the same instruction is fetched again next cycle.
        if redirect is not None:
            pc = redirect
            fetch_enabled = True
        else:
            if latched:
                pc += 4
            fetch_enabled = fetch_enabled_next
        mem_wb, ex_mem, id_ex, if_id = mem_wb_next, ex_mem_next, id_ex_next, if_id_next

        trace.append(_trace_row(cycle, cells, annotations))

        # ---- termination ----
        if (
            if_id is None
            and id_ex is None
            and ex_mem is None
            and mem_wb is None
            and (not fetch_enabled or pc >= program_end)
        ):
            break
    else:
        termination = "cycle_limit"

    cpi = (cycle / retired) if retired else None
    return PipelineResult(
        instructions=program,
        total_cycles=cycle,
        retired=retired,
        cpi=cpi,
        hazard_stall_cycles=hazard_stalls,
        mem_stall_cycles=mem_stalls,
        stall_cycles=hazard_stalls + mem_stalls,
        flush_cycles=flushes,
        branches=branches,
        mispredicts=mispredicts,
        forwarded_operands=forwards,
        trace=trace,
        branch_events=branch_events,
        memory_events=memory_events,
        retired_instructions=retired_log,
        final_registers=regs,
        memory_writes=memory_writes,
        termination=termination,
        notes=notes,
    )


def _trace_row(
    cycle: int, cells: Dict[str, Dict[str, object]], annotations: List[str]
) -> Dict[str, object]:
    return {
        "cycle": cycle,
        "IF": cells["IF"],
        "ID": cells["ID"],
        "EX": cells["EX"],
        "MEM": cells["MEM"],
        "WB": cells["WB"],
        "annotations": annotations,
    }


def _forwarded_operands(
    consumer: _Latch, ex_mem: Optional[_Latch], mem_wb: Optional[_Latch]
) -> Dict[int, int]:
    """Forwarding search at the start of the consumer's EX cycle.

    Priority: MEM-stage producer (distance 1) then the WB latch (distance 2,
    or a load that just took its stall). Only registers the consumer reads
    are forwarded. A not-yet-completed load can never be here: the front end
    is frozen while MEM is busy, so the consumer cannot execute yet.
    """
    out: Dict[int, int] = {}
    sources = set(consumer.inst.sources)
    if ex_mem is not None and ex_mem.inst.writes_register and ex_mem.inst.rd in sources:
        out[ex_mem.inst.rd] = ex_mem.result
    if mem_wb is not None and mem_wb.inst.writes_register and mem_wb.inst.rd in sources:
        out[mem_wb.inst.rd] = mem_wb.result
    return out


def _needs_stall(
    consumer: _Latch,
    producer_dist1: Optional[_Latch],  # executed EX this cycle (in MEM next)
    producer_dist2: Optional[_Latch],  # in MEM this cycle (in WB next)
    producer_dist3: Optional[_Latch],  # in WB this cycle (commits now)
    config: PipelineConfig,
) -> bool:
    """True when `consumer` (sitting in ID) must wait another cycle.

    Re-evaluated every stalled cycle, so the loop naturally produces the
    textbook stall counts: 2/1 without forwarding, 1 for load-use with it.
    """

    def depends(producer: Optional[_Latch]) -> bool:
        return (
            producer is not None
            and producer.inst.writes_register
            and producer.inst.rd in consumer.inst.sources
        )

    if depends(producer_dist1):
        if not config.forwarding:
            return True  # must wait until producer reaches WB (2 cycles total)
        return producer_dist1.inst.is_load  # load-use: value exists only after MEM
    if depends(producer_dist2):
        return not config.forwarding  # wait one more cycle (producer in WB next)
    # dist3 commits this cycle: write-first / read-late makes it safe.
    return False


def _producer_text(
    consumer: _Latch,
    producer_dist1: Optional[_Latch],
    producer_dist2: Optional[_Latch],
    producer_dist3: Optional[_Latch],
) -> str:
    for producer in (producer_dist1, producer_dist2, producer_dist3):
        if producer is None:
            continue
        if producer.inst.writes_register and producer.inst.rd in consumer.inst.sources:
            return producer.inst.text
    return "an older instruction"


def _first_source(latch: _Latch) -> int:
    sources = latch.inst.sources
    return sources[0] if sources else 0
