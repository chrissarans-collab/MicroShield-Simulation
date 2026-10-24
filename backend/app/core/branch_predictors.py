"""Branch predictor models — MicroShield Stage 1.

CANONICAL MODEL
===============
This module is the reference implementation. It is mirrored 1:1 in TypeScript
at ``frontend/src/lib/engines/branch.ts``; both implementations are validated
against the shared fixtures in ``fixtures/branch_predictor.json``.

Predictors
----------
``always_taken`` / ``always_not_taken``
    Stateless static predictors.
``one_bit``
    One predictor bit per branch label; predicts whatever the last observed
    outcome for that label was (labels start at the configured initial state).
``two_bit``
    2-bit saturating counter per label: 0 = Strongly Not Taken (SN),
    1 = Weakly Not Taken (WN), 2 = Weakly Taken (WT), 3 = Strongly Taken (ST).
    Predicts Taken when the counter >= 2. Counters saturate at 0 and 3.
``ghr``
    gshare-style predictor: a table of 2-bit counters indexed by
    ``(fnv1a32(label) % tableSize) XOR globalHistory`` where
    ``tableSize = 2 ** historyBits``. The global history register stores the
    last ``historyBits`` outcomes (most recent in the low bit).

TRAINING POLICY (documented, applies to every predictor)
--------------------------------------------------------
Updates are *online*: a predictor's state changes only after the actual
outcome of the branch is revealed — exactly like a real CPU updating its
predictor when a branch resolves. Prediction happens strictly before the
update. Unseen labels begin in the configured initial state.

LIMITATION
----------
These are textbook educational predictors. Commercial CPUs use much larger,
tagged, skewed and hybrid predictors; do not expect these numbers to match
any real processor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

PREDICTORS = (
    "always_taken",
    "always_not_taken",
    "one_bit",
    "two_bit",
    "ghr",
)

MAX_SEQUENCE_LENGTH = 500

_COUNTER_NAMES = {0: "SN", 1: "WN", 2: "WT", 3: "ST"}


def fnv1a32(text: str) -> int:
    """FNV-1a 32-bit hash — used to map branch labels to table indices.

    Implemented identically in TypeScript (with Math.imul) so both engines
    index gshare tables the same way for the same labels.
    """
    h = 0x811C9DC5
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


@dataclass(frozen=True)
class BranchEvent:
    """One dynamic branch instance: a named branch with a known outcome."""

    label: str
    outcome: bool  # True = taken


@dataclass
class BranchConfig:
    """User-configurable assumptions for a prediction experiment."""

    predictor: str = "two_bit"
    # one_bit: state for labels that have never been seen.
    initial_prediction: str = "taken"  # "taken" | "not_taken"
    # two_bit / ghr: counter value every table entry starts from (0..3).
    initial_counter: int = 2
    # ghr: initial global history register value (masked to history_bits).
    initial_ghr: int = 0
    # ghr: history register width; table size = 2 ** history_bits.
    history_bits: int = 4
    # Cycles assumed lost per misprediction (displayed as an estimate).
    misprediction_penalty: int = 2


def validate_config(config: BranchConfig) -> List[str]:
    """Return a list of human-readable problems; empty list means valid."""
    errors: List[str] = []
    if config.predictor not in PREDICTORS:
        errors.append(
            f"Unknown predictor '{config.predictor}'. Choose one of: {', '.join(PREDICTORS)}."
        )
    if config.initial_prediction not in ("taken", "not_taken"):
        errors.append("initial_prediction must be 'taken' or 'not_taken'.")
    if not (0 <= config.initial_counter <= 3):
        errors.append("initial_counter must be between 0 and 3 (2-bit saturating counter).")
    if not (1 <= config.history_bits <= 8):
        errors.append("history_bits must be between 1 and 8.")
    if config.initial_ghr < 0:
        errors.append("initial_ghr must be >= 0.")
    if not (0 <= config.misprediction_penalty <= 20):
        errors.append("misprediction_penalty must be between 0 and 20 cycles.")
    return errors


# --------------------------------------------------------------------------- #
# Predictor implementations
# --------------------------------------------------------------------------- #


class BasePredictor:
    """Interface: predict before the outcome is known, then update."""

    name = "base"

    def predict(self, label: str) -> bool:
        raise NotImplementedError

    def update(self, label: str, outcome: bool) -> None:
        raise NotImplementedError

    def state_before(self, label: str) -> str:
        """Short human-readable state *before* predicting (for the trace)."""
        raise NotImplementedError

    def state_after(self, label: str) -> str:
        """Short human-readable state *after* the outcome is applied."""
        raise NotImplementedError

    def detail_after(self, label: str) -> Dict[str, object]:
        """Structured state after the update (consumed by the UI)."""
        return {}


class AlwaysTaken(BasePredictor):
    name = "always_taken"

    def predict(self, label: str) -> bool:
        return True

    def update(self, label: str, outcome: bool) -> None:
        return None

    def state_before(self, label: str) -> str:
        return "static"

    def state_after(self, label: str) -> str:
        return "static"


class AlwaysNotTaken(BasePredictor):
    name = "always_not_taken"

    def predict(self, label: str) -> bool:
        return False

    def update(self, label: str, outcome: bool) -> None:
        return None

    def state_before(self, label: str) -> str:
        return "static"

    def state_after(self, label: str) -> str:
        return "static"


class OneBitPredictor(BasePredictor):
    name = "one_bit"

    def __init__(self, initial_prediction: str) -> None:
        self._initial_taken = initial_prediction == "taken"
        self._initial_label = (
            "init(T)" if self._initial_taken else "init(N)"
        )
        self.table: Dict[str, bool] = {}

    def predict(self, label: str) -> bool:
        return self.table.get(label, self._initial_taken)

    def update(self, label: str, outcome: bool) -> None:
        self.table[label] = outcome

    def _state(self, label: str) -> str:
        if label not in self.table:
            return self._initial_label
        return "T" if self.table[label] else "N"

    def state_before(self, label: str) -> str:
        return self._state(label)

    def state_after(self, label: str) -> str:
        return self._state(label)

    def detail_after(self, label: str) -> Dict[str, object]:
        return {"last_outcome": "T" if self.table.get(label) else "N"}


class TwoBitPredictor(BasePredictor):
    name = "two_bit"

    def __init__(self, initial_counter: int) -> None:
        self._initial = initial_counter
        self.table: Dict[str, int] = {}

    def _get(self, label: str) -> int:
        return self.table.get(label, self._initial)

    def predict(self, label: str) -> bool:
        return self._get(label) >= 2

    def update(self, label: str, outcome: bool) -> None:
        counter = self._get(label)
        counter = min(3, counter + 1) if outcome else max(0, counter - 1)
        self.table[label] = counter

    def _state(self, label: str) -> str:
        if label not in self.table:
            return f"init({self._initial})"
        counter = self.table[label]
        return f"{counter} {_COUNTER_NAMES[counter]}"

    def state_before(self, label: str) -> str:
        return self._state(label)

    def state_after(self, label: str) -> str:
        return self._state(label)

    def detail_after(self, label: str) -> Dict[str, object]:
        counter = self._get(label)
        return {"counter": counter, "name": _COUNTER_NAMES[counter]}


class GsharePredictor(BasePredictor):
    name = "ghr"

    def __init__(self, history_bits: int, initial_counter: int, initial_ghr: int) -> None:
        self.history_bits = history_bits
        self.table_size = 1 << history_bits
        self._mask = self.table_size - 1
        self._initial = initial_counter
        self.ghr = initial_ghr & self._mask
        self.table: Dict[int, int] = {}
        # Index of the entry touched by the most recent update. Because the
        # history register shifts on update, the entry that was just modified
        # is not necessarily the entry _index(label) points at afterwards.
        self._last_index: Optional[int] = None

    def _index(self, label: str) -> int:
        # gshare: fold the branch's "PC" (label hash) with global history.
        return (fnv1a32(label) % self.table_size) ^ self.ghr

    def _get(self, index: int) -> int:
        return self.table.get(index, self._initial)

    def predict(self, label: str) -> bool:
        return self._get(self._index(label)) >= 2

    def update(self, label: str, outcome: bool) -> None:
        index = self._index(label)  # computed BEFORE the history shifts
        self._last_index = index
        counter = self._get(index)
        counter = min(3, counter + 1) if outcome else max(0, counter - 1)
        self.table[index] = counter
        self.ghr = ((self.ghr << 1) | (1 if outcome else 0)) & self._mask

    def _state(self, label: str, use_last_index: bool = False) -> str:
        index = self._last_index if (use_last_index and self._last_index is not None) else self._index(label)
        counter = self._get(index)
        ghr_bits = format(self.ghr, f"0{self.history_bits}b")
        return f"{counter} {_COUNTER_NAMES[counter]} ghr={ghr_bits}"

    def state_before(self, label: str) -> str:
        return self._state(label)

    def state_after(self, label: str) -> str:
        # Report the entry that was just updated, plus the NEW history value.
        return self._state(label, use_last_index=True)

    def detail_after(self, label: str) -> Dict[str, object]:
        index = self._last_index if self._last_index is not None else self._index(label)
        counter = self._get(index)
        return {
            "counter": counter,
            "name": _COUNTER_NAMES[counter],
            "ghr": self.ghr,
            "ghr_bits": format(self.ghr, f"0{self.history_bits}b"),
            "table_index": index,
        }


def make_predictor(config: BranchConfig) -> BasePredictor:
    if config.predictor == "always_taken":
        return AlwaysTaken()
    if config.predictor == "always_not_taken":
        return AlwaysNotTaken()
    if config.predictor == "one_bit":
        return OneBitPredictor(config.initial_prediction)
    if config.predictor == "two_bit":
        return TwoBitPredictor(config.initial_counter)
    if config.predictor == "ghr":
        return GsharePredictor(config.history_bits, config.initial_counter, config.initial_ghr)
    raise ValueError(f"Unknown predictor '{config.predictor}'")


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #


@dataclass
class BranchStep:
    index: int
    label: str
    prediction: bool
    actual: bool
    correct: bool
    state_before: str
    state_after: str
    detail: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "label": self.label,
            "prediction": self.prediction,
            "actual": self.actual,
            "correct": self.correct,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "detail": self.detail,
        }


@dataclass
class LabelStats:
    label: str
    total: int = 0
    correct: int = 0

    @property
    def mispredicted(self) -> int:
        return self.total - self.correct

    @property
    def accuracy(self) -> float:
        return (100.0 * self.correct / self.total) if self.total else 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "total": self.total,
            "correct": self.correct,
            "mispredicted": self.mispredicted,
            "accuracy": round(self.accuracy, 6),
        }


@dataclass
class BranchResult:
    predictor: str
    steps: List[BranchStep]
    total: int
    correct: int
    mispredicted: int
    accuracy: float
    penalty_cycles: int
    label_stats: List[LabelStats]
    notes: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "predictor": self.predictor,
            "steps": [s.to_dict() for s in self.steps],
            "total": self.total,
            "correct": self.correct,
            "mispredicted": self.mispredicted,
            "accuracy": round(self.accuracy, 6),
            "penalty_cycles": self.penalty_cycles,
            "label_stats": [ls.to_dict() for ls in self.label_stats],
            "notes": self.notes,
        }


def _notes(config: BranchConfig) -> List[str]:
    notes = [
        "Updates are online: state changes only after each outcome is revealed.",
    ]
    if config.predictor in ("always_taken", "always_not_taken"):
        notes.append("Static predictor — no state, every prediction is identical.")
    elif config.predictor == "one_bit":
        notes.append(
            f"Unseen branch labels predict '{config.initial_prediction}' until their first outcome."
        )
    elif config.predictor == "two_bit":
        notes.append(
            f"Every label starts at counter {config.initial_counter} "
            f"({_COUNTER_NAMES[config.initial_counter]})."
        )
    else:
        notes.append(
            f"gshare: table size {1 << config.history_bits} "
            f"(2^{config.history_bits}), counters start at {config.initial_counter}, "
            f"history register starts at {format(config.initial_ghr & ((1 << config.history_bits) - 1), '0%db' % config.history_bits)}."
        )
    return notes


def simulate(sequence: List[BranchEvent], config: BranchConfig) -> BranchResult:
    """Run one predictor over one branch-outcome sequence.

    An empty sequence is valid and yields zeroed statistics (the UI shows an
    empty state instead of 0%).
    """
    errors = validate_config(config)
    if errors:
        raise ValueError("; ".join(errors))
    if len(sequence) > MAX_SEQUENCE_LENGTH:
        raise ValueError(f"Sequence longer than {MAX_SEQUENCE_LENGTH} branches.")

    predictor = make_predictor(config)
    steps: List[BranchStep] = []
    stats: Dict[str, LabelStats] = {}

    for index, event in enumerate(sequence):
        prediction = predictor.predict(event.label)
        state_before = predictor.state_before(event.label)
        correct = prediction == event.outcome
        predictor.update(event.label, event.outcome)
        steps.append(
            BranchStep(
                index=index,
                label=event.label,
                prediction=prediction,
                actual=event.outcome,
                correct=correct,
                state_before=state_before,
                state_after=predictor.state_after(event.label),
                detail=predictor.detail_after(event.label),
            )
        )
        entry = stats.setdefault(event.label, LabelStats(label=event.label))
        entry.total += 1
        entry.correct += 1 if correct else 0

    total = len(steps)
    correct_total = sum(1 for s in steps if s.correct)
    mispredicted = total - correct_total
    accuracy = (100.0 * correct_total / total) if total else 0.0

    return BranchResult(
        predictor=config.predictor,
        steps=steps,
        total=total,
        correct=correct_total,
        mispredicted=mispredicted,
        accuracy=accuracy,
        penalty_cycles=mispredicted * config.misprediction_penalty,
        label_stats=[stats[k] for k in sorted(stats)],
        notes=_notes(config),
    )


# --------------------------------------------------------------------------- #
# Input parsing (shared with the UI so both show identical errors)
# --------------------------------------------------------------------------- #

_TAKEN_TOKENS = {"t", "1", "taken", "y", "yes"}
_NOT_TAKEN_TOKENS = {"n", "0", "not", "not_taken", "nt", "no"}


def parse_sequence(text: str) -> List[BranchEvent]:
    """Parse user text into branch events.

    accepted forms per whitespace/comma separated token:
      ``T`` / ``N``              — a single branch (all tokens share label "B1")
      ``<label>:T`` / ``<label>:N`` — named branches, e.g. ``loop:T if:N``
    """
    events: List[BranchEvent] = []
    shared_label = "B1"
    uses_named_labels = False
    for raw in text.replace(",", " ").replace("\n", " ").split():
        if not raw:
            continue
        label = shared_label
        token = raw
        if ":" in raw:
            label, _, token = raw.partition(":")
            label = label.strip()
            uses_named_labels = True
            if not label or not all(c.isalnum() or c == "_" for c in label):
                raise ValueError(
                    f"Invalid branch label '{label}' (letters, digits and '_' only)."
                )
        low = token.strip().lower()
        if low in _TAKEN_TOKENS:
            outcome = True
        elif low in _NOT_TAKEN_TOKENS:
            outcome = False
        else:
            raise ValueError(
                f"Cannot read outcome '{token}'. Use T/N, 1/0, taken/not."
            )
        events.append(BranchEvent(label=label, outcome=outcome))
        if len(events) > MAX_SEQUENCE_LENGTH:
            raise ValueError(f"Sequence longer than {MAX_SEQUENCE_LENGTH} branches.")
    if uses_named_labels:
        # sanity: named sequences are fine as-is
        return events
    return events
