"""Set-associative cache simulator — MicroShield Stage 3.

CANONICAL MODEL
===============
Mirrored 1:1 in TypeScript at ``frontend/src/lib/engines/cache.ts`` and
validated against ``fixtures/cache.json`` (hand-computed cases).

Geometry
--------
    sets = capacity / (line_size * associativity)

``capacity``, ``line_size`` are powers of two; the division must be exact
(so ``associativity`` must divide ``capacity / line_size``). Addresses are
byte addresses; the accessed word may be at any offset inside its block.

    block_base = address // line_size * line_size
    set_index  = (address // line_size) % sets
    tag        = address // (line_size * sets)

Replacement
-----------
* ``lru``  — per-set recency order; a hit moves the line to the most-recent
  end. The least-recently-used line is the next victim.
* ``fifo`` — per-set insertion order; hits do NOT reorder. The
  oldest-inserted line is the next victim.

Both orders are maintained as an "eviction order" list per set
(index 0 = next victim), which is what the UI displays.

Miss classification (the 3 C's, documented)
-------------------------------------------
Every miss is classified against a *shadow* cache: a fully-associative LRU
cache with the same capacity and line size (ways = capacity / line_size):

* compulsory — the block has never been accessed before (cold start).
* capacity  — the shadow cache misses too (the data no longer fits).
* conflict  — the shadow cache hits (the data would fit, but this set
  mapping evicted it).

The classification is computed BEFORE the access mutates either cache.
A fully-associative real cache therefore reports zero conflict misses.

Writes
------
Reads and writes both allocate on miss (write-allocate) and hit/miss the
same way; stats split them (``read_hits`` / ``write_hits``). The model has
no dirty-bit/write-back traffic — see docs/LIMITATIONS.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

REPLACEMENT_POLICIES = ("lru", "fifo")

MAX_ACCESSES = 2000
DEFAULT_CAPACITY = 256
DEFAULT_LINE_SIZE = 16
DEFAULT_ASSOCIATIVITY = 4


class CacheParseError(ValueError):
    """Invalid access-sequence text, carrying the offending token index."""

    def __init__(self, token: int, message: str) -> None:
        super().__init__(f"token {token}: {message}")
        self.token = token
        self.message = message


@dataclass(frozen=True)
class CacheConfig:
    capacity: int = DEFAULT_CAPACITY
    line_size: int = DEFAULT_LINE_SIZE
    associativity: int = DEFAULT_ASSOCIATIVITY
    replacement: str = "lru"

    @property
    def sets(self) -> Optional[int]:
        denom = self.line_size * self.associativity
        if denom <= 0 or self.capacity % denom != 0:
            return None
        return self.capacity // denom

    @property
    def shadow_ways(self) -> Optional[int]:
        if self.line_size <= 0 or self.capacity % self.line_size != 0:
            return None
        return self.capacity // self.line_size


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def validate_cache_config(config: CacheConfig) -> List[str]:
    errors: List[str] = []
    if not _is_power_of_two(config.capacity) or not (4 <= config.capacity <= 65536):
        errors.append("capacity must be a power of two between 4 and 65536 bytes.")
    if not _is_power_of_two(config.line_size) or not (4 <= config.line_size <= 1024):
        errors.append("line_size must be a power of two between 4 and 1024 bytes.")
    if config.line_size > config.capacity:
        errors.append("line_size cannot exceed capacity.")
    if not (1 <= config.associativity <= 64):
        errors.append("associativity must be between 1 and 64.")
    if config.line_size <= config.capacity and config.capacity % config.line_size == 0:
        ways = config.capacity // config.line_size
        if config.associativity > ways:
            errors.append(
                f"associativity {config.associativity} exceeds the {ways} possible "
                "lines for this capacity/line_size."
            )
        elif config.associativity >= 1 and ways % config.associativity != 0:
            errors.append(
                f"associativity {config.associativity} must divide "
                f"capacity/line_size ({ways}) so every set holds whole lines."
            )
    if config.replacement not in REPLACEMENT_POLICIES:
        errors.append(
            f"Unknown replacement policy '{config.replacement}'. "
            f"Choose one of: {', '.join(REPLACEMENT_POLICIES)}."
        )
    return errors


# ---------------------------------------------------------------------------
# Access-sequence parsing (identical rules to the TypeScript implementation)
# ---------------------------------------------------------------------------


def _parse_address(token: str) -> int:
    t = token.strip()
    sign = 1
    if t[:1] in ("+", "-"):
        if t[0] == "-":
            raise CacheParseError(0, f"address '{t}' must be non-negative")
        t = t[1:]
    low = t.lower()
    try:
        if low.startswith("0x"):
            value = int(t[2:], 16)
        elif low.startswith("0b"):
            value = int(t[2:], 2)
        elif low.startswith("0o"):
            value = int(t[2:], 8)
        elif low.isdigit():
            value = int(t, 10)
        else:
            raise CacheParseError(0, f"cannot read address '{token}'")
    except CacheParseError:
        raise
    except ValueError:
        raise CacheParseError(0, f"cannot read address '{token}'") from None
    if not (0 <= value <= 0x7FFFFFFF):
        raise CacheParseError(0, f"address '{token}' out of range")
    return sign * value


def parse_accesses(text: str) -> List[Tuple[str, int]]:
    """Parse ``R16 W0x20 r4 ...`` into (op, address) pairs.

    Tokens may be separated by whitespace, commas or newlines. Each token
    starts with R (read) or W (write) followed by the byte address.
    """
    accesses: List[Tuple[str, int]] = []
    raw_tokens = text.replace(",", " ").replace("\n", " ").split()
    for index, raw in enumerate(raw_tokens):
        if not raw:
            continue
        head = raw[0].upper()
        if head not in ("R", "W"):
            raise CacheParseError(index, f"token '{raw}' must start with R or W")
        if len(raw) < 2:
            raise CacheParseError(index, f"token '{raw}' is missing an address")
        try:
            address = _parse_address(raw[1:])
        except CacheParseError as exc:
            raise CacheParseError(index, exc.message) from None
        accesses.append((head, address))
        if len(accesses) > MAX_ACCESSES:
            raise CacheParseError(index, f"sequence longer than {MAX_ACCESSES} accesses")
    return accesses


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass
class _Line:
    tag: int
    block_base: int


@dataclass
class CacheResult:
    config: CacheConfig
    sets: int
    accesses: int
    reads: int
    writes: int
    hits: int
    misses: int
    read_hits: int
    write_hits: int
    evictions: int
    misses_by_type: Dict[str, int]
    steps: List[Dict[str, object]]
    final_state: List[List[int]]
    notes: List[str]

    @property
    def hit_rate(self) -> Optional[float]:
        return (self.hits / self.accesses) if self.accesses else None

    def to_dict(self) -> Dict[str, object]:
        return {
            "config": {
                "capacity": self.config.capacity,
                "line_size": self.config.line_size,
                "associativity": self.config.associativity,
                "replacement": self.config.replacement,
            },
            "sets": self.sets,
            "accesses": self.accesses,
            "reads": self.reads,
            "writes": self.writes,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 6) if self.hit_rate is not None else None,
            "read_hits": self.read_hits,
            "write_hits": self.write_hits,
            "evictions": self.evictions,
            "misses_by_type": self.misses_by_type,
            "steps": self.steps,
            "final_state": self.final_state,
            "notes": self.notes,
        }


def _model_notes(config: CacheConfig, sets: int) -> List[str]:
    ways = config.capacity // config.line_size if config.line_size else 0
    notes = [
        f"{sets} set(s) x {config.associativity} way(s) x {config.line_size} B "
        f"= {config.capacity} B.",
        "Cold start: every line is invalid before the first access.",
        "Misses are classified with a fully-associative LRU shadow cache of "
        "the same capacity (compulsory / capacity / conflict).",
    ]
    if config.replacement == "lru":
        notes.append(
            "LRU: a hit refreshes recency; the least-recently-used line is evicted next."
        )
    else:
        notes.append(
            "FIFO: hits do not reorder; the oldest-inserted line is evicted next."
        )
    if sets == 1:
        notes.append("Fully associative: conflict misses are impossible here.")
    elif config.associativity == 1:
        notes.append("Direct-mapped: every miss other than compulsory is a conflict miss.")
    notes.append("Writes allocate on miss (write-allocate); no dirty-bit traffic is modeled.")
    if ways and config.associativity == ways:
        pass  # fully associative — covered by the sets == 1 note
    return notes


def simulate_cache(
    accesses: Sequence[Tuple[str, int]],
    config: Optional[CacheConfig] = None,
) -> CacheResult:
    """Replay a read/write sequence through the cache and classify misses."""
    config = config or CacheConfig()
    errors = validate_cache_config(config)
    if errors:
        raise ValueError("; ".join(errors))

    sets = config.sets
    if sets is None:
        raise ValueError(
            "cache geometry is inconsistent: capacity must divide into "
            "capacity/(line_size*associativity) whole sets"
        )
    shadow_ways = config.shadow_ways or 0

    if len(accesses) > MAX_ACCESSES:
        raise ValueError(f"sequence longer than {MAX_ACCESSES} accesses")

    notes = _model_notes(config, sets)

    if not accesses:
        return CacheResult(
            config=config,
            sets=sets,
            accesses=0,
            reads=0,
            writes=0,
            hits=0,
            misses=0,
            read_hits=0,
            write_hits=0,
            evictions=0,
            misses_by_type={"compulsory": 0, "capacity": 0, "conflict": 0},
            steps=[],
            final_state=[[] for _ in range(sets)],
            notes=notes,
        )

    # per-set resident lines in eviction order (index 0 = next victim)
    state: List[List[_Line]] = [[] for _ in range(sets)]
    # shadow cache: fully-associative LRU, eviction order too
    shadow: List[_Line] = []
    seen_blocks: set = set()

    reads = writes = hits = misses = read_hits = write_hits = evictions = 0
    misses_by_type = {"compulsory": 0, "capacity": 0, "conflict": 0}
    steps: List[Dict[str, object]] = []

    for index, raw in enumerate(accesses):
        op, address = raw[0].upper(), int(raw[1])
        if op not in ("R", "W"):
            raise ValueError(f"token {index}: op must be 'R' or 'W'")
        if address < 0:
            raise ValueError(f"token {index}: address must be non-negative")

        block_base = address // config.line_size * config.line_size
        set_index = (address // config.line_size) % sets
        tag = address // (config.line_size * sets)

        if op == "R":
            reads += 1
        else:
            writes += 1

        lines = state[set_index]
        resident_index = next(
            (i for i, line in enumerate(lines) if line.tag == tag), None
        )
        hit = resident_index is not None
        miss_type: Optional[str] = None
        evicted_block: Optional[int] = None
        evicted_tag: Optional[int] = None

        if hit:
            hits += 1
            if op == "R":
                read_hits += 1
            else:
                write_hits += 1
            if config.replacement == "lru":
                line = lines.pop(resident_index)  # type: ignore[arg-type]
                lines.append(line)
        else:
            misses += 1
            # classify BEFORE mutating either cache
            if block_base not in seen_blocks:
                miss_type = "compulsory"
            elif any(line.block_base == block_base for line in shadow):
                miss_type = "conflict"
            else:
                miss_type = "capacity"
            misses_by_type[miss_type] += 1

            if len(lines) >= config.associativity:
                victim = lines.pop(0)
                evicted_block = victim.block_base
                evicted_tag = victim.tag
                evictions += 1
            lines.append(_Line(tag=tag, block_base=block_base))
            seen_blocks.add(block_base)

        # shadow cache (fully-associative LRU)
        shadow_index = next(
            (i for i, line in enumerate(shadow) if line.block_base == block_base), None
        )
        if shadow_index is not None:
            line = shadow.pop(shadow_index)
            shadow.append(line)
        else:
            if len(shadow) >= shadow_ways:
                shadow.pop(0)
            shadow.append(_Line(tag=tag, block_base=block_base))

        steps.append(
            {
                "index": index,
                "op": op,
                "address": address,
                "set": set_index,
                "tag": tag,
                "block_base": block_base,
                "hit": hit,
                "miss_type": miss_type,
                "evicted_block": evicted_block,
                "evicted_tag": evicted_tag,
                "state_after": [[line.block_base for line in s] for s in state],
            }
        )

    return CacheResult(
        config=config,
        sets=sets,
        accesses=len(accesses),
        reads=reads,
        writes=writes,
        hits=hits,
        misses=misses,
        read_hits=read_hits,
        write_hits=write_hits,
        evictions=evictions,
        misses_by_type=misses_by_type,
        steps=steps,
        final_state=[[line.block_base for line in s] for s in state],
        notes=notes,
    )
