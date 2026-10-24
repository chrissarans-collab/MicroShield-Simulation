/**
 * Branch predictor models — MicroShield Stage 1 (TypeScript engine).
 *
 * CANONICAL MODEL: backend/app/core/branch_predictors.py (Python reference).
 * Both implementations are validated against fixtures/branch_predictor.json.
 *
 * Training policy (documented): updates are *online* — state changes only
 * after the actual outcome is revealed, exactly like a real CPU updating its
 * predictor when a branch resolves. Prediction happens strictly before the
 * update. Unseen labels begin in the configured initial state.
 */

export const PREDICTORS = [
  'always_taken',
  'always_not_taken',
  'one_bit',
  'two_bit',
  'ghr',
] as const;
export type PredictorType = (typeof PREDICTORS)[number];

export const MAX_BRANCH_SEQUENCE = 500;

const COUNTER_NAMES = ['SN', 'WN', 'WT', 'ST'] as const;

export interface BranchEvent {
  label: string;
  outcome: boolean; // true = taken
}

export interface BranchConfig {
  predictor: PredictorType;
  /** one_bit: state for labels never seen before */
  initialPrediction: 'taken' | 'not_taken';
  /** two_bit / ghr: counter every table entry starts from (0..3) */
  initialCounter: number;
  /** ghr: initial global history register (masked to historyBits) */
  initialGhr: number;
  /** ghr: history width; table size = 2 ** historyBits */
  historyBits: number;
  /** cycles assumed lost per misprediction */
  mispredictionPenalty: number;
}

export const DEFAULT_BRANCH_CONFIG: BranchConfig = {
  predictor: 'two_bit',
  initialPrediction: 'taken',
  initialCounter: 2,
  initialGhr: 0,
  historyBits: 4,
  mispredictionPenalty: 2,
};

export interface BranchStep {
  index: number;
  label: string;
  prediction: boolean;
  actual: boolean;
  correct: boolean;
  stateBefore: string;
  stateAfter: string;
  detail: Record<string, string | number>;
}

export interface BranchLabelStats {
  label: string;
  total: number;
  correct: number;
  mispredicted: number;
  accuracy: number;
}

export interface BranchResult {
  predictor: PredictorType;
  steps: BranchStep[];
  total: number;
  correct: number;
  mispredicted: number;
  accuracy: number;
  penaltyCycles: number;
  labelStats: BranchLabelStats[];
  notes: string[];
}

/** FNV-1a 32-bit; matches the Python reference (mod 2^32 arithmetic). */
export function fnv1a32(text: string): number {
  const bytes = new TextEncoder().encode(text);
  let h = 0x811c9dc5;
  for (const b of bytes) {
    h = Math.imul(h ^ b, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

export function validateBranchConfig(config: BranchConfig): string[] {
  const errors: string[] = [];
  if (!PREDICTORS.includes(config.predictor)) {
    errors.push(`Unknown predictor '${config.predictor}'.`);
  }
  if (config.initialPrediction !== 'taken' && config.initialPrediction !== 'not_taken') {
    errors.push("initialPrediction must be 'taken' or 'not_taken'.");
  }
  if (!Number.isInteger(config.initialCounter) || config.initialCounter < 0 || config.initialCounter > 3) {
    errors.push('initialCounter must be an integer between 0 and 3.');
  }
  if (!Number.isInteger(config.historyBits) || config.historyBits < 1 || config.historyBits > 8) {
    errors.push('historyBits must be an integer between 1 and 8.');
  }
  if (config.initialGhr < 0) {
    errors.push('initialGhr must be >= 0.');
  }
  if (
    !Number.isInteger(config.mispredictionPenalty) ||
    config.mispredictionPenalty < 0 ||
    config.mispredictionPenalty > 20
  ) {
    errors.push('mispredictionPenalty must be an integer between 0 and 20.');
  }
  return errors;
}

// ---------------------------------------------------------------------------
// Predictor implementations
// ---------------------------------------------------------------------------

export interface Predictor {
  readonly name: PredictorType;
  predict(label: string): boolean;
  update(label: string, outcome: boolean): void;
  stateBefore(label: string): string;
  stateAfter(label: string): string;
  detailAfter(label: string): Record<string, string | number>;
}

class StaticPredictor implements Predictor {
  constructor(
    public readonly name: PredictorType,
    private readonly value: boolean,
  ) {}
  predict(): boolean {
    return this.value;
  }
  update(): void {}
  stateBefore(): string {
    return 'static';
  }
  stateAfter(): string {
    return 'static';
  }
  detailAfter(): Record<string, string | number> {
    return {};
  }
}

class OneBitPredictor implements Predictor {
  readonly name = 'one_bit' as const;
  private readonly table = new Map<string, boolean>();
  private readonly initialTaken: boolean;
  private readonly initialLabel: string;

  constructor(initialPrediction: 'taken' | 'not_taken') {
    this.initialTaken = initialPrediction === 'taken';
    this.initialLabel = this.initialTaken ? 'init(T)' : 'init(N)';
  }

  predict(label: string): boolean {
    return this.table.get(label) ?? this.initialTaken;
  }
  update(label: string, outcome: boolean): void {
    this.table.set(label, outcome);
  }
  private state(label: string): string {
    const v = this.table.get(label);
    return v === undefined ? this.initialLabel : v ? 'T' : 'N';
  }
  stateBefore(label: string): string {
    return this.state(label);
  }
  stateAfter(label: string): string {
    return this.state(label);
  }
  detailAfter(label: string): Record<string, string | number> {
    return { lastOutcome: this.table.get(label) ? 'T' : 'N' };
  }
}

class TwoBitPredictor implements Predictor {
  readonly name = 'two_bit' as const;
  private readonly table = new Map<string, number>();
  constructor(private readonly initial: number) {}
  private get(label: string): number {
    return this.table.get(label) ?? this.initial;
  }
  predict(label: string): boolean {
    return this.get(label) >= 2;
  }
  update(label: string, outcome: boolean): void {
    const c = this.get(label);
    this.table.set(label, outcome ? Math.min(3, c + 1) : Math.max(0, c - 1));
  }
  private state(label: string): string {
    if (!this.table.has(label)) return `init(${this.initial})`;
    const c = this.table.get(label)!;
    return `${c} ${COUNTER_NAMES[c]}`;
  }
  stateBefore(label: string): string {
    return this.state(label);
  }
  stateAfter(label: string): string {
    return this.state(label);
  }
  detailAfter(label: string): Record<string, string | number> {
    const c = this.get(label);
    return { counter: c, name: COUNTER_NAMES[c] };
  }
}

class GsharePredictor implements Predictor {
  readonly name = 'ghr' as const;
  readonly tableSize: number;
  private readonly mask: number;
  private readonly table = new Map<number, number>();
  private ghrValue: number;
  /** Index of the entry touched by the most recent update: because the
   * history shifts on update, the just-modified entry is not necessarily the
   * one index(label) points at afterwards. */
  private lastIndex: number | null = null;

  constructor(
    private readonly historyBits: number,
    private readonly initial: number,
    initialGhr: number,
  ) {
    this.tableSize = 1 << historyBits;
    this.mask = this.tableSize - 1;
    this.ghrValue = initialGhr & this.mask;
  }

  private index(label: string): number {
    return (fnv1a32(label) % this.tableSize) ^ this.ghrValue;
  }
  private get(index: number): number {
    return this.table.get(index) ?? this.initial;
  }
  predict(label: string): boolean {
    return this.get(this.index(label)) >= 2;
  }
  update(label: string, outcome: boolean): void {
    const index = this.index(label); // computed BEFORE the history shifts
    this.lastIndex = index;
    const c = this.get(index);
    this.table.set(index, outcome ? Math.min(3, c + 1) : Math.max(0, c - 1));
    this.ghrValue = ((this.ghrValue << 1) | (outcome ? 1 : 0)) & this.mask;
  }
  private state(label: string, useLastIndex = false): string {
    const index = useLastIndex && this.lastIndex !== null ? this.lastIndex : this.index(label);
    const c = this.get(index);
    return `${c} ${COUNTER_NAMES[c]} ghr=${this.ghrValue.toString(2).padStart(this.historyBits, '0')}`;
  }
  stateBefore(label: string): string {
    return this.state(label);
  }
  stateAfter(label: string): string {
    // Report the entry that was just updated, plus the NEW history value.
    return this.state(label, true);
  }
  detailAfter(label: string): Record<string, string | number> {
    const index = this.lastIndex !== null ? this.lastIndex : this.index(label);
    const c = this.get(index);
    return {
      counter: c,
      name: COUNTER_NAMES[c],
      ghr: this.ghrValue,
      ghrBits: this.ghrValue.toString(2).padStart(this.historyBits, '0'),
      tableIndex: index,
    };
  }
}

export function makePredictor(config: BranchConfig): Predictor {
  switch (config.predictor) {
    case 'always_taken':
      return new StaticPredictor('always_taken', true);
    case 'always_not_taken':
      return new StaticPredictor('always_not_taken', false);
    case 'one_bit':
      return new OneBitPredictor(config.initialPrediction);
    case 'two_bit':
      return new TwoBitPredictor(config.initialCounter);
    case 'ghr':
      return new GsharePredictor(config.historyBits, config.initialCounter, config.initialGhr);
  }
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

function predictorNotes(config: BranchConfig): string[] {
  const notes = ['Updates are online: state changes only after each outcome is revealed.'];
  switch (config.predictor) {
    case 'always_taken':
    case 'always_not_taken':
      notes.push('Static predictor — no state, every prediction is identical.');
      break;
    case 'one_bit':
      notes.push(
        `Unseen branch labels predict '${config.initialPrediction}' until their first outcome.`,
      );
      break;
    case 'two_bit':
      notes.push(
        `Every label starts at counter ${config.initialCounter} (${COUNTER_NAMES[config.initialCounter]}).`,
      );
      break;
    case 'ghr':
      notes.push(
        `gshare: table size ${1 << config.historyBits} (2^${config.historyBits}), counters start at ` +
          `${config.initialCounter}, history register starts at ` +
          `${(config.initialGhr & ((1 << config.historyBits) - 1)).toString(2).padStart(config.historyBits, '0')}.`,
      );
      break;
  }
  return notes;
}

/**
 * Run one predictor over one branch-outcome sequence.
 * An empty sequence is valid and yields zeroed statistics.
 */
export function simulateBranches(
  sequence: BranchEvent[],
  config: BranchConfig = DEFAULT_BRANCH_CONFIG,
): BranchResult {
  const errors = validateBranchConfig(config);
  if (errors.length > 0) throw new Error(errors.join(' '));
  if (sequence.length > MAX_BRANCH_SEQUENCE) {
    throw new Error(`Sequence longer than ${MAX_BRANCH_SEQUENCE} branches.`);
  }

  const predictor = makePredictor(config);
  const steps: BranchStep[] = [];
  const stats = new Map<string, BranchLabelStats>();

  sequence.forEach((event, index) => {
    const prediction = predictor.predict(event.label);
    const stateBefore = predictor.stateBefore(event.label);
    const correct = prediction === event.outcome;
    predictor.update(event.label, event.outcome);
    steps.push({
      index,
      label: event.label,
      prediction,
      actual: event.outcome,
      correct,
      stateBefore,
      stateAfter: predictor.stateAfter(event.label),
      detail: predictor.detailAfter(event.label),
    });
    let entry = stats.get(event.label);
    if (!entry) {
      entry = { label: event.label, total: 0, correct: 0, mispredicted: 0, accuracy: 0 };
      stats.set(event.label, entry);
    }
    entry.total += 1;
    entry.correct += correct ? 1 : 0;
  });

  const total = steps.length;
  const correctTotal = steps.filter((s) => s.correct).length;
  const mispredicted = total - correctTotal;

  const labelStats = [...stats.values()]
    .sort((a, b) => (a.label < b.label ? -1 : 1))
    .map((ls) => ({
      ...ls,
      mispredicted: ls.total - ls.correct,
      accuracy: ls.total ? (100 * ls.correct) / ls.total : 0,
    }));

  return {
    predictor: config.predictor,
    steps,
    total,
    correct: correctTotal,
    mispredicted,
    accuracy: total ? (100 * correctTotal) / total : 0,
    penaltyCycles: mispredicted * config.mispredictionPenalty,
    labelStats,
    notes: predictorNotes(config),
  };
}

// ---------------------------------------------------------------------------
// Input parsing (identical rules to the Python implementation)
// ---------------------------------------------------------------------------

const TAKEN_TOKENS = new Set(['t', '1', 'taken', 'y', 'yes']);
const NOT_TAKEN_TOKENS = new Set(['n', '0', 'not', 'not_taken', 'nt', 'no']);

/**
 * Parse user text into branch events.
 *   T / N                 — a single branch (all tokens share label "B1")
 *   <label>:T / <label>:N — named branches, e.g. "loop:T if:N"
 */
export function parseBranchSequence(text: string): BranchEvent[] {
  const events: BranchEvent[] = [];
  const tokens = text.replace(/[,;\n]/g, ' ').split(/\s+/).filter(Boolean);
  for (const raw of tokens) {
    let label = 'B1';
    let token = raw;
    if (raw.includes(':')) {
      const idx = raw.indexOf(':');
      label = raw.slice(0, idx).trim();
      token = raw.slice(idx + 1);
      if (!label || !/^[A-Za-z0-9_]+$/.test(label)) {
        throw new Error(`Invalid branch label '${label}' (letters, digits and '_' only).`);
      }
    }
    const low = token.trim().toLowerCase();
    let outcome: boolean;
    if (TAKEN_TOKENS.has(low)) outcome = true;
    else if (NOT_TAKEN_TOKENS.has(low)) outcome = false;
    else throw new Error(`Cannot read outcome '${token}'. Use T/N, 1/0, taken/not.`);
    events.push({ label, outcome });
    if (events.length > MAX_BRANCH_SEQUENCE) {
      throw new Error(`Sequence longer than ${MAX_BRANCH_SEQUENCE} branches.`);
    }
  }
  return events;
}
