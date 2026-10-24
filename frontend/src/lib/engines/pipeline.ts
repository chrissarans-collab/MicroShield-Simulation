/**
 * Five-stage in-order pipeline simulator — MicroShield Stage 2 (TS engine).
 *
 * CANONICAL MODEL: backend/app/core/pipeline.py (Python reference).
 * Both implementations are validated against fixtures/pipeline.json.
 *
 * See the Python module docstring for the full documented model: stages
 * IF->ID->EX->MEM->WB, hazard/forwarding rules, branch prediction in ID with
 * resolution at the end of EX, blocking memory, and the cycle-count
 * convention (CPI = total_cycles / retired, null when retired == 0).
 */

import {
  type Predictor,
  makePredictor,
  DEFAULT_BRANCH_CONFIG,
} from './branch';

export const BRANCH_POLICIES = [
  'static_not_taken',
  'static_taken',
  'one_bit',
  'two_bit',
  'ghr',
] as const;
export type BranchPolicy = (typeof BRANCH_POLICIES)[number];

export const MAX_PROGRAM_LENGTH = 400;
export const DEFAULT_MAX_CYCLES = 5000;
export const NUM_REGISTERS = 16;

const ALU_OPS = new Set(['ADD', 'SUB', 'AND', 'OR', 'XOR', 'SLT']);

/** memory access hook: returns [latency_cycles, hit_or_null] */
export type MemoryAccessHook = (
  op: 'LW' | 'SW',
  address: number,
) => [number, boolean | null];

export class PipelineParseError extends Error {
  constructor(
    public readonly line: number,
    message: string,
  ) {
    super(`line ${line}: ${message}`);
    this.name = 'PipelineParseError';
  }
}

export function wrap32(value: number): number {
  return (((value + 2 ** 31) % 2 ** 32) + 2 ** 32) % 2 ** 32 - 2 ** 31;
}

function normalizeReg(token: string): string {
  const stripped = token.trim();
  if (/^x\d+$/i.test(stripped)) return `x${parseInt(stripped.slice(1), 10)}`;
  return stripped;
}

// ---------------------------------------------------------------------------
// Instruction representation
// ---------------------------------------------------------------------------

export interface Instruction {
  index: number;
  pc: number;
  mnemonic: string;
  text: string;
  rd: number | null;
  rs1: number | null;
  rs2: number | null;
  imm: number;
  targetPc: number | null;
  isBranch: boolean;
  isLoad: boolean;
  isStore: boolean;
  isHalt: boolean;
  /** derived at construction (mirrors the Python properties) */
  writesRegister: boolean;
  sources: number[];
  usesMemory: boolean;
}

export interface PipelineConfig {
  forwarding: boolean;
  branchPolicy: BranchPolicy;
  memoryLatency: number;
  // dynamic-predictor options (branch_policy one_bit / two_bit / ghr)
  initialPrediction: 'taken' | 'not_taken';
  initialCounter: number;
  historyBits: number;
  initialGhr: number;
  maxCycles: number;
}

export const DEFAULT_PIPELINE_CONFIG: PipelineConfig = {
  forwarding: true,
  branchPolicy: 'static_not_taken',
  memoryLatency: 1,
  initialPrediction: 'taken',
  initialCounter: 2,
  historyBits: 4,
  initialGhr: 0,
  maxCycles: DEFAULT_MAX_CYCLES,
};

export function validatePipelineConfig(config: PipelineConfig): string[] {
  const errors: string[] = [];
  if (!BRANCH_POLICIES.includes(config.branchPolicy)) {
    errors.push(
      `Unknown branch policy '${config.branchPolicy}'. ` +
        `Choose one of: ${BRANCH_POLICIES.join(', ')}.`,
    );
  }
  if (config.initialPrediction !== 'taken' && config.initialPrediction !== 'not_taken') {
    errors.push("initialPrediction must be 'taken' or 'not_taken'.");
  }
  if (
    !Number.isInteger(config.initialCounter) ||
    config.initialCounter < 0 ||
    config.initialCounter > 3
  ) {
    errors.push('initial_counter must be between 0 and 3.');
  }
  if (!Number.isInteger(config.historyBits) || config.historyBits < 1 || config.historyBits > 8) {
    errors.push('history_bits must be between 1 and 8.');
  }
  if (config.initialGhr < 0) {
    errors.push('initial_ghr must be >= 0.');
  }
  if (
    !Number.isInteger(config.memoryLatency) ||
    config.memoryLatency < 1 ||
    config.memoryLatency > 8
  ) {
    errors.push('memory_latency must be between 1 and 8 cycles.');
  }
  if (
    !Number.isInteger(config.maxCycles) ||
    config.maxCycles < 10 ||
    config.maxCycles > 100_000
  ) {
    errors.push('max_cycles must be between 10 and 100000.');
  }
  return errors;
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

function parseRegister(token: string): number {
  const low = token.trim().toLowerCase();
  if (!/^x\d+$/.test(low)) {
    throw new PipelineParseError(0, `bad register '${token}' (expected x0..x15)`);
  }
  const number = parseInt(low.slice(1), 10);
  if (!(number >= 0 && number < NUM_REGISTERS)) {
    throw new PipelineParseError(0, `register '${token}' out of range (x0..x15)`);
  }
  return number;
}

function parseImm(token: string): number {
  const t = token.trim();
  const match = /^([-+]?)(0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+|0|[1-9][0-9]*)$/.exec(t);
  if (!match) throw new PipelineParseError(0, `bad immediate '${t}'`);
  const sign = match[1] === '-' ? -1 : 1;
  const body = match[2];
  let value: number;
  if (body.startsWith('0x') || body.startsWith('0X')) value = parseInt(body.slice(2), 16);
  else if (body.startsWith('0b') || body.startsWith('0B')) value = parseInt(body.slice(2), 2);
  else if (body.startsWith('0o') || body.startsWith('0O')) value = parseInt(body.slice(2), 8);
  else value = parseInt(body, 10);
  const signed = sign * value;
  if (!(signed >= -999999 && signed <= 999999)) {
    throw new PipelineParseError(0, `immediate '${t}' out of range`);
  }
  return signed;
}

function parseMemOperand(token: string): [number, number] {
  /** Parse `8(x2)` -> [offset, base register]. */
  const t = token.trim();
  const open = t.indexOf('(');
  if (open < 0 || !t.endsWith(')')) {
    throw new PipelineParseError(0, `bad memory operand '${t}' (expected imm(reg))`);
  }
  const offsetPart = t.slice(0, open);
  const regPart = t.slice(open + 1, -1);
  const offset = offsetPart.trim() ? parseImm(offsetPart) : 0;
  return [offset, parseRegister(regPart)];
}

function canonicalText(mnemonic: string, args: string[]): string {
  const regs = args.map((a) => (/^x\d+$/i.test(a) ? normalizeReg(a) : a));
  if (args.length === 0) return mnemonic;
  if (mnemonic === 'LW' || mnemonic === 'SW') {
    return `${mnemonic} ${regs[0]}, ${args.slice(1).join('')}`;
  }
  if (mnemonic === 'BEQ' || mnemonic === 'BNE') {
    return `${mnemonic} ${regs[0]}, ${regs[1]}, ${args[2]}`;
  }
  if (mnemonic === 'ADDI') {
    return `${mnemonic} ${regs[0]}, ${regs[1]}, ${args[2]}`;
  }
  if (ALU_OPS.has(mnemonic)) {
    return `${mnemonic} ${regs[0]}, ${regs[1]}, ${regs[2]}`;
  }
  return mnemonic;
}

function makeInstruction(
  base: Omit<Instruction, 'writesRegister' | 'sources' | 'usesMemory'>,
): Instruction {
  const writesRegister = base.rd !== null && base.rd !== 0;
  const sources = [base.rs1, base.rs2].filter((r): r is number => r !== null);
  return {
    ...base,
    writesRegister,
    sources,
    usesMemory: base.isLoad || base.isStore,
  };
}

/** Parse toy assembly into instructions. Throws PipelineParseError with line numbers. */
export function parseProgram(source: string): Instruction[] {
  // Pass 1: strip comments, buffer labels, collect statement bodies.
  const entries: Array<[number, string]> = []; // (lineNo, body)
  const labelTargets = new Map<string, number>(); // label -> entry position
  let pending: Array<[number, string]> = []; // labels waiting for a statement

  const sourceLines = source.split('\n');
  for (let i = 0; i < sourceLines.length; i += 1) {
    const lineNo = i + 1;
    let line = sourceLines[i].split('#')[0].split(';')[0].trim();
    if (!line) continue;
    // one or more "name:" prefixes may precede the statement
    const collected = [...pending];
    pending = [];
    for (;;) {
      const sep = line.indexOf(':');
      if (sep < 0) break;
      const name = line.slice(0, sep).trim();
      if (!name || /[0-9]/.test(name[0]) || !/^[A-Za-z0-9_]+$/.test(name)) {
        throw new PipelineParseError(lineNo, `invalid label '${name}'`);
      }
      collected.push([lineNo, name]);
      line = line.slice(sep + 1).trim();
    }
    if (!line) {
      // label with no statement yet — attach to the NEXT statement
      pending = collected;
      continue;
    }
    const position = entries.length;
    for (const [ln, name] of collected) {
      if (labelTargets.has(name)) {
        throw new PipelineParseError(ln, `duplicate label '${name}'`);
      }
      labelTargets.set(name, position);
    }
    entries.push([lineNo, line]);
  }

  // Trailing labels point at an implicit HALT after the last statement.
  if (pending.length > 0) {
    const position = entries.length;
    for (const [ln, name] of pending) {
      if (labelTargets.has(name)) {
        throw new PipelineParseError(ln, `duplicate label '${name}'`);
      }
      labelTargets.set(name, position);
    }
    entries.push([pending[pending.length - 1][0], 'HALT']);
  }

  if (entries.length > MAX_PROGRAM_LENGTH) {
    throw new PipelineParseError(1, `program longer than ${MAX_PROGRAM_LENGTH} instructions`);
  }

  // Pass 2: parse each statement.
  return entries.map(([lineNo, body], position) => {
    try {
      return parseStatement(position, body, labelTargets, entries.length);
    } catch (err) {
      if (err instanceof PipelineParseError && err.line === 0) {
        // re-raise carrying the statement's real line number
        const inner: PipelineParseError = err;
        throw new PipelineParseError(lineNo, inner.message.replace(/^line 0: /, ''));
      }
      throw err;
    }
  });
}

function parseStatement(
  position: number,
  body: string,
  labelTargets: Map<string, number>,
  size: number,
): Instruction {
  const tokens = body.replace(/,/g, ' ').split(/\s+/).filter(Boolean);
  const mnemonic = tokens[0].toUpperCase();
  const args = tokens.slice(1);
  const pc = position * 4;

  const targetPcOf = (label: string): number => {
    const target = labelTargets.get(label);
    if (target === undefined) throw new PipelineParseError(0, `unknown label '${label}'`);
    if (target >= size) throw new PipelineParseError(0, `label '${label}' has no instruction`);
    return target * 4;
  };

  const finish = (
    extra: Partial<Omit<Instruction, 'writesRegister' | 'sources' | 'usesMemory'>>,
  ): Instruction =>
    makeInstruction({
      index: position,
      pc,
      mnemonic,
      text: canonicalText(mnemonic, args),
      rd: null,
      rs1: null,
      rs2: null,
      imm: 0,
      targetPc: null,
      isBranch: false,
      isLoad: false,
      isStore: false,
      isHalt: false,
      ...extra,
    });

  if (mnemonic === 'NOP') {
    return finish({ mnemonic: 'NOP', text: 'NOP', rd: 0, rs1: 0, rs2: 0, imm: 0 });
  }
  if (mnemonic === 'HALT') {
    if (args.length > 0) throw new PipelineParseError(0, 'HALT takes no operands');
    return finish({ text: 'HALT', isHalt: true });
  }
  if (ALU_OPS.has(mnemonic)) {
    if (args.length !== 3) throw new PipelineParseError(0, `${mnemonic} expects rd, rs1, rs2`);
    return finish({
      rd: parseRegister(args[0]),
      rs1: parseRegister(args[1]),
      rs2: parseRegister(args[2]),
    });
  }
  if (mnemonic === 'ADDI') {
    if (args.length !== 3) throw new PipelineParseError(0, 'ADDI expects rd, rs1, imm');
    return finish({
      rd: parseRegister(args[0]),
      rs1: parseRegister(args[1]),
      imm: parseImm(args[2]),
    });
  }
  if (mnemonic === 'LW' || mnemonic === 'SW') {
    if (args.length !== 2) throw new PipelineParseError(0, `${mnemonic} expects rd, imm(reg)`);
    const first = parseRegister(args[0]);
    const [offset, base] = parseMemOperand(args[1]);
    if (mnemonic === 'LW') {
      return finish({ rd: first, rs1: base, imm: offset, isLoad: true });
    }
    return finish({ rd: null, rs1: base, rs2: first, imm: offset, isStore: true });
  }
  if (mnemonic === 'BEQ' || mnemonic === 'BNE') {
    if (args.length !== 3) throw new PipelineParseError(0, `${mnemonic} expects rs1, rs2, label`);
    const rs1 = parseRegister(args[0]);
    const rs2 = parseRegister(args[1]);
    const targetPc = targetPcOf(args[2]);
    return finish({
      rs1,
      rs2,
      imm: targetPc - pc,
      targetPc,
      isBranch: true,
    });
  }
  throw new PipelineParseError(0, `unknown instruction '${tokens[0]}'`);
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

interface Latch {
  /** an instruction occupying a pipeline stage for one or more cycles */
  inst: Instruction;
  seq: number; // dynamic instance id, assigned when latched into IF/ID
  operands: Map<number, number>; // values read in ID
  result: number; // ALU result / loaded value
  storeValue: number;
  address: number;
  memLeft: number;
  latency: number;
  hit: boolean | null;
  predicted: boolean | null;
}

function newLatch(inst: Instruction, seq: number): Latch {
  return {
    inst,
    seq,
    operands: new Map(),
    result: 0,
    storeValue: 0,
    address: 0,
    memLeft: 1,
    latency: 1,
    hit: null,
    predicted: null,
  };
}

export interface TraceCell {
  kind: 'instr' | 'bubble';
  seq: number | null;
  text: string;
  state: string;
  note: string;
}

export interface TraceRow {
  cycle: number;
  IF: TraceCell;
  ID: TraceCell;
  EX: TraceCell;
  MEM: TraceCell;
  WB: TraceCell;
  annotations: string[];
}

export interface PipelineBranchEvent {
  seq: number;
  pc: number;
  text: string;
  predicted: boolean;
  actual: boolean;
  correct: boolean;
  flushes: number;
  cycle: number;
}

export interface PipelineMemoryEvent {
  seq: number;
  op: 'LW' | 'SW';
  address: number;
  value: number;
  latency: number;
  hit: boolean | null;
  cycle: number;
}

export interface RetiredEntry {
  seq: number;
  pc: number;
  text: string;
  cycle: number;
}

export interface MemoryWrite {
  address: number;
  value: number;
  cycle: number;
}

export interface PipelineResult {
  instructions: Instruction[];
  totalCycles: number;
  retired: number;
  cpi: number | null;
  hazardStallCycles: number;
  memStallCycles: number;
  stallCycles: number;
  flushCycles: number;
  branches: number;
  mispredicts: number;
  forwardedOperands: number;
  trace: TraceRow[];
  branchEvents: PipelineBranchEvent[];
  memoryEvents: PipelineMemoryEvent[];
  retiredInstructions: RetiredEntry[];
  finalRegisters: number[];
  memoryWrites: MemoryWrite[];
  termination: 'completed' | 'cycle_limit';
  notes: string[];
}

function bubble(): TraceCell {
  return { kind: 'bubble', seq: null, text: '', state: 'bubble', note: '' };
}

function cell(latch: Latch, state = 'ok', note = ''): TraceCell {
  return { kind: 'instr', seq: latch.seq, text: latch.inst.text, state, note };
}

function makeBranchPredictor(config: PipelineConfig): Predictor | null {
  switch (config.branchPolicy) {
    case 'static_not_taken':
      return makePredictor({ ...DEFAULT_BRANCH_CONFIG, predictor: 'always_not_taken' });
    case 'static_taken':
      return makePredictor({ ...DEFAULT_BRANCH_CONFIG, predictor: 'always_taken' });
    case 'one_bit':
      return makePredictor({
        ...DEFAULT_BRANCH_CONFIG,
        predictor: 'one_bit',
        initialPrediction: config.initialPrediction,
      });
    case 'two_bit':
      return makePredictor({
        ...DEFAULT_BRANCH_CONFIG,
        predictor: 'two_bit',
        initialCounter: config.initialCounter,
      });
    case 'ghr':
      return makePredictor({
        ...DEFAULT_BRANCH_CONFIG,
        predictor: 'ghr',
        initialCounter: config.initialCounter,
        initialGhr: config.initialGhr,
        historyBits: config.historyBits,
      });
  }
}

function modelNotes(config: PipelineConfig): string[] {
  const notes = [
    'Cycle count includes pipeline fill and drain.',
    'Branches resolve at the end of EX; a misprediction flushes every ' +
      'younger instruction (ID latch + current fetch = up to 2 flushed).',
    'Register file: WB writes first half of the cycle, ID reads second half.',
  ];
  if (config.branchPolicy === 'static_not_taken') {
    notes.push('Branch policy: static predict-not-taken (no prediction state).');
  } else if (config.branchPolicy === 'static_taken') {
    notes.push('Branch policy: static predict-taken (no prediction state).');
  } else {
    notes.push(`Branch policy: dynamic '${config.branchPolicy}' predictor trained in EX.`);
  }
  notes.push(
    config.forwarding
      ? 'Forwarding enabled: only load-use hazards cost 1 stall cycle.'
      : 'Forwarding disabled: a distance-1 dependency costs 2 stall cycles, distance-2 costs 1.',
  );
  return notes;
}

/**
 * Run the toy CPU to completion and return every observable.
 * `memoryAccess` (optional) supplies per-access latency/hit from the cache
 * model — used by the Architecture Comparison lab.
 */
export function simulatePipeline(
  source: string,
  config: Partial<PipelineConfig> = {},
  initialRegisters: Record<number, number> = {},
  memoryAccess: MemoryAccessHook | null = null,
): PipelineResult {
  const cfg: PipelineConfig = { ...DEFAULT_PIPELINE_CONFIG, ...config };
  const errors = validatePipelineConfig(cfg);
  if (errors.length > 0) throw new Error(errors.join('; '));

  const program = parseProgram(source);
  const notes = modelNotes(cfg);

  if (program.length === 0) {
    return {
      instructions: [],
      totalCycles: 0,
      retired: 0,
      cpi: null,
      hazardStallCycles: 0,
      memStallCycles: 0,
      stallCycles: 0,
      flushCycles: 0,
      branches: 0,
      mispredicts: 0,
      forwardedOperands: 0,
      trace: [],
      branchEvents: [],
      memoryEvents: [],
      retiredInstructions: [],
      finalRegisters: new Array<number>(NUM_REGISTERS).fill(0),
      memoryWrites: [],
      termination: 'completed',
      notes,
    };
  }

  const regs = new Array<number>(NUM_REGISTERS).fill(0);
  for (const rawIndex of Object.keys(initialRegisters)) {
    const index = Number(rawIndex);
    if (index > 0 && index < NUM_REGISTERS) {
      // x0 is hardwired to zero
      regs[index] = wrap32(initialRegisters[index]);
    }
  }
  const memory = new Map<number, number>();
  const memoryWrites: MemoryWrite[] = [];
  const predictor = makeBranchPredictor(cfg);

  const programEnd = program.length * 4;
  let pc = 0;
  let fetchEnabled: boolean = true;
  let ifId: Latch | null = null;
  let idEx: Latch | null = null;
  let exMem: Latch | null = null;
  let memWb: Latch | null = null;

  let cycle = 0;
  let retired = 0;
  let hazardStalls = 0;
  let memStalls = 0;
  let flushes = 0;
  let branches = 0;
  let mispredicts = 0;
  let forwards = 0;
  let nextSeq = 0;
  const trace: TraceRow[] = [];
  const branchEvents: PipelineBranchEvent[] = [];
  const memoryEvents: PipelineMemoryEvent[] = [];
  const retiredLog: RetiredEntry[] = [];
  let finished = false;

  while (cycle < cfg.maxCycles) {
    // ---- IF (fetch happens first: it uses the PC latched last cycle) ----
    let fetched: Instruction | null = null;
    if (fetchEnabled && pc >= 0 && pc < programEnd) {
      fetched = program[Math.floor(pc / 4)];
    }

    cycle += 1;
    const annotations: string[] = [];
    const cells: Record<'IF' | 'ID' | 'EX' | 'MEM' | 'WB', TraceCell> = {
      IF: fetched
        ? { kind: 'instr', seq: null, text: fetched.text, state: 'ok', note: '' }
        : bubble(),
      ID: ifId ? cell(ifId) : bubble(),
      EX: idEx ? cell(idEx) : bubble(),
      MEM: exMem ? cell(exMem) : bubble(),
      WB: memWb ? cell(memWb) : bubble(),
    };

    // next-state registers for the end of this cycle
    let memWbNext: Latch | null = null;
    let exMemNext: Latch | null = null;
    let idExNext: Latch | null = null;
    let ifIdNext: Latch | null = null;
    let redirect: number | null = null;
    let fetchEnabledNext: boolean = fetchEnabled;

    // ---- 1. WB: always completes; register write happens first half ----
    if (memWb !== null) {
      if (memWb.inst.writesRegister && memWb.inst.rd !== null) {
        regs[memWb.inst.rd] = wrap32(memWb.result);
      }
      retired += 1;
      retiredLog.push({
        seq: memWb.seq,
        pc: memWb.inst.pc,
        text: memWb.inst.text,
        cycle,
      });
      cells.WB = cell(memWb, 'ok', 'retired');
    }

    // ---- 2. MEM: the stage is occupied for `latency` cycles ----
    let frozen = false;
    if (exMem !== null) {
      exMem.memLeft -= 1;
      if (exMem.memLeft > 0) {
        frozen = true;
        memStalls += 1;
        cells.MEM = cell(exMem, 'mem_stall', 'memory access in progress');
        cells.EX = idEx ? cell(idEx, 'stall', 'front end frozen by MEM') : bubble();
        cells.ID = ifId ? cell(ifId, 'stall', 'front end frozen by MEM') : bubble();
        cells.IF = fetched
          ? {
              kind: 'instr',
              seq: null,
              text: fetched.text,
              state: 'stall',
              note: 'refetch after memory freeze',
            }
          : bubble();
        annotations.push(
          `MEM busy: ${exMem.inst.text} waits ${exMem.latency} cycles for memory ` +
            '(blocking) — front end frozen',
        );
      } else {
        const inst = exMem.inst;
        if (inst.isLoad) {
          exMem.result = wrap32(memory.get(exMem.address) ?? 0);
        } else if (inst.isStore) {
          const value = wrap32(exMem.storeValue);
          memory.set(exMem.address, value);
          memoryWrites.push({ address: exMem.address, value, cycle });
        }
        if (inst.usesMemory) {
          memoryEvents.push({
            seq: exMem.seq,
            op: inst.isLoad ? 'LW' : 'SW',
            address: exMem.address,
            value: inst.isLoad ? exMem.result : wrap32(exMem.storeValue),
            latency: exMem.latency,
            hit: exMem.hit,
            cycle,
          });
        }
        memWbNext = exMem;
      }
    }

    if (frozen) {
      // idEx, ifId and pc all hold; the consumed WB slot stays empty.
      exMemNext = exMem;
      idExNext = idEx;
      ifIdNext = ifId;
      trace.push({ cycle, ...cells, annotations });
      memWb = memWbNext;
      exMem = exMemNext;
      idEx = idExNext;
      ifId = ifIdNext;
      continue;
    }

    // ---- 3. EX: executes when the MEM slot is free; branches resolve ----
    if (idEx !== null) {
      const inst = idEx.inst;
      if (cfg.forwarding) {
        for (const [reg, value] of forwardedOperands(idEx, exMem, memWb)) {
          if (idEx.operands.get(reg) !== value) forwards += 1;
          idEx.operands.set(reg, value);
        }
      }

      const a = inst.rs1 !== null ? idEx.operands.get(inst.rs1) ?? 0 : 0;
      const b = inst.rs2 !== null ? idEx.operands.get(inst.rs2) ?? 0 : 0;

      switch (inst.mnemonic) {
        case 'ADD':
          idEx.result = wrap32(a + b);
          break;
        case 'SUB':
          idEx.result = wrap32(a - b);
          break;
        case 'AND':
          idEx.result = wrap32(a & b);
          break;
        case 'OR':
          idEx.result = wrap32(a | b);
          break;
        case 'XOR':
          idEx.result = wrap32(a ^ b);
          break;
        case 'SLT':
          idEx.result = a < b ? 1 : 0;
          break;
        case 'ADDI':
          idEx.result = wrap32(a + inst.imm);
          break;
        default:
          if (inst.usesMemory) {
            idEx.address = wrap32(a + inst.imm);
            if (inst.isStore) {
              // data register value; forwarded like any other operand
              idEx.storeValue = b;
            }
            idEx.latency = cfg.memoryLatency;
            idEx.hit = null;
            if (memoryAccess !== null) {
              const [latency, hit] = memoryAccess(inst.isLoad ? 'LW' : 'SW', idEx.address);
              idEx.latency = Math.max(1, Math.trunc(latency));
              idEx.hit = hit;
            }
            idEx.result = idEx.address;
            idEx.memLeft = idEx.latency;
          } else if (inst.isBranch) {
            branches += 1;
            const actual = inst.mnemonic === 'BEQ' ? a === b : a !== b;
            const predicted = idEx.predicted ?? false;
            const correct = predicted === actual;
            let flushCount = 0;

            if (!correct) {
              mispredicts += 1;
              if (ifId !== null) {
                cells.ID = cell(ifId, 'flush', 'flushed: branch misprediction');
                flushCount += 1;
                ifId = null;
              }
              if (fetched !== null) {
                cells.IF = {
                  kind: 'instr',
                  seq: null,
                  text: fetched.text,
                  state: 'flush',
                  note: 'flushed: branch misprediction',
                };
                flushCount += 1;
                fetched = null;
              }
              redirect = actual ? inst.targetPc ?? 0 : inst.pc + 4;
              fetchEnabledNext = true;
              annotations.push(
                `MISPREDICT ${inst.text} → actually ${actual ? 'taken' : 'not taken'}, ` +
                  `predicted ${predicted ? 'taken' : 'not taken'}: ` +
                  `flushed ${flushCount}, redirect to 0x${redirect.toString(16)}`,
              );
            } else if (actual && ifId !== null) {
              // correct "taken" prediction: ID redirected fetch earlier,
              // so the sequential instruction sitting in ID is wrong-path.
              cells.ID = cell(ifId, 'flush', 'flushed: wrong-path after taken branch');
              flushCount += 1;
              ifId = null;
              annotations.push(
                `${inst.text} taken (predicted correctly): flushed 1 wrong-path instruction`,
              );
            }

            flushes += flushCount;
            if (
              cfg.branchPolicy !== 'static_not_taken' &&
              cfg.branchPolicy !== 'static_taken' &&
              predictor !== null
            ) {
              predictor.update(`B@${inst.pc}`, actual);
            }
            branchEvents.push({
              seq: idEx.seq,
              pc: inst.pc,
              text: inst.text,
              predicted,
              actual,
              correct,
              flushes: flushCount,
              cycle,
            });
            cells.EX = cell(idEx, 'ok', `resolve ${actual ? 'taken' : 'not taken'}`);
          }
          break;
      }

      exMemNext = idEx;
      idEx = null;
    }

    // ---- 4. ID: hazard check, operand read, branch prediction ----
    let idFreed = true;
    if (ifId !== null) {
      if (needsStall(ifId, exMemNext, exMem, memWb, cfg)) {
        hazardStalls += 1;
        const producer = producerText(ifId, exMemNext, exMem, memWb);
        const source = firstSource(ifId);
        cells.ID = cell(
          ifId,
          'stall',
          `RAW hazard: waits for ${producer} (reads x${source})`,
        );
        annotations.push(
          `RAW hazard: ${ifId.inst.text} waits for ${producer} — stall, bubble into EX`,
        );
        idFreed = false;
        ifIdNext = ifId;
      } else {
        for (const reg of ifId.inst.sources) {
          ifId.operands.set(reg, regs[reg]);
        }
        if (ifId.inst.isBranch) {
          let predicted = false;
          if (cfg.branchPolicy === 'static_taken') {
            predicted = true;
          } else if (cfg.branchPolicy === 'static_not_taken') {
            predicted = false;
          } else if (predictor !== null) {
            predicted = predictor.predict(`B@${ifId.inst.pc}`);
          }
          ifId.predicted = predicted;
          if (predicted) {
            redirect = ifId.inst.targetPc ?? 0;
            fetchEnabledNext = true;
            annotations.push(
              `${ifId.inst.text} predicted TAKEN in ID — ` +
                `fetch redirected to 0x${redirect.toString(16)}`,
            );
          }
        }
        idExNext = ifId;
        ifId = null;
      }
    }

    // ---- 5. IF: latch the fetched instruction if the ID slot is free ----
    let latched = false;
    if (ifId === null && idFreed && fetched !== null) {
      if (fetched.isHalt) {
        fetchEnabledNext = false;
      }
      ifIdNext = newLatch(fetched, nextSeq);
      nextSeq += 1;
      latched = true;
    } else if (!idFreed && fetched !== null) {
      cells.IF = {
        kind: 'instr',
        seq: null,
        text: fetched.text,
        state: 'stall',
        note: 'ID stalled — fetch not latched',
      };
    }

    // Commit next state.
    // PC rules: a redirect always wins (and re-enables fetch, in case a
    // wrong-path HALT had disabled it); otherwise the PC only advances
    // when this cycle's fetch was actually latched — a stalled ID holds
    // the PC so the same instruction is fetched again next cycle.
    if (redirect !== null) {
      pc = redirect;
      fetchEnabled = true;
    } else {
      if (latched) pc += 4;
      fetchEnabled = fetchEnabledNext;
    }
    memWb = memWbNext;
    exMem = exMemNext;
    idEx = idExNext;
    ifId = ifIdNext;

    trace.push({ cycle, ...cells, annotations });

    // ---- termination ----
    if (
      ifId === null &&
      idEx === null &&
      exMem === null &&
      memWb === null &&
      (!fetchEnabled || pc >= programEnd)
    ) {
      finished = true;
      break;
    }
  }

  return {
    instructions: program,
    totalCycles: cycle,
    retired,
    cpi: retired ? cycle / retired : null,
    hazardStallCycles: hazardStalls,
    memStallCycles: memStalls,
    stallCycles: hazardStalls + memStalls,
    flushCycles: flushes,
    branches,
    mispredicts,
    forwardedOperands: forwards,
    trace,
    branchEvents,
    memoryEvents,
    retiredInstructions: retiredLog,
    finalRegisters: regs,
    memoryWrites,
    termination: finished ? 'completed' : 'cycle_limit',
    notes,
  };
}

/**
 * Forwarding search at the start of the consumer's EX cycle.
 * Priority: MEM-stage producer (distance 1) then the WB latch (distance 2,
 * or a load that just took its stall). Only registers the consumer reads are
 * forwarded. A not-yet-completed load can never be here: the front end is
 * frozen while MEM is busy, so the consumer cannot execute yet.
 */
function forwardedOperands(
  consumer: Latch,
  exMem: Latch | null,
  memWb: Latch | null,
): Map<number, number> {
  const out = new Map<number, number>();
  const sources = new Set(consumer.inst.sources);
  if (
    exMem !== null &&
    exMem.inst.writesRegister &&
    exMem.inst.rd !== null &&
    sources.has(exMem.inst.rd)
  ) {
    out.set(exMem.inst.rd, exMem.result);
  }
  if (
    memWb !== null &&
    memWb.inst.writesRegister &&
    memWb.inst.rd !== null &&
    sources.has(memWb.inst.rd)
  ) {
    out.set(memWb.inst.rd, memWb.result);
  }
  return out;
}

/**
 * True when `consumer` (sitting in ID) must wait another cycle.
 * Re-evaluated every stalled cycle, so the loop naturally produces the
 * textbook stall counts: 2/1 without forwarding, 1 for load-use with it.
 */
function needsStall(
  consumer: Latch,
  producerDist1: Latch | null, // executed EX this cycle (in MEM next)
  producerDist2: Latch | null, // in MEM this cycle (in WB next)
  _producerDist3: Latch | null, // in WB this cycle: always safe (see below)
  config: PipelineConfig,
): boolean {
  const depends = (producer: Latch | null): boolean =>
    producer !== null &&
    producer.inst.writesRegister &&
    producer.inst.rd !== null &&
    consumer.inst.sources.includes(producer.inst.rd);

  if (depends(producerDist1)) {
    if (!config.forwarding) return true; // must wait until producer reaches WB
    return producerDist1!.inst.isLoad; // load-use: value exists only after MEM
  }
  if (depends(producerDist2)) {
    return !config.forwarding; // wait one more cycle (producer in WB next)
  }
  // dist3 commits this cycle: write-first / read-late makes it safe.
  return false;
}

function producerText(
  consumer: Latch,
  producerDist1: Latch | null,
  producerDist2: Latch | null,
  producerDist3: Latch | null,
): string {
  for (const producer of [producerDist1, producerDist2, producerDist3]) {
    if (producer === null) continue;
    if (
      producer.inst.writesRegister &&
      producer.inst.rd !== null &&
      consumer.inst.sources.includes(producer.inst.rd)
    ) {
      return producer.inst.text;
    }
  }
  return 'an older instruction';
}

function firstSource(latch: Latch): number {
  return latch.inst.sources[0] ?? 0;
}
