/**
 * Vitest suite for the 5-stage pipeline engine.
 *
 * Asserts the SAME shared fixtures as the Python suite
 * (fixtures/pipeline.json) — cycle counts, stall/flush counts and traces
 * derived on paper from the documented model — guaranteeing cross-language
 * parity, plus parser, termination and validation tests mirroring pytest.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  type BranchPolicy,
  type PipelineConfig,
  type TraceRow,
  PipelineParseError,
  parseProgram,
  simulatePipeline,
} from './pipeline';

interface FixtureCase {
  name: string;
  verified: 'hand' | 'reference';
  program: string;
  config: Record<string, unknown>;
  expected: {
    total_cycles: number;
    retired: number;
    cpi?: number | null;
    hazard_stall_cycles?: number;
    mem_stall_cycles?: number;
    flush_cycles?: number;
    mispredicts?: number;
    branches?: number;
    termination: string;
    trace_texts?: string[][];
    final_registers?: number[];
    branch_events?: Array<Record<string, unknown>>;
    flush_cells?: Array<{ cycle: number; stage: 'IF' | 'ID' | 'EX' | 'MEM' | 'WB' }>;
    retired_texts?: string[];
    memory_events?: Array<Record<string, unknown>>;
    memory_writes?: Array<{ address: number; value: number }>;
  };
}

const fixtureUrl = new URL('../../../../fixtures/pipeline.json', import.meta.url);
const fixtures: { cases: FixtureCase[] } = JSON.parse(readFileSync(fileURLToPath(fixtureUrl), 'utf-8'));

function configFrom(raw: Record<string, unknown>): Partial<PipelineConfig> {
  return {
    forwarding: (raw.forwarding as boolean | undefined) ?? true,
    branchPolicy: ((raw.branch_policy as BranchPolicy | undefined) ?? 'static_not_taken') as BranchPolicy,
    memoryLatency: (raw.memory_latency as number | undefined) ?? 1,
    initialCounter: (raw.initial_counter as number | undefined) ?? 2,
    initialPrediction: (raw.initial_prediction as 'taken' | 'not_taken' | undefined) ?? 'taken',
    historyBits: (raw.history_bits as number | undefined) ?? 4,
    initialGhr: (raw.initial_ghr as number | undefined) ?? 0,
    maxCycles: (raw.max_cycles as number | undefined) ?? 5000,
  };
}

function stageTexts(rows: TraceRow[]): string[][] {
  return rows.map((row) =>
    (['IF', 'ID', 'EX', 'MEM', 'WB'] as const).map((stage) =>
      row[stage].kind === 'instr' ? row[stage].text : '',
    ),
  );
}

function assertCase(c: FixtureCase) {
  const result = simulatePipeline(c.program, configFrom(c.config));
  const expected = c.expected;

  expect(result.totalCycles).toBe(expected.total_cycles);
  expect(result.retired).toBe(expected.retired);
  if ('cpi' in expected) {
    if (expected.cpi === null || expected.cpi === undefined) {
      expect(result.cpi).toBeNull();
    } else {
      expect(result.cpi).toBeCloseTo(expected.cpi, 4);
    }
  }
  if (expected.hazard_stall_cycles !== undefined) {
    expect(result.hazardStallCycles).toBe(expected.hazard_stall_cycles);
  }
  if (expected.mem_stall_cycles !== undefined) {
    expect(result.memStallCycles).toBe(expected.mem_stall_cycles);
  }
  if (expected.flush_cycles !== undefined) {
    expect(result.flushCycles).toBe(expected.flush_cycles);
  }
  if (expected.mispredicts !== undefined) {
    expect(result.mispredicts).toBe(expected.mispredicts);
  }
  if (expected.branches !== undefined) {
    expect(result.branches).toBe(expected.branches);
  }
  expect(result.termination).toBe(expected.termination);

  if (expected.trace_texts) {
    expect(stageTexts(result.trace)).toEqual(expected.trace_texts);
  }
  if (expected.final_registers) {
    expect(result.finalRegisters).toEqual(expected.final_registers);
  }
  if (expected.branch_events) {
    expect(result.branchEvents.length).toBe(expected.branch_events.length);
    result.branchEvents.forEach((got, i) => {
      const want = expected.branch_events![i];
      for (const key of Object.keys(want)) {
        expect(got[key as keyof typeof got]).toEqual(want[key]);
      }
    });
  }
  if (expected.flush_cells) {
    for (const cell of expected.flush_cells) {
      const row = result.trace[cell.cycle - 1];
      expect(row[cell.stage].state).toBe('flush');
    }
  }
  if (expected.retired_texts) {
    expect(result.retiredInstructions.map((r) => r.text)).toEqual(expected.retired_texts);
  }
  if (expected.memory_events) {
    expect(result.memoryEvents.length).toBe(expected.memory_events.length);
    expected.memory_events.forEach((want, i) => {
      const got = result.memoryEvents[i];
      for (const key of Object.keys(want)) {
        expect(got[key as keyof typeof got]).toEqual(want[key as keyof typeof got]);
      }
    });
  }
  if (expected.memory_writes) {
    expect(
      result.memoryWrites.map((w) => ({ address: w.address, value: w.value })),
    ).toEqual(expected.memory_writes);
  }
}

describe('shared fixtures (parity with Python reference)', () => {
  for (const c of fixtures.cases) {
    it(c.name, () => {
      assertCase(c);
    });
  }
});

describe('CPI and termination', () => {
  it('never divides by zero', () => {
    const empty = simulatePipeline('');
    expect(empty.retired).toBe(0);
    expect(empty.cpi).toBeNull();
    const single = simulatePipeline('ADDI x1, x0, 5');
    expect(single.cpi).toBeCloseTo(single.totalCycles / single.retired, 9);
  });

  it('hits the cycle limit on an infinite loop', () => {
    const result = simulatePipeline('loop: BEQ x0, x0, loop', { maxCycles: 50 });
    expect(result.termination).toBe('cycle_limit');
    expect(result.totalCycles).toBe(50);
    expect(result.mispredicts).toBeGreaterThanOrEqual(1);
  });
});

describe('parser', () => {
  it('accepts comments, hex and labels', () => {
    const program = parseProgram(
      [
        '# leading comment',
        'ADDI x1, x0, 0x10   ; trailing comment',
        'start: ADD x2, x1, x1',
        'BEQ x1, x1, start   # backward branch',
        'HALT',
      ].join('\n'),
    );
    expect(program.map((i) => i.mnemonic)).toEqual(['ADDI', 'ADD', 'BEQ', 'HALT']);
    expect(program[0].imm).toBe(16);
    expect(program[2].targetPc).toBe(4);
    expect(program[2].imm).toBe(4 - 8); // pc-relative displacement
  });

  it('reports the offending line number', () => {
    let caught: unknown;
    try {
      parseProgram('NOP\nFOO x1, x2, x3');
    } catch (err) {
      caught = err;
    }
    expect(caught).toBeInstanceOf(PipelineParseError);
    expect((caught as PipelineParseError).line).toBe(2);

    expect(() => parseProgram('ADD x1, x99, x2')).toThrow(PipelineParseError);
    expect(() => parseProgram('BEQ x1, x2, nowhere')).toThrow(PipelineParseError);
    expect(() => parseProgram('dup: NOP\ndup: HALT')).toThrow(PipelineParseError);
    expect(() => parseProgram('LW x1, x2')).toThrow(PipelineParseError);
  });

  it('points a trailing label at an implicit HALT', () => {
    const program = parseProgram('ADDI x1, x0, 1\nend:');
    expect(program[program.length - 1].isHalt).toBe(true);
  });
});

describe('configuration', () => {
  it('uses initial registers', () => {
    const result = simulatePipeline('ADD x1, x2, x0', {}, { 2: 21 });
    expect(result.finalRegisters[1]).toBe(21);
  });

  it('rejects invalid values', () => {
    expect(() => simulatePipeline('HALT', { memoryLatency: 0 })).toThrow();
    expect(() =>
      simulatePipeline('HALT', { branchPolicy: 'psychic' as BranchPolicy }),
    ).toThrow();
    expect(() => simulatePipeline('HALT', { maxCycles: 5 })).toThrow();
  });

  it('dynamic predictor does not mispredict more than static on a loop', () => {
    const loop = [
      'ADDI x1, x0, 3',
      'loop: ADDI x1, x1, -1',
      'BEQ x1, x0, done',
      'BEQ x0, x0, loop',
      'done: HALT',
    ].join('\n');
    const staticResult = simulatePipeline(loop, { branchPolicy: 'static_not_taken' });
    const twoBit = simulatePipeline(loop, { branchPolicy: 'two_bit' });
    expect(staticResult.termination).toBe('completed');
    expect(twoBit.termination).toBe('completed');
    expect(twoBit.mispredicts).toBeLessThanOrEqual(staticResult.mispredicts);
  });
});

describe('memory model', () => {
  it('records latency, addresses and store data', () => {
    const result = simulatePipeline('ADDI x1, x0, 4\nSW x1, 4(x0)\nLW x2, 4(x0)\nHALT', {
      memoryLatency: 2,
    });
    const ops = result.memoryEvents.map((e) => [e.op, e.address, e.latency] as const);
    expect(ops).toEqual([
      ['SW', 4, 2],
      ['LW', 4, 2],
    ]);
    expect(result.memStallCycles).toBeGreaterThan(0); // latency 2 freezes the front end
    expect(result.finalRegisters[2]).toBe(4); // load reads the value just stored
    expect(
      result.memoryWrites.map((w) => ({ address: w.address, value: w.value })),
    ).toEqual([{ address: 4, value: 4 }]);
  });
});
