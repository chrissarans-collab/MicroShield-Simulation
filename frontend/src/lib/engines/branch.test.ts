/**
 * Vitest suite for the branch prediction engine.
 *
 * Asserts the SAME shared fixtures as the Python suite
 * (fixtures/branch_predictor.json), guaranteeing cross-language parity, plus
 * TS-specific parser and validation tests.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  DEFAULT_BRANCH_CONFIG,
  type BranchConfig,
  type BranchEvent,
  fnv1a32,
  parseBranchSequence,
  simulateBranches,
  validateBranchConfig,
} from './branch';

interface FixtureCase {
  name: string;
  verified: 'hand' | 'reference';
  input: {
    sequence: Array<{ label: string; outcome: boolean }>;
    config: Record<string, unknown>;
  };
  expected: {
    total: number;
    correct: number;
    mispredicted: number;
    accuracy: number;
    penalty_cycles: number;
    predictions: boolean[];
    state_before?: string[];
    state_after?: string[];
    label_stats?: Array<{ label: string; total: number; correct: number }>;
  };
}

const fixtureUrl = new URL('../../../../fixtures/branch_predictor.json', import.meta.url);
const fixtures: { cases: FixtureCase[] } = JSON.parse(readFileSync(fileURLToPath(fixtureUrl), 'utf-8'));

function configFrom(raw: Record<string, unknown>): BranchConfig {
  return {
    predictor: (raw.predictor as BranchConfig['predictor']) ?? DEFAULT_BRANCH_CONFIG.predictor,
    initialPrediction:
      (raw.initial_prediction as BranchConfig['initialPrediction']) ??
      DEFAULT_BRANCH_CONFIG.initialPrediction,
    initialCounter: (raw.initial_counter as number) ?? DEFAULT_BRANCH_CONFIG.initialCounter,
    initialGhr: (raw.initial_ghr as number) ?? DEFAULT_BRANCH_CONFIG.initialGhr,
    historyBits: (raw.history_bits as number) ?? DEFAULT_BRANCH_CONFIG.historyBits,
    mispredictionPenalty:
      (raw.misprediction_penalty as number) ?? DEFAULT_BRANCH_CONFIG.mispredictionPenalty,
  };
}

describe('shared fixtures (parity with Python reference)', () => {
  for (const c of fixtures.cases) {
    it(c.name, () => {
      const sequence: BranchEvent[] = c.input.sequence.map((e) => ({ ...e }));
      const result = simulateBranches(sequence, configFrom(c.input.config));
      expect(result.total).toBe(c.expected.total);
      expect(result.correct).toBe(c.expected.correct);
      expect(result.mispredicted).toBe(c.expected.mispredicted);
      expect(result.accuracy).toBeCloseTo(c.expected.accuracy, 4);
      expect(result.penaltyCycles).toBe(c.expected.penalty_cycles);
      expect(result.steps.map((s) => s.prediction)).toEqual(c.expected.predictions);
      if (c.expected.state_before) {
        expect(result.steps.map((s) => s.stateBefore)).toEqual(c.expected.state_before);
      }
      if (c.expected.state_after) {
        expect(result.steps.map((s) => s.stateAfter)).toEqual(c.expected.state_after);
      }
      if (c.expected.label_stats) {
        const got = new Map(result.labelStats.map((ls) => [ls.label, [ls.total, ls.correct]]));
        for (const want of c.expected.label_stats) {
          expect(got.get(want.label)).toEqual([want.total, want.correct]);
        }
      }
    });
  }
});

describe('fnv1a32', () => {
  it('matches published FNV-1a 32-bit vectors', () => {
    expect(fnv1a32('')).toBe(0x811c9dc5);
    expect(fnv1a32('a')).toBe(0xe40c292c);
    expect(fnv1a32('foobar')).toBe(0xbf9cf968);
  });
});

describe('simulateBranches', () => {
  it('reports the gshare entry that was actually updated (not the post-shift index)', () => {
    const result = simulateBranches(
      [{ label: 'B1', outcome: true }],
      { ...DEFAULT_BRANCH_CONFIG, predictor: 'ghr', historyBits: 2, initialCounter: 2, initialGhr: 0 },
    );
    // Table index of B1 with ghr=0 is 0; after update the history is 01 but the
    // reported state must still describe entry 0, now counter 3 (ST).
    expect(result.steps[0].stateAfter).toBe('3 ST ghr=01');
    expect(result.steps[0].detail.tableIndex).toBe(0);
  });

  it('saturates the two-bit counter at 3', () => {
    const seq: BranchEvent[] = Array.from({ length: 5 }, () => ({ label: 'B1', outcome: true }));
    const result = simulateBranches(seq, { ...DEFAULT_BRANCH_CONFIG, initialCounter: 3 });
    expect(result.correct).toBe(5);
    expect(result.steps[4].stateAfter).toBe('3 ST');
  });

  it('uses the configured initial state for unseen labels', () => {
    const result = simulateBranches(
      [{ label: 'brand_new', outcome: true }],
      { ...DEFAULT_BRANCH_CONFIG, predictor: 'one_bit', initialPrediction: 'not_taken' },
    );
    expect(result.steps[0].prediction).toBe(false);
    expect(result.steps[0].stateBefore).toBe('init(N)');
  });

  it('handles an empty sequence safely', () => {
    const result = simulateBranches([]);
    expect(result.total).toBe(0);
    expect(result.accuracy).toBe(0);
    expect(result.penaltyCycles).toBe(0);
    expect(result.steps).toEqual([]);
  });

  it('applies the configured penalty', () => {
    const seq: BranchEvent[] = [
      { label: 'B1', outcome: false },
      { label: 'B1', outcome: false },
    ];
    const a = simulateBranches(seq, { ...DEFAULT_BRANCH_CONFIG, predictor: 'always_taken', mispredictionPenalty: 1 });
    const b = simulateBranches(seq, { ...DEFAULT_BRANCH_CONFIG, predictor: 'always_taken', mispredictionPenalty: 5 });
    expect(a.penaltyCycles).toBe(2);
    expect(b.penaltyCycles).toBe(10);
  });
});

describe('parseBranchSequence', () => {
  it('parses plain T/N tokens into one shared label', () => {
    const events = parseBranchSequence('T n 1 0 taken not');
    expect(events.map((e) => e.outcome)).toEqual([true, false, true, false, true, false]);
    expect(events.every((e) => e.label === 'B1')).toBe(true);
  });

  it('parses named labels', () => {
    const events = parseBranchSequence('loop:T if:N loop:N');
    expect(events.map((e) => [e.label, e.outcome])).toEqual([
      ['loop', true],
      ['if', false],
      ['loop', false],
    ]);
  });

  it('rejects garbage outcomes and labels', () => {
    expect(() => parseBranchSequence('T N maybe')).toThrow();
    expect(() => parseBranchSequence('bad-label:T')).toThrow();
  });
});

describe('validateBranchConfig', () => {
  it('accepts the default config', () => {
    expect(validateBranchConfig(DEFAULT_BRANCH_CONFIG)).toEqual([]);
  });

  it('rejects out-of-range values', () => {
    expect(validateBranchConfig({ ...DEFAULT_BRANCH_CONFIG, initialCounter: 9 })).not.toEqual([]);
    expect(validateBranchConfig({ ...DEFAULT_BRANCH_CONFIG, historyBits: 0 })).not.toEqual([]);
    expect(validateBranchConfig({ ...DEFAULT_BRANCH_CONFIG, mispredictionPenalty: -1 })).not.toEqual([]);
  });
});
