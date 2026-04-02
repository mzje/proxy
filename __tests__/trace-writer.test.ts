/**
 * Unit tests for TraceWriter (CAP 3 — Phase 2 Session 2)
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as os from 'node:os';
import * as path from 'node:path';
import * as fs from 'node:fs';
import { TraceWriter, defaultTracesConfig } from '../src/trace-writer.js';

// Detect whether better-sqlite3 native bindings are available in this environment.
// The native binary is loaded lazily on Database construction, so we must try to
// open an in-memory DB to confirm availability.
let hasSqlite = false;
try {
  const { createRequire } = await import('node:module');
  const req = createRequire(import.meta.url);
  const Db = req('better-sqlite3') as (path: string) => unknown;
  Db(':memory:');
  hasSqlite = true;
} catch {
  hasSqlite = false;
}

function makeTempDir(): string {
  const dir = path.join(os.tmpdir(), `rp-trace-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

describe('TraceWriter', () => {
  let dir: string;

  beforeEach(() => {
    TraceWriter.reset();
    dir = makeTempDir();
  });

  afterEach(() => {
    TraceWriter.reset();
    try { fs.rmSync(dir, { recursive: true, force: true }); } catch { /* ignore */ }
  });

  function makeWriter(): TraceWriter {
    return TraceWriter.getInstance({
      ...defaultTracesConfig(),
      directory: dir,
    });
  }

  describe('getInstance / reset', () => {
    it('returns the same singleton instance', () => {
      const a = makeWriter();
      const b = TraceWriter.getInstance();
      expect(a).toBe(b);
    });

    it('reset creates a fresh instance', () => {
      const a = makeWriter();
      TraceWriter.reset();
      const b = makeWriter();
      expect(a).not.toBe(b);
    });
  });

  describe('isEnabled', () => {
    it('returns true by default', () => {
      const tw = makeWriter();
      expect(tw.isEnabled()).toBe(true);
    });

    it('returns false when enabled=false', () => {
      TraceWriter.reset();
      const tw = TraceWriter.getInstance({ ...defaultTracesConfig(), directory: dir, enabled: false });
      expect(tw.isEnabled()).toBe(false);
    });
  });

  describe('write()', () => {
    it('creates a JSONL file for a trace', async () => {
      const tw = makeWriter();
      const sessionId = 'sess-test-1';
      const traceId = 'trace-test-1';

      await tw.write(sessionId, traceId, {
        eventType: 'request.start',
        payload: { model: 'claude-opus-4-6', messageCount: 3 },
      });

      // Give the sync write a tick to complete
      await new Promise(r => setImmediate(r));

      // Find the file
      const sessionDir = path.join(dir, sessionId);
      expect(fs.existsSync(sessionDir)).toBe(true);
      const dateDirs = fs.readdirSync(sessionDir);
      expect(dateDirs.length).toBeGreaterThan(0);
      const dateDir = dateDirs[0]!;
      const traceFile = path.join(sessionDir, dateDir, `${traceId}.jsonl`);
      expect(fs.existsSync(traceFile)).toBe(true);

      const content = fs.readFileSync(traceFile, 'utf-8');
      const lines = content.trim().split('\n').filter(Boolean);
      expect(lines.length).toBeGreaterThanOrEqual(1);

      const lastEvent = JSON.parse(lines[lines.length - 1]!);
      expect(lastEvent.eventType).toBe('request.start');
      expect(lastEvent.traceId).toBe(traceId);
      expect(lastEvent.sessionId).toBe(sessionId);
      expect(lastEvent.payload.model).toBe('claude-opus-4-6');
    });

    it('appends multiple events to the same file', async () => {
      const tw = makeWriter();
      const sessionId = 'sess-multi';
      const traceId = 'trace-multi';

      await tw.write(sessionId, traceId, { eventType: 'request.start', payload: { model: 'test' } });
      await tw.write(sessionId, traceId, { eventType: 'request.end', durationMs: 500, payload: { costUsd: 0.001 } });

      await new Promise(r => setImmediate(r));

      const sessionDir = path.join(dir, sessionId);
      const dateDir = fs.readdirSync(sessionDir)[0]!;
      const traceFile = path.join(sessionDir, dateDir, `${traceId}.jsonl`);
      const content = fs.readFileSync(traceFile, 'utf-8');
      const lines = content.trim().split('\n').filter(l => {
        try { JSON.parse(l); return true; } catch { return false; }
      });
      // At minimum: session.start + request.start + request.end
      expect(lines.length).toBeGreaterThanOrEqual(2);
    });

    it('does nothing when disabled', async () => {
      TraceWriter.reset();
      const tw = TraceWriter.getInstance({ ...defaultTracesConfig(), directory: dir, enabled: false });
      await tw.write('sess', 'trace', { eventType: 'request.start', payload: {} });
      await new Promise(r => setImmediate(r));
      expect(fs.existsSync(path.join(dir, 'sess'))).toBe(false);
    });
  });

  describe('finalizeTrace()', () => {
    it.skipIf(!hasSqlite)('updates the SQLite index with cost and duration', async () => {
      const tw = makeWriter();
      const sessionId = 'sess-fin';
      const traceId = 'trace-fin';

      await tw.write(sessionId, traceId, {
        eventType: 'request.start',
        payload: { model: 'claude-opus-4-6' },
      });
      await new Promise(r => setImmediate(r));

      await tw.finalizeTrace(traceId, sessionId, {
        costUsd: 0.0123,
        modelUsed: 'claude-opus-4-6',
        durationMs: 1234,
      });

      const traces = tw.getRecentTraces(10);
      const found = (traces as Record<string, unknown>[]).find(t => t['trace_id'] === traceId);
      expect(found).toBeDefined();
      expect(found?.['cost_usd']).toBeCloseTo(0.0123, 4);
      expect(found?.['duration_ms']).toBe(1234);
    });

    it('does not throw when db is unavailable', async () => {
      const tw = makeWriter();
      // finalizeTrace should silently succeed even without SQLite
      await expect(
        tw.finalizeTrace('trace-nodb', 'sess-nodb', { durationMs: 10, modelUsed: 'x' })
      ).resolves.not.toThrow();
    });
  });

  describe('getRecentTraces()', () => {
    it('returns empty array when no traces exist', () => {
      const tw = makeWriter();
      const traces = tw.getRecentTraces(10);
      expect(traces).toEqual([]);
    });

    it.skipIf(!hasSqlite)('returns traces after write + finalize', async () => {
      const tw = makeWriter();
      const sessionId = 'sess-list';

      await tw.write(sessionId, 'trace-a', { eventType: 'request.start', payload: { model: 'm1' } });
      await new Promise(r => setImmediate(r));
      await tw.finalizeTrace('trace-a', sessionId, { costUsd: 0.001, modelUsed: 'm1', durationMs: 100 });

      await tw.write(sessionId, 'trace-b', { eventType: 'request.start', payload: { model: 'm2' } });
      await new Promise(r => setImmediate(r));
      await tw.finalizeTrace('trace-b', sessionId, { costUsd: 0.002, modelUsed: 'm2', durationMs: 200 });

      const traces = tw.getRecentTraces(10);
      expect(traces.length).toBe(2);
    });
  });

  describe('getTraceEvents()', () => {
    it.skipIf(!hasSqlite)('returns events from the trace JSONL file', async () => {
      const tw = makeWriter();
      const sessionId = 'sess-events';
      const traceId = 'trace-events';

      await tw.write(sessionId, traceId, { eventType: 'request.start', payload: { model: 'x' } });
      await tw.write(sessionId, traceId, { eventType: 'request.end', durationMs: 10, payload: {} });
      await new Promise(r => setImmediate(r));
      await tw.finalizeTrace(traceId, sessionId, { durationMs: 10, modelUsed: 'x' });

      const events = tw.getTraceEvents(traceId);
      expect(events.length).toBeGreaterThanOrEqual(2);
      const types = events.map(e => e.eventType);
      expect(types).toContain('request.start');
    });

    it('returns empty array for unknown traceId', () => {
      const tw = makeWriter();
      expect(tw.getTraceEvents('nonexistent')).toEqual([]);
    });
  });

  describe('getSessionGraph()', () => {
    it('returns in-memory graph for active session', async () => {
      const tw = makeWriter();
      const sessionId = 'sess-graph';

      await tw.write(sessionId, 'trace-g1', { eventType: 'request.start', payload: { model: 'gpt' } });
      await new Promise(r => setImmediate(r));

      const graph = tw.getSessionGraph(sessionId);
      expect(graph).not.toBeNull();
      expect(graph?.sessionId).toBe(sessionId);
      expect(graph?.nodes.length).toBeGreaterThan(0);
    });

    it('returns null for unknown session', () => {
      const tw = makeWriter();
      expect(tw.getSessionGraph('unknown-session')).toBeNull();
    });
  });

  describe('pruneOldFiles()', () => {
    it('does not throw even when db is unavailable', () => {
      const tw = makeWriter();
      expect(() => tw.pruneOldFiles()).not.toThrow();
    });

    it.skipIf(!hasSqlite)('keeps traces within retention period', async () => {
      TraceWriter.reset();
      const tw = TraceWriter.getInstance({ ...defaultTracesConfig(), directory: dir, retentionDays: 1 });

      const sessionId = 'sess-prune';
      const traceId = 'trace-prune';
      await tw.write(sessionId, traceId, { eventType: 'request.start', payload: {} });
      await new Promise(r => setImmediate(r));
      await tw.finalizeTrace(traceId, sessionId, { durationMs: 10, modelUsed: 'x' });

      tw.pruneOldFiles();
      const traces = tw.getRecentTraces(10);
      expect(traces.length).toBe(1);
    });
  });

  describe('export()', () => {
    it('returns a string for jsonl format', async () => {
      const tw = makeWriter();
      const output = await tw.export({ format: 'jsonl' });
      expect(typeof output).toBe('string');
    });

    it('returns a string containing # for markdown format', async () => {
      const tw = makeWriter();
      const output = await tw.export({ format: 'markdown' });
      expect(typeof output).toBe('string');
      expect(output).toContain('#');
    });

    it.skipIf(!hasSqlite)('exports finalized traces to jsonl', async () => {
      const tw = makeWriter();
      const sessionId = 'sess-exp';
      await tw.write(sessionId, 'trace-e1', { eventType: 'request.start', payload: { model: 'm' } });
      await new Promise(r => setImmediate(r));
      await tw.finalizeTrace('trace-e1', sessionId, { costUsd: 0.005, modelUsed: 'm', durationMs: 50 });

      const output = await tw.export({ format: 'jsonl' });
      expect(output).toContain('trace-e1');
    });

    it.skipIf(!hasSqlite)('exports finalized traces to markdown', async () => {
      const tw = makeWriter();
      const sessionId = 'sess-md';
      await tw.write(sessionId, 'trace-md1', { eventType: 'request.start', payload: { model: 'z' } });
      await new Promise(r => setImmediate(r));
      await tw.finalizeTrace('trace-md1', sessionId, { costUsd: 0.001, modelUsed: 'z', durationMs: 99 });

      const output = await tw.export({ format: 'markdown' });
      expect(output).toContain('# RelayPlane Trace Export');
      expect(output).toContain('|');
    });
  });

  describe('replay()', () => {
    it('rejects when storeFullRequests=false', async () => {
      const tw = makeWriter();
      await expect(tw.replay('any-trace-id')).rejects.toThrow(/storeFullRequests/);
    });
  });
});
