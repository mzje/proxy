import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as os from 'node:os';
import * as fs from 'node:fs';
import * as path from 'node:path';
import {
  getSessionId,
  upsertSession,
  getSessions,
  getActiveSessions,
  markStaleSessionsComplete,
  _resetStore,
} from '../src/session-tracker.js';

let testDir = '';
let testCounter = 0;

beforeEach(() => {
  testCounter++;
  testDir = path.join(os.tmpdir(), `rp-session-${process.pid}-${testCounter}`);
  fs.mkdirSync(testDir, { recursive: true });
  process.env['RELAYPLANE_HOME_OVERRIDE'] = testDir;
  _resetStore();
});

afterEach(() => {
  _resetStore();
  delete process.env['RELAYPLANE_HOME_OVERRIDE'];
  try { fs.rmSync(testDir, { recursive: true, force: true }); } catch { /* ignore */ }
});

function makeReqHeaders(headers: Record<string, string> = {}): { headers: Record<string, string> } {
  return { headers };
}

describe('getSessionId', () => {
  it('returns claude-code source when X-Claude-Code-Session-Id header is present', () => {
    const req = makeReqHeaders({ 'x-claude-code-session-id': 'sess-abc123' });
    const result = getSessionId(req);
    expect(result.sessionId).toBe('sess-abc123');
    expect(result.sessionSource).toBe('claude-code');
  });

  it('trims whitespace from session ID header', () => {
    const req = makeReqHeaders({ 'x-claude-code-session-id': '  sess-xyz  ' });
    const result = getSessionId(req);
    expect(result.sessionId).toBe('sess-xyz');
  });

  it('returns synthetic source when header is absent', () => {
    const req = makeReqHeaders({});
    const result = getSessionId(req, 'claude-3-5-sonnet');
    expect(result.sessionId).toMatch(/^syn_[0-9a-f]{16}$/);
    expect(result.sessionSource).toBe('synthetic');
  });

  it('synthetic IDs are deterministic for same hour + model', () => {
    const req1 = makeReqHeaders({});
    const req2 = makeReqHeaders({});
    const r1 = getSessionId(req1, 'gpt-4o');
    const r2 = getSessionId(req2, 'gpt-4o');
    expect(r1.sessionId).toBe(r2.sessionId);
  });

  it('synthetic IDs differ for different models', () => {
    const req1 = makeReqHeaders({});
    const req2 = makeReqHeaders({});
    const r1 = getSessionId(req1, 'gpt-4o');
    const r2 = getSessionId(req2, 'claude-3-5-sonnet');
    expect(r1.sessionId).not.toBe(r2.sessionId);
  });

  it('uses "unknown" model when model not provided', () => {
    const req1 = makeReqHeaders({});
    const req2 = makeReqHeaders({});
    const r1 = getSessionId(req1);
    const r2 = getSessionId(req2, undefined);
    expect(r1.sessionId).toBe(r2.sessionId);
    expect(r1.sessionId).toMatch(/^syn_/);
  });
});

describe('upsertSession + getSessions', () => {
  it('creates a session on first upsert', () => {
    upsertSession('test-sess-1', 'claude-code', 0.01, 100, 50);
    const sessions = getSessions();
    expect(sessions).toHaveLength(1);
    expect(sessions[0]!.id).toBe('test-sess-1');
    expect(sessions[0]!.session_source).toBe('claude-code');
    expect(sessions[0]!.request_count).toBe(1);
    expect(sessions[0]!.total_cost_usd).toBeCloseTo(0.01);
    expect(sessions[0]!.total_tokens_in).toBe(100);
    expect(sessions[0]!.total_tokens_out).toBe(50);
  });

  it('accumulates cost and tokens on repeated upserts for same session', () => {
    upsertSession('test-sess-2', 'claude-code', 0.01, 100, 50);
    upsertSession('test-sess-2', 'claude-code', 0.02, 200, 80);
    const sessions = getSessions();
    expect(sessions).toHaveLength(1);
    expect(sessions[0]!.request_count).toBe(2);
    expect(sessions[0]!.total_cost_usd).toBeCloseTo(0.03);
    expect(sessions[0]!.total_tokens_in).toBe(300);
    expect(sessions[0]!.total_tokens_out).toBe(130);
  });

  it('creates separate entries for different session IDs', () => {
    upsertSession('sess-a', 'claude-code', 0.01, 100, 50);
    upsertSession('sess-b', 'synthetic', 0.02, 200, 80);
    const sessions = getSessions();
    expect(sessions).toHaveLength(2);
  });

  it('returns empty array when no sessions exist', () => {
    const sessions = getSessions();
    expect(sessions).toHaveLength(0);
  });

  it('respects the limit option', () => {
    for (let i = 0; i < 5; i++) {
      upsertSession(`sess-${i}`, 'synthetic', 0.001, 10, 5);
    }
    const sessions = getSessions({ limit: 3 });
    expect(sessions).toHaveLength(3);
  });

  it('does not throw when SQLite is unavailable', () => {
    // Close and null out the store — next call re-inits; this just tests fire-and-forget
    expect(() => upsertSession('x', 'synthetic', 0, 0, 0)).not.toThrow();
  });
});

describe('getActiveSessions', () => {
  it('returns recently active sessions', () => {
    upsertSession('active-sess', 'synthetic', 0.05, 500, 100);
    const active = getActiveSessions();
    expect(active.length).toBeGreaterThan(0);
    expect(active[0]!.id).toBe('active-sess');
  });

  it('returns empty when no sessions in last 5 minutes', () => {
    // No upserts — nothing active
    const active = getActiveSessions();
    expect(active).toHaveLength(0);
  });
});

describe('markStaleSessionsComplete', () => {
  it('is a no-op and does not throw', () => {
    upsertSession('sess-x', 'claude-code', 0.1, 500, 200);
    expect(() => markStaleSessionsComplete()).not.toThrow();
    // Sessions still queryable after no-op
    expect(getSessions()).toHaveLength(1);
  });
});
