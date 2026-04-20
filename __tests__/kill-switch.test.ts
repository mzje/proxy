/**
 * Phase 0 — Kill switch / session intervene tests
 *
 * Task: rp-phase0-kill-switch
 * Acceptance: `killSession(sessionId)` removes a session from the tracker store
 * so that subsequent `getSessions()` / `getActiveSessions()` calls no longer
 * surface it. This is the proxy-side handler the CLI kill prompt will call
 * via DELETE /v1/sessions/:id.
 *
 * Also: a stuck_agent anomaly detail must expose sessionId in its `data`
 * payload so the CLI can prompt the user with the correct session to kill.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as os from 'node:os';
import * as fs from 'node:fs';
import * as path from 'node:path';
import {
  upsertSession,
  getSessions,
  getActiveSessions,
  killSession,
  _resetStore,
} from '../src/session-tracker.js';
import type { AnomalyType } from '../src/anomaly.js';

let testDir = '';
let testCounter = 0;

beforeEach(() => {
  testCounter++;
  testDir = path.join(os.tmpdir(), `rp-kill-switch-${process.pid}-${testCounter}`);
  fs.mkdirSync(testDir, { recursive: true });
  process.env['RELAYPLANE_HOME_OVERRIDE'] = testDir;
  _resetStore();
});

afterEach(() => {
  _resetStore();
  delete process.env['RELAYPLANE_HOME_OVERRIDE'];
  try {
    fs.rmSync(testDir, { recursive: true, force: true });
  } catch {
    /* ignore */
  }
});

describe('killSession', () => {
  it('removes a tracked session so getSessions() no longer returns it', () => {
    upsertSession('sess-kill-me', 'claude-code', 0.42, 1000, 500);
    expect(getSessions().map(s => s.id)).toContain('sess-kill-me');

    killSession('sess-kill-me');

    expect(getSessions().map(s => s.id)).not.toContain('sess-kill-me');
  });

  it('removes a session from the active-session view', () => {
    upsertSession('active-doomed', 'claude-code', 0.05, 500, 100);
    expect(getActiveSessions().map(s => s.id)).toContain('active-doomed');

    killSession('active-doomed');

    expect(getActiveSessions().map(s => s.id)).not.toContain('active-doomed');
  });

  it('leaves unrelated sessions intact', () => {
    upsertSession('sess-keep', 'claude-code', 0.01, 100, 50);
    upsertSession('sess-drop', 'claude-code', 0.01, 100, 50);

    killSession('sess-drop');

    const ids = getSessions().map(s => s.id);
    expect(ids).toContain('sess-keep');
    expect(ids).not.toContain('sess-drop');
  });

  it('does not throw when the session does not exist', () => {
    expect(() => killSession('never-was')).not.toThrow();
  });
});

describe('stuck_agent anomaly type (Phase 0 kill-switch contract)', () => {
  it('is a valid AnomalyType', () => {
    // Compile-time guarantee: "stuck_agent" must be part of the AnomalyType union
    // so the CLI can narrow on `anomaly.type === 'stuck_agent'` and prompt the
    // user to kill the offending session.
    const t: AnomalyType = 'stuck_agent';
    expect(t).toBe('stuck_agent');
  });
});
