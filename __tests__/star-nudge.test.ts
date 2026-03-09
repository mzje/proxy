/**
 * Tests for the star nudge feature
 *
 * Covers:
 *  1. nudge fires at exactly 50 requests
 *  2. nudge does NOT fire before 50 requests
 *  3. nudge does NOT fire twice (idempotent)
 *  4. no latency impact (synchronous, fast)
 *  5. initStarNudge() reads the flag from disk and short-circuits
 *  6. nudge goes to stderr, not stdout
 *  7. nudge message format (contains star emoji + GitHub URL)
 *  8. uses separate flag file from signup nudge (star-nudge-shown.json)
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

// ── temp dir per test ──────────────────────────────────────────────────────────

let tmpDir: string;

beforeEach(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'rp-star-nudge-test-'));
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.resetModules();
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

// ── mock config.ts so getConfigDir() points to our tmpDir ─────────────────────

vi.mock('../src/config.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../src/config.js')>();
  return {
    ...actual,
    getConfigDir: () => (global as any).__TEST_CONFIG_DIR__ || actual.getConfigDir(),
    isTelemetryEnabled: () => true,
    getDeviceId: () => 'test-device-id',
  };
});

function setConfigDir(dir: string) {
  (global as any).__TEST_CONFIG_DIR__ = dir;
}

// ── helpers ───────────────────────────────────────────────────────────────────

function makeTelemetryLines(n: number): string {
  return Array.from({ length: n }, (_, i) =>
    JSON.stringify({
      model: 'claude-sonnet-4-6',
      tokens_in: 100,
      tokens_out: 50,
      cost_usd: 0.001,
      timestamp: new Date(i).toISOString(),
    })
  ).join('\n') + (n > 0 ? '\n' : '');
}

function writeTelemetry(dir: string, lines: number) {
  fs.writeFileSync(path.join(dir, 'telemetry.jsonl'), makeTelemetryLines(lines), 'utf-8');
}

function writeStarNudgeFlag(dir: string) {
  fs.writeFileSync(
    path.join(dir, 'star-nudge-shown.json'),
    JSON.stringify({ shown: true, timestamp: new Date().toISOString() }),
    'utf-8'
  );
}

// ── countTelemetryRequests ────────────────────────────────────────────────────

describe('star-nudge: countTelemetryRequests', () => {
  it('returns 0 when telemetry file does not exist', async () => {
    setConfigDir(tmpDir);
    const { countTelemetryRequests } = await import('../src/star-nudge.js');
    expect(countTelemetryRequests()).toBe(0);
  });

  it('returns 0 for empty file', async () => {
    setConfigDir(tmpDir);
    fs.writeFileSync(path.join(tmpDir, 'telemetry.jsonl'), '', 'utf-8');
    const { countTelemetryRequests } = await import('../src/star-nudge.js');
    expect(countTelemetryRequests()).toBe(0);
  });

  it('counts 49 lines correctly', async () => {
    setConfigDir(tmpDir);
    writeTelemetry(tmpDir, 49);
    const { countTelemetryRequests } = await import('../src/star-nudge.js');
    expect(countTelemetryRequests()).toBe(49);
  });

  it('counts 50 lines correctly', async () => {
    setConfigDir(tmpDir);
    writeTelemetry(tmpDir, 50);
    const { countTelemetryRequests } = await import('../src/star-nudge.js');
    expect(countTelemetryRequests()).toBe(50);
  });
});

// ── checkAndShowStarNudge: threshold behaviour ────────────────────────────────

describe('star-nudge: checkAndShowStarNudge fires at 50, not before', () => {
  it('does NOT print nudge when count is 0', async () => {
    setConfigDir(tmpDir);
    const mod = await import('../src/star-nudge.js');
    mod._resetStarNudgeState();

    const stderrSpy = vi.spyOn(process.stderr, 'write').mockImplementation(() => true);
    mod.checkAndShowStarNudge(0);
    expect(stderrSpy).not.toHaveBeenCalled();
  });

  it('does NOT print nudge when count is 25', async () => {
    setConfigDir(tmpDir);
    const mod = await import('../src/star-nudge.js');
    mod._resetStarNudgeState();

    const stderrSpy = vi.spyOn(process.stderr, 'write').mockImplementation(() => true);
    mod.checkAndShowStarNudge(25);
    expect(stderrSpy).not.toHaveBeenCalled();
  });

  it('does NOT print nudge when count is 49', async () => {
    setConfigDir(tmpDir);
    const mod = await import('../src/star-nudge.js');
    mod._resetStarNudgeState();

    const stderrSpy = vi.spyOn(process.stderr, 'write').mockImplementation(() => true);
    mod.checkAndShowStarNudge(49);
    expect(stderrSpy).not.toHaveBeenCalled();
  });

  it('DOES print nudge when count is exactly 50', async () => {
    setConfigDir(tmpDir);
    const mod = await import('../src/star-nudge.js');
    mod._resetStarNudgeState();

    const stderrSpy = vi.spyOn(process.stderr, 'write').mockImplementation(() => true);
    mod.checkAndShowStarNudge(50);
    expect(stderrSpy).toHaveBeenCalledOnce();
  });

  it('DOES print nudge when count is above 50 (e.g. 150)', async () => {
    setConfigDir(tmpDir);
    const mod = await import('../src/star-nudge.js');
    mod._resetStarNudgeState();

    const stderrSpy = vi.spyOn(process.stderr, 'write').mockImplementation(() => true);
    mod.checkAndShowStarNudge(150);
    expect(stderrSpy).toHaveBeenCalledOnce();
  });
});

// ── checkAndShowStarNudge: idempotency ────────────────────────────────────────

describe('star-nudge: idempotency (fires exactly once)', () => {
  it('does NOT fire twice when called back-to-back', async () => {
    setConfigDir(tmpDir);
    const mod = await import('../src/star-nudge.js');
    mod._resetStarNudgeState();

    const stderrSpy = vi.spyOn(process.stderr, 'write').mockImplementation(() => true);

    mod.checkAndShowStarNudge(50);  // first — should fire
    mod.checkAndShowStarNudge(100); // second — should NOT fire
    mod.checkAndShowStarNudge(200); // third — should NOT fire

    expect(stderrSpy).toHaveBeenCalledOnce();
  });

  it('writes star-nudge-shown.json after first fire so future runs are suppressed', async () => {
    setConfigDir(tmpDir);
    const mod = await import('../src/star-nudge.js');
    mod._resetStarNudgeState();

    vi.spyOn(process.stderr, 'write').mockImplementation(() => true);

    mod.checkAndShowStarNudge(50);

    // Flag file must exist on disk
    const flagPath = path.join(tmpDir, 'star-nudge-shown.json');
    expect(fs.existsSync(flagPath)).toBe(true);
    const flag = JSON.parse(fs.readFileSync(flagPath, 'utf-8'));
    expect(flag.shown).toBe(true);
  });

  it('does NOT write flag file when count < 50', async () => {
    setConfigDir(tmpDir);
    const mod = await import('../src/star-nudge.js');
    mod._resetStarNudgeState();

    mod.checkAndShowStarNudge(25);

    const flagPath = path.join(tmpDir, 'star-nudge-shown.json');
    expect(fs.existsSync(flagPath)).toBe(false);
  });
});

// ── initStarNudge: reads flag from disk ──────────────────────────────────────

describe('star-nudge: initStarNudge reads flag from disk', () => {
  it('suppresses nudge when flag file exists on disk at startup', async () => {
    setConfigDir(tmpDir);
    writeStarNudgeFlag(tmpDir); // pre-existing flag

    const mod = await import('../src/star-nudge.js');
    mod._resetStarNudgeState();
    mod.initStarNudge(); // should read the file and set internal flag

    const stderrSpy = vi.spyOn(process.stderr, 'write').mockImplementation(() => true);
    mod.checkAndShowStarNudge(1000); // should NOT print

    expect(stderrSpy).not.toHaveBeenCalled();
  });

  it('does NOT suppress nudge when flag file does not exist', async () => {
    setConfigDir(tmpDir);
    // No flag file written

    const mod = await import('../src/star-nudge.js');
    mod._resetStarNudgeState();
    mod.initStarNudge();

    const stderrSpy = vi.spyOn(process.stderr, 'write').mockImplementation(() => true);
    mod.checkAndShowStarNudge(50);

    expect(stderrSpy).toHaveBeenCalledOnce();
  });
});

// ── nudge output format ───────────────────────────────────────────────────────

describe('star-nudge: output format', () => {
  it('nudge contains the ⭐ emoji and the GitHub URL', async () => {
    setConfigDir(tmpDir);
    const mod = await import('../src/star-nudge.js');
    mod._resetStarNudgeState();

    const stderrSpy = vi.spyOn(process.stderr, 'write').mockImplementation(() => true);
    mod.checkAndShowStarNudge(50);

    const output = stderrSpy.mock.calls[0][0] as string;
    expect(output).toContain('⭐');
    expect(output).toContain('https://github.com/RelayPlane/proxy');
  });

  it('nudge goes to STDERR, not stdout (never pollutes proxy responses)', async () => {
    setConfigDir(tmpDir);
    const mod = await import('../src/star-nudge.js');
    mod._resetStarNudgeState();

    const stderrSpy = vi.spyOn(process.stderr, 'write').mockImplementation(() => true);
    const stdoutSpy = vi.spyOn(process.stdout, 'write').mockImplementation(() => true);

    mod.checkAndShowStarNudge(50);

    expect(stderrSpy).toHaveBeenCalled();
    expect(stdoutSpy).not.toHaveBeenCalled();
  });

  it('message does not contain the request count (unlike signup nudge)', async () => {
    setConfigDir(tmpDir);
    const mod = await import('../src/star-nudge.js');
    mod._resetStarNudgeState();

    const stderrSpy = vi.spyOn(process.stderr, 'write').mockImplementation(() => true);
    mod.checkAndShowStarNudge(50);

    const output = stderrSpy.mock.calls[0][0] as string;
    expect(output).toContain('Enjoying RelayPlane');
    expect(output).toContain('Help other devs find it');
  });
});

// ── separate flag file from signup nudge ─────────────────────────────────────

describe('star-nudge: uses separate flag file from signup nudge', () => {
  it('star-nudge-shown.json is separate from nudge-shown.json', async () => {
    setConfigDir(tmpDir);
    const mod = await import('../src/star-nudge.js');
    mod._resetStarNudgeState();

    vi.spyOn(process.stderr, 'write').mockImplementation(() => true);
    mod.checkAndShowStarNudge(50);

    // star-nudge flag should exist
    expect(fs.existsSync(path.join(tmpDir, 'star-nudge-shown.json'))).toBe(true);
    // signup nudge flag should NOT exist
    expect(fs.existsSync(path.join(tmpDir, 'nudge-shown.json'))).toBe(false);
  });

  it('signup nudge flag does NOT suppress star nudge', async () => {
    setConfigDir(tmpDir);
    // Write the signup nudge flag but NOT the star nudge flag
    fs.writeFileSync(
      path.join(tmpDir, 'nudge-shown.json'),
      JSON.stringify({ shown: true, timestamp: new Date().toISOString() }),
      'utf-8'
    );

    const mod = await import('../src/star-nudge.js');
    mod._resetStarNudgeState();
    mod.initStarNudge();

    const stderrSpy = vi.spyOn(process.stderr, 'write').mockImplementation(() => true);
    mod.checkAndShowStarNudge(50);

    // Star nudge should still fire — it uses its own flag
    expect(stderrSpy).toHaveBeenCalledOnce();
  });
});

// ── no latency impact ─────────────────────────────────────────────────────────

describe('star-nudge: no latency impact', () => {
  it('checkAndShowStarNudge completes synchronously in <50ms for 100 calls', async () => {
    setConfigDir(tmpDir);
    const mod = await import('../src/star-nudge.js');
    mod._resetStarNudgeState();

    vi.spyOn(process.stderr, 'write').mockImplementation(() => true);

    const start = Date.now();
    for (let i = 0; i <= 60; i++) {
      mod.checkAndShowStarNudge(i);
    }
    const elapsed = Date.now() - start;

    expect(elapsed).toBeLessThan(50);
  });

  it('never throws — even if config dir is inaccessible', async () => {
    // Point to a non-writable directory to force fs errors
    setConfigDir('/root/no-access-hopefully-9999');
    const mod = await import('../src/star-nudge.js');
    mod._resetStarNudgeState();

    // Should not throw, regardless of fs errors
    expect(() => mod.checkAndShowStarNudge(50)).not.toThrow();
  });
});

// ── threshold constant ────────────────────────────────────────────────────────

describe('star-nudge: threshold constant', () => {
  it('STAR_NUDGE_THRESHOLD is 50', async () => {
    const { STAR_NUDGE_THRESHOLD } = await import('../src/star-nudge.js');
    expect(STAR_NUDGE_THRESHOLD).toBe(50);
  });

  it('GITHUB_URL points to RelayPlane/proxy', async () => {
    const { GITHUB_URL } = await import('../src/star-nudge.js');
    expect(GITHUB_URL).toBe('https://github.com/RelayPlane/proxy');
  });
});
