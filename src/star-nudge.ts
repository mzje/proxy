/**
 * RelayPlane Star Nudge
 *
 * After the 50th cumulative proxied request, prints a one-time CLI nudge
 * to stderr encouraging the user to star the GitHub repo.
 *
 * Guarantees:
 *  - Fires exactly once per install (flag written to ~/.relayplane/star-nudge-shown.json)
 *  - Prints to stderr — never pollutes proxy response stdout
 *  - Zero added latency — call checkAndShowStarNudge() *after* forwarding the response
 *  - Never throws — all errors are silently swallowed
 */

import * as fs from 'fs';
import * as path from 'path';
import { getConfigDir } from './config.js';

const STAR_NUDGE_THRESHOLD = 50;
const GITHUB_URL = 'https://github.com/RelayPlane/proxy';

/** Path to the star-nudge-shown flag file */
function getStarNudgeFlagFile(): string {
  return path.join(getConfigDir(), 'star-nudge-shown.json');
}

/** Path to the telemetry event log */
function getTelemetryFile(): string {
  return path.join(getConfigDir(), 'telemetry.jsonl');
}

/** Whether the star nudge has already been shown (checked once at startup) */
let starNudgeAlreadyShown = false;

/**
 * Call this once at proxy startup.
 * Reads the flag file and caches the result so we never re-read it per-request.
 */
export function initStarNudge(): void {
  try {
    const flagPath = getStarNudgeFlagFile();
    if (fs.existsSync(flagPath)) {
      starNudgeAlreadyShown = true;
    }
  } catch {
    // Silently ignore — nudge is non-critical
  }
}

/**
 * Count cumulative requests from the telemetry.jsonl file.
 * Returns 0 on any read/parse error.
 */
export function countTelemetryRequests(): number {
  try {
    const file = getTelemetryFile();
    if (!fs.existsSync(file)) return 0;
    const content = fs.readFileSync(file, 'utf-8');
    // Each non-empty line is one request event
    return content.split('\n').filter(l => l.trim().length > 0).length;
  } catch {
    return 0;
  }
}

/**
 * Write the star-nudge-shown flag so it never fires again.
 */
function markStarNudgeShown(): void {
  try {
    const flagPath = getStarNudgeFlagFile();
    const configDir = getConfigDir();
    if (!fs.existsSync(configDir)) {
      fs.mkdirSync(configDir, { recursive: true });
    }
    fs.writeFileSync(flagPath, JSON.stringify({ shown: true, timestamp: new Date().toISOString() }), 'utf-8');
    starNudgeAlreadyShown = true;
  } catch {
    // Silently ignore
  }
}

/**
 * Print the star nudge to stderr.
 */
function printStarNudge(): void {
  process.stderr.write(
    `\n⭐ Enjoying RelayPlane? Help other devs find it → ${GITHUB_URL}\n\n`
  );
}

/**
 * Check whether the star nudge should fire and, if so, show it.
 *
 * Call this AFTER the proxy response has been forwarded so there is
 * zero added latency on the request path.  This function is intentionally
 * synchronous so it can be fire-and-forgotten without creating a dangling
 * promise.
 *
 * @param requestCount  Optional: pass the current cumulative count if you
 *                      already have it (avoids re-reading the file).
 */
export function checkAndShowStarNudge(requestCount?: number): void {
  // Fast path — already shown, skip all I/O
  if (starNudgeAlreadyShown) return;

  try {
    const count = requestCount ?? countTelemetryRequests();
    if (count >= STAR_NUDGE_THRESHOLD) {
      printStarNudge();
      markStarNudgeShown();
    }
  } catch {
    // Star nudge must never break the proxy
  }
}

// ── Test-seam exports (not part of public API) ────────────────────────────────

/** Reset in-memory flag (used in tests only) */
export function _resetStarNudgeState(): void {
  starNudgeAlreadyShown = false;
}

/** Expose paths for tests */
export { getStarNudgeFlagFile, getTelemetryFile };

/** Expose threshold for tests */
export { STAR_NUDGE_THRESHOLD, GITHUB_URL };
