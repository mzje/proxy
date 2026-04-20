/**
 * Session Intelligence — X-Claude-Code-Session-Id tracking
 *
 * Stores per-session aggregates in ~/.relayplane/sessions.db (SQLite via better-sqlite3).
 * Falls back to an in-memory Map if SQLite is unavailable (e.g. no native bindings).
 *
 * Extracts the X-Claude-Code-Session-Id header from every request;
 * if missing, synthesises a session ID from sha256(hour + model).slice(0,16)
 * prefixed with "syn_".
 *
 * All writes are fire-and-forget; errors are silently swallowed.
 */

import * as crypto from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import type * as http from 'node:http';

export interface SessionEntry {
  id: string;
  started_at: number;
  last_seen_at: number;
  total_cost_usd: number;
  total_tokens_in: number;
  total_tokens_out: number;
  request_count: number;
  session_source: 'claude-code' | 'synthetic';
}

const SESSIONS_SCHEMA_SQL = `
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  started_at INTEGER NOT NULL,
  last_seen_at INTEGER NOT NULL,
  total_cost_usd REAL NOT NULL DEFAULT 0,
  total_tokens_in INTEGER NOT NULL DEFAULT 0,
  total_tokens_out INTEGER NOT NULL DEFAULT 0,
  request_count INTEGER NOT NULL DEFAULT 1,
  session_source TEXT NOT NULL DEFAULT 'synthetic'
);
CREATE INDEX IF NOT EXISTS sessions_last_seen ON sessions(last_seen_at);
`;

let _db: import('better-sqlite3').Database | null | undefined = undefined;
let _upsertStmt: import('better-sqlite3').Statement | null = null;

/** In-memory fallback when SQLite is unavailable. */
const _memStore = new Map<string, SessionEntry>();

function getRelayplaneDir(): string {
  const override = process.env['RELAYPLANE_HOME_OVERRIDE'];
  const base = override ?? os.homedir();
  return path.join(base, '.relayplane');
}

function ensureDir(dir: string): void {
  fs.mkdirSync(dir, { recursive: true });
}

function initDb(): import('better-sqlite3').Database | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const Database = require('better-sqlite3') as typeof import('better-sqlite3');
    const dir = getRelayplaneDir();
    ensureDir(dir);
    const dbPath = path.join(dir, 'sessions.db');
    const db = new Database(dbPath);
    db.pragma('journal_mode = WAL');
    db.exec(SESSIONS_SCHEMA_SQL);
    return db;
  } catch {
    return null;
  }
}

function getDb(): import('better-sqlite3').Database | null {
  if (_db !== undefined) return _db;
  _db = initDb();
  if (_db) {
    _upsertStmt = _db.prepare(`
      INSERT INTO sessions
        (id, started_at, last_seen_at, total_cost_usd, total_tokens_in, total_tokens_out, request_count, session_source)
      VALUES
        (@id, @started_at, @last_seen_at, @total_cost_usd, @total_tokens_in, @total_tokens_out, 1, @session_source)
      ON CONFLICT(id) DO UPDATE SET
        last_seen_at   = excluded.last_seen_at,
        total_cost_usd = total_cost_usd + excluded.total_cost_usd,
        total_tokens_in  = total_tokens_in  + excluded.total_tokens_in,
        total_tokens_out = total_tokens_out + excluded.total_tokens_out,
        request_count  = request_count + 1
    `);
  }
  return _db;
}

/**
 * Extract or synthesise a session ID from an incoming HTTP request.
 *
 * - If the X-Claude-Code-Session-Id header is present, use it (source = 'claude-code').
 * - Otherwise, generate a synthetic ID: sha256(`${hourBucket}:${model}`).slice(0,16)
 *   prefixed with "syn_" (source = 'synthetic').
 */
export function getSessionId(
  req: Pick<http.IncomingMessage, 'headers'>,
  model?: string,
): { sessionId: string; sessionSource: 'claude-code' | 'synthetic' } {
  const headerVal = req.headers['x-claude-code-session-id'];
  const raw = Array.isArray(headerVal) ? headerVal[0] : headerVal;
  const MAX_SESSION_ID_LEN = 128;
  const SESSION_ID_RE = /^[\w\-.:@]+$/;

  if (raw && raw.trim()) {
    const trimmed = raw.trim().slice(0, MAX_SESSION_ID_LEN);
    if (SESSION_ID_RE.test(trimmed)) {
      return { sessionId: trimmed, sessionSource: 'claude-code' };
    }
    // Fall through to synthetic if header is malformed
  }

  // Synthetic: keyed by UTC hour bucket + model
  const now = new Date();
  const hourBucket = `${now.getUTCFullYear()}-${now.getUTCMonth()}-${now.getUTCDate()}-${now.getUTCHours()}`;
  const key = `${hourBucket}:${model ?? 'unknown'}`;
  const hash = crypto.createHash('sha256').update(key).digest('hex').slice(0, 16);
  return { sessionId: `syn_${hash}`, sessionSource: 'synthetic' };
}

/**
 * Upsert a session record, accumulating cost and token counts.
 * Fire-and-forget; never throws. Falls back to in-memory store if SQLite unavailable.
 */
export function upsertSession(
  sessionId: string,
  sessionSource: 'claude-code' | 'synthetic',
  costUsd: number,
  tokensIn: number,
  tokensOut: number,
): void {
  const now = Date.now();
  try {
    const db = getDb();
    if (db && _upsertStmt) {
      _upsertStmt.run({
        id: sessionId,
        started_at: now,
        last_seen_at: now,
        total_cost_usd: costUsd,
        total_tokens_in: tokensIn,
        total_tokens_out: tokensOut,
        session_source: sessionSource,
      });
      return;
    }
  } catch {
    // fall through to in-memory
  }

  // In-memory fallback
  const existing = _memStore.get(sessionId);
  if (existing) {
    existing.last_seen_at = now;
    existing.total_cost_usd += costUsd;
    existing.total_tokens_in += tokensIn;
    existing.total_tokens_out += tokensOut;
    existing.request_count += 1;
  } else {
    _memStore.set(sessionId, {
      id: sessionId,
      started_at: now,
      last_seen_at: now,
      total_cost_usd: costUsd,
      total_tokens_in: tokensIn,
      total_tokens_out: tokensOut,
      request_count: 1,
      session_source: sessionSource,
    });
  }
}

export interface SessionQueryOptions {
  limit?: number;
  days?: number;
}

/**
 * Query sessions from the last N days, most-recent first, paginated.
 * Falls back to in-memory store if SQLite unavailable.
 */
export function getSessions(options: SessionQueryOptions = {}): SessionEntry[] {
  const limit = options.limit ?? 20;
  const days = options.days ?? 7;
  const cutoff = Date.now() - days * 86400000;

  try {
    const db = getDb();
    if (db) {
      const rows = db
        .prepare(
          `SELECT id, started_at, last_seen_at, total_cost_usd, total_tokens_in, total_tokens_out, request_count, session_source
           FROM sessions
           WHERE last_seen_at >= ?
           ORDER BY last_seen_at DESC
           LIMIT ?`,
        )
        .all(cutoff, limit) as SessionEntry[];
      return rows;
    }
  } catch {
    // fall through to in-memory
  }

  // In-memory fallback
  return Array.from(_memStore.values())
    .filter(s => s.last_seen_at >= cutoff)
    .sort((a, b) => b.last_seen_at - a.last_seen_at)
    .slice(0, limit);
}

/**
 * Return sessions active in the last 5 minutes.
 * Falls back to in-memory store if SQLite unavailable.
 */
export function getActiveSessions(): SessionEntry[] {
  const cutoff = Date.now() - 5 * 60 * 1000;

  try {
    const db = getDb();
    if (db) {
      const rows = db
        .prepare(
          `SELECT id, started_at, last_seen_at, total_cost_usd, total_tokens_in, total_tokens_out, request_count, session_source
           FROM sessions
           WHERE last_seen_at >= ?
           ORDER BY last_seen_at DESC`,
        )
        .all(cutoff) as SessionEntry[];
      return rows;
    }
  } catch {
    // fall through to in-memory
  }

  // In-memory fallback
  return Array.from(_memStore.values())
    .filter(s => s.last_seen_at >= cutoff)
    .sort((a, b) => b.last_seen_at - a.last_seen_at);
}

/**
 * Mark stale sessions complete — staleness is computed dynamically via
 * last_seen_at filters, so this is a no-op exposed for API symmetry.
 */
export function markStaleSessionsComplete(): void {
  // No-op: staleness determined dynamically in getActiveSessions()
}

/**
 * Remove a session from the tracker store.
 *
 * Used by the Phase 0 kill switch: when `relayplane status` detects a
 * `stuck_agent` or `token_explosion` anomaly, the CLI prompts the user to
 * kill the offending session. The local proxy's DELETE /v1/sessions/:id
 * handler invokes this to drop the session from both SQLite and the
 * in-memory fallback store. Silent on unknown IDs.
 */
export function killSession(sessionId: string): void {
  try {
    const db = getDb();
    if (db) {
      db.prepare('DELETE FROM sessions WHERE id = ?').run(sessionId);
    }
  } catch {
    /* ignore */
  }
  _memStore.delete(sessionId);
}

/** Exposed for testing — reset singleton state. */
export function _resetStore(): void {
  if (_db) {
    try {
      _db.close();
    } catch {
      /* ignore */
    }
  }
  _db = undefined;
  _upsertStmt = null;
  _memStore.clear();
}
