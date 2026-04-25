/**
 * Kill-Switch API
 *
 * Instantly halts ALL traffic for a specific tenant. Designed for:
 * - Runaway agent loops detected by anomaly monitoring
 * - Billing emergencies (unexpected spend spike)
 * - Security incidents requiring immediate isolation
 * - Client-requested suspension
 *
 * The kill-switch is checked in-memory first (O(1), no disk I/O) on every
 * request so it activates within a single request cycle — effectively
 * instantaneous. The flag is also persisted to disk so it survives proxy
 * restarts.
 *
 * HTTP API (served by the proxy server):
 *   POST   /v1/tenants/:tenantId/kill    { reason?: string } → activates
 *   DELETE /v1/tenants/:tenantId/kill                        → lifts
 *   GET    /v1/tenants/:tenantId/kill                        → status
 */

import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

export interface KillSwitchEntry {
  tenant_id: string;
  active: boolean;
  reason?: string;
  activated_at?: string;
  activated_by?: string;
  lifted_at?: string;
  lifted_by?: string;
}

export interface KillSwitchStore {
  entries: Record<string, KillSwitchEntry>;
  updated_at: string;
}

export interface ActivateOptions {
  reason?: string;
  activated_by?: string;
}

export interface LiftOptions {
  lifted_by?: string;
}

function resolveRelayplaneDir(): string {
  const homeOverride = process.env['RELAYPLANE_HOME_OVERRIDE'];
  const base = homeOverride ?? os.homedir();
  return path.join(base, '.relayplane');
}

/**
 * KillSwitchManager provides instant tenant traffic halting with persistence.
 *
 * Thread-safety note: this is single-process Node.js, so the in-memory Set
 * is the authoritative source of truth for the hot path. Disk writes are
 * fire-and-forget for persistence across restarts.
 */
export class KillSwitchManager {
  private storePath: string;
  private activeKillSwitches: Set<string> = new Set();
  private entries: Map<string, KillSwitchEntry> = new Map();

  constructor(storePath?: string) {
    const dir = resolveRelayplaneDir();
    this.storePath = storePath ?? path.join(dir, 'kill-switches.json');
    this.load();
  }

  private load(): void {
    if (fs.existsSync(this.storePath)) {
      try {
        const store = JSON.parse(fs.readFileSync(this.storePath, 'utf-8')) as KillSwitchStore;
        for (const [id, entry] of Object.entries(store.entries)) {
          this.entries.set(id, entry);
          if (entry.active) this.activeKillSwitches.add(id);
        }
      } catch {
        // Corrupt store — start fresh
      }
    }
  }

  private persist(): void {
    const dir = path.dirname(this.storePath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

    const store: KillSwitchStore = {
      entries: Object.fromEntries(this.entries),
      updated_at: new Date().toISOString(),
    };

    // Atomic write
    const tmp = this.storePath + '.tmp';
    fs.writeFileSync(tmp, JSON.stringify(store, null, 2));
    fs.renameSync(tmp, this.storePath);
  }

  /**
   * Activate kill-switch for a tenant. Subsequent isActive() checks return
   * true immediately (in-memory). Fire-and-forget persist to disk.
   */
  activate(tenantId: string, options: ActivateOptions = {}): KillSwitchEntry {
    const now = new Date().toISOString();
    const entry: KillSwitchEntry = {
      tenant_id: tenantId,
      active: true,
      reason: options.reason,
      activated_at: now,
      activated_by: options.activated_by,
    };

    // In-memory first (hot path)
    this.activeKillSwitches.add(tenantId);
    this.entries.set(tenantId, entry);

    // Persist (async-ish — but Node.js writeFileSync is sync, this is fast enough)
    try {
      this.persist();
    } catch {
      // Don't throw — in-memory state is authoritative
    }

    return entry;
  }

  /**
   * Lift kill-switch for a tenant. Requests resume immediately.
   * Returns false if no active kill-switch was found.
   */
  lift(tenantId: string, options: LiftOptions = {}): boolean {
    if (!this.activeKillSwitches.has(tenantId)) return false;

    const now = new Date().toISOString();
    const existing = this.entries.get(tenantId);
    const entry: KillSwitchEntry = {
      ...(existing ?? { tenant_id: tenantId }),
      active: false,
      lifted_at: now,
      lifted_by: options.lifted_by,
    };

    this.activeKillSwitches.delete(tenantId);
    this.entries.set(tenantId, entry);

    try {
      this.persist();
    } catch {
      // In-memory state is authoritative
    }

    return true;
  }

  /**
   * Check if a kill-switch is active for a tenant.
   * O(1) — reads only from the in-memory Set.
   */
  isActive(tenantId: string): boolean {
    return this.activeKillSwitches.has(tenantId);
  }

  /** Get the full kill-switch entry for a tenant. */
  getEntry(tenantId: string): KillSwitchEntry | undefined {
    return this.entries.get(tenantId);
  }

  /** List all kill-switch entries (active and historical). */
  listAll(): KillSwitchEntry[] {
    return Array.from(this.entries.values());
  }

  /** List only currently active kill-switches. */
  listActive(): KillSwitchEntry[] {
    return Array.from(this.entries.values()).filter(e => e.active);
  }

  /**
   * Build the response payload for a blocked request.
   * Use with HTTP 429 status and x-relayplane-kill-switch: true header.
   */
  buildBlockedResponse(tenantId: string): object {
    const entry = this.entries.get(tenantId);
    return {
      error: {
        type: 'kill_switch_active',
        message: `All traffic for tenant '${tenantId}' has been suspended.`,
        tenant_id: tenantId,
        activated_at: entry?.activated_at,
        reason: entry?.reason ?? 'No reason provided.',
        contact: 'Contact your RelayPlane administrator to lift the kill-switch.',
      },
    };
  }
}

/** Singleton instance. */
let _instance: KillSwitchManager | undefined;

export function getKillSwitchManager(storePath?: string): KillSwitchManager {
  if (!_instance) _instance = new KillSwitchManager(storePath);
  return _instance;
}

export function resetKillSwitchManager(): void {
  _instance = undefined;
}
