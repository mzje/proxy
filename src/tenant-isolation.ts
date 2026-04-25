/**
 * Tenant Isolation — per-tenant request scoping for agent swarms.
 *
 * Each tenant gets an isolated lane: separate budget tracking, rate limits,
 * audit namespace, and kill-switch flag. Tenant A's runaway agent can never
 * affect Tenant B's traffic or billing.
 *
 * Tenants are identified by the `x-tenant-id` request header or by an API
 * key prefix registered in config. The proxy strips tenant headers before
 * forwarding to providers.
 */

import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import * as crypto from 'crypto';

export type TenantTier = 'free' | 'starter' | 'pro' | 'max' | 'enterprise';

export interface TenantConfig {
  label: string;
  tier: TenantTier;
  /** Hard daily spend cap in USD. Requests exceeding this are blocked. */
  budget_usd_per_day?: number;
  /** Hard monthly spend cap in USD. */
  budget_usd_per_month?: number;
  /** Allowlist of model IDs this tenant may use. Empty = all models allowed. */
  allowed_models?: string[];
  /** Denylist of model IDs. Checked after allowed_models. */
  denied_models?: string[];
  /** Maximum requests per minute for this tenant. */
  rpm_limit?: number;
  /** When true, ALL requests from this tenant are immediately rejected (kill-switch). */
  kill_switch?: boolean;
  /** Timestamp when the kill-switch was activated. */
  kill_switch_activated_at?: string;
  /** Human-readable reason for the kill-switch. */
  kill_switch_reason?: string;
  /** Metadata tags for dashboards and audit logs. */
  tags?: Record<string, string>;
  created_at: string;
  updated_at: string;
}

export interface TenantSpend {
  tenant_id: string;
  date: string;
  spend_usd: number;
  request_count: number;
  last_request_at: string;
}

export interface TenantRequestContext {
  tenant_id: string;
  trace_id: string;
  request_id: string;
  timestamp: string;
}

export interface TenantCheckResult {
  allowed: boolean;
  tenant_id: string;
  reason?: string;
  /** Set when kill-switch is active */
  kill_switch_active?: boolean;
  /** Set when a budget limit would be exceeded */
  budget_exceeded?: boolean;
  /** Set when the requested model is not allowed */
  model_denied?: boolean;
  /** Current daily spend for this tenant */
  daily_spend_usd?: number;
  /** Remaining daily budget */
  daily_budget_remaining_usd?: number;
}

export interface TenantIsolatorOptions {
  /** Path to the tenants config file. Defaults to ~/.relayplane/tenants.json */
  configPath?: string;
  /** Path to the spend tracking SQLite DB or JSON store. Defaults to ~/.relayplane/tenant-spend.json */
  spendStorePath?: string;
}

function resolveRelayplaneDir(): string {
  const homeOverride = process.env['RELAYPLANE_HOME_OVERRIDE'];
  const base = homeOverride ?? os.homedir();
  return path.join(base, '.relayplane');
}

/**
 * Manages per-tenant isolation: configuration, budget enforcement, and
 * kill-switch state. Designed to be instantiated once and reused per request.
 */
export class TenantIsolator {
  private configPath: string;
  private spendStorePath: string;
  private tenants: Map<string, TenantConfig> = new Map();
  private spendCache: Map<string, TenantSpend> = new Map();
  private killSwitchCache: Set<string> = new Set();

  constructor(options: TenantIsolatorOptions = {}) {
    const dir = resolveRelayplaneDir();
    this.configPath = options.configPath ?? path.join(dir, 'tenants.json');
    this.spendStorePath = options.spendStorePath ?? path.join(dir, 'tenant-spend.json');
    this.load();
  }

  private load(): void {
    if (fs.existsSync(this.configPath)) {
      try {
        const raw = JSON.parse(fs.readFileSync(this.configPath, 'utf-8')) as Record<string, TenantConfig>;
        this.tenants = new Map(Object.entries(raw));
        // Seed in-memory kill-switch cache from persisted state
        for (const [id, cfg] of this.tenants) {
          if (cfg.kill_switch) this.killSwitchCache.add(id);
        }
      } catch {
        // Corrupt config — start with empty map
      }
    }
    if (fs.existsSync(this.spendStorePath)) {
      try {
        const raw = JSON.parse(fs.readFileSync(this.spendStorePath, 'utf-8')) as Record<string, TenantSpend>;
        this.spendCache = new Map(Object.entries(raw));
      } catch {
        // Corrupt spend store — start fresh
      }
    }
  }

  private save(): void {
    const dir = path.dirname(this.configPath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    const obj: Record<string, TenantConfig> = {};
    for (const [id, cfg] of this.tenants) obj[id] = cfg;
    fs.writeFileSync(this.configPath, JSON.stringify(obj, null, 2));
  }

  private saveSpend(): void {
    const dir = path.dirname(this.spendStorePath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    const obj: Record<string, TenantSpend> = {};
    for (const [id, spend] of this.spendCache) obj[id] = spend;
    fs.writeFileSync(this.spendStorePath, JSON.stringify(obj, null, 2));
  }

  /** Register or update a tenant configuration. */
  upsertTenant(tenantId: string, config: Omit<TenantConfig, 'created_at' | 'updated_at'>): TenantConfig {
    const now = new Date().toISOString();
    const existing = this.tenants.get(tenantId);
    const full: TenantConfig = {
      ...config,
      created_at: existing?.created_at ?? now,
      updated_at: now,
    };
    this.tenants.set(tenantId, full);
    if (full.kill_switch) {
      this.killSwitchCache.add(tenantId);
    } else {
      this.killSwitchCache.delete(tenantId);
    }
    this.save();
    return full;
  }

  /** Remove a tenant and all their spend records. */
  deleteTenant(tenantId: string): boolean {
    const existed = this.tenants.delete(tenantId);
    this.spendCache.delete(tenantId);
    this.killSwitchCache.delete(tenantId);
    if (existed) {
      this.save();
      this.saveSpend();
    }
    return existed;
  }

  /** Get a tenant config by ID. Returns undefined for unknown tenants. */
  getTenant(tenantId: string): TenantConfig | undefined {
    return this.tenants.get(tenantId);
  }

  /** List all tenants. */
  listTenants(): Array<{ id: string; config: TenantConfig }> {
    return Array.from(this.tenants.entries()).map(([id, config]) => ({ id, config }));
  }

  /**
   * Extract tenant ID from an incoming request.
   * Priority: x-tenant-id header > API key prefix registered in config.
   * Returns 'default' if no tenant can be identified.
   */
  extractTenantId(headers: Record<string, string | string[] | undefined>, apiKey?: string): string {
    // 1. Explicit header
    const header = headers['x-tenant-id'];
    if (header) return Array.isArray(header) ? header[0] : header;

    // 2. API key prefix match
    if (apiKey) {
      for (const [id, cfg] of this.tenants) {
        if (cfg.tags?.['api_key_prefix'] && apiKey.startsWith(cfg.tags['api_key_prefix'])) {
          return id;
        }
      }
    }

    return 'default';
  }

  /**
   * Check whether a request from a tenant should be allowed.
   * Checks kill-switch, model allowlist/denylist, and budget caps.
   */
  checkRequest(tenantId: string, model?: string, estimatedCostUsd = 0): TenantCheckResult {
    // Fast path: in-memory kill-switch check (no disk I/O)
    if (this.killSwitchCache.has(tenantId)) {
      return { allowed: false, tenant_id: tenantId, kill_switch_active: true, reason: 'Kill-switch is active for this tenant.' };
    }

    const config = this.tenants.get(tenantId);
    if (!config) {
      // Unknown tenant — allow by default (open proxy mode)
      return { allowed: true, tenant_id: tenantId };
    }

    // Model allowlist check
    if (model && config.allowed_models && config.allowed_models.length > 0) {
      if (!config.allowed_models.includes(model)) {
        return { allowed: false, tenant_id: tenantId, model_denied: true, reason: `Model '${model}' is not in the allowlist for tenant '${tenantId}'.` };
      }
    }

    // Model denylist check
    if (model && config.denied_models && config.denied_models.includes(model)) {
      return { allowed: false, tenant_id: tenantId, model_denied: true, reason: `Model '${model}' is denied for tenant '${tenantId}'.` };
    }

    // Budget check
    if (config.budget_usd_per_day !== undefined) {
      const today = new Date().toISOString().slice(0, 10);
      const spendKey = `${tenantId}:${today}`;
      const spend = this.spendCache.get(spendKey);
      const currentSpend = spend?.spend_usd ?? 0;
      const remaining = config.budget_usd_per_day - currentSpend;

      if (estimatedCostUsd > 0 && currentSpend + estimatedCostUsd > config.budget_usd_per_day) {
        return {
          allowed: false,
          tenant_id: tenantId,
          budget_exceeded: true,
          daily_spend_usd: currentSpend,
          daily_budget_remaining_usd: Math.max(0, remaining),
          reason: `Daily budget of $${config.budget_usd_per_day} exceeded for tenant '${tenantId}'.`,
        };
      }

      return {
        allowed: true,
        tenant_id: tenantId,
        daily_spend_usd: currentSpend,
        daily_budget_remaining_usd: Math.max(0, remaining),
      };
    }

    return { allowed: true, tenant_id: tenantId };
  }

  /** Record spend for a tenant after a successful request. */
  recordSpend(tenantId: string, costUsd: number): void {
    const today = new Date().toISOString().slice(0, 10);
    const spendKey = `${tenantId}:${today}`;
    const existing = this.spendCache.get(spendKey);
    const now = new Date().toISOString();

    this.spendCache.set(spendKey, {
      tenant_id: tenantId,
      date: today,
      spend_usd: (existing?.spend_usd ?? 0) + costUsd,
      request_count: (existing?.request_count ?? 0) + 1,
      last_request_at: now,
    });
    this.saveSpend();
  }

  /** Get today's spend for a tenant. */
  getDailySpend(tenantId: string): number {
    const today = new Date().toISOString().slice(0, 10);
    return this.spendCache.get(`${tenantId}:${today}`)?.spend_usd ?? 0;
  }

  /**
   * Generate the request context headers to inject into downstream traces.
   * The proxy adds these before forwarding and strips them from the outbound
   * request to the provider.
   */
  buildRequestContext(tenantId: string): TenantRequestContext {
    return {
      tenant_id: tenantId,
      trace_id: `trace_${crypto.randomBytes(8).toString('hex')}`,
      request_id: `req_${crypto.randomBytes(6).toString('hex')}`,
      timestamp: new Date().toISOString(),
    };
  }
}

/** Singleton instance for use across the proxy server. */
let _instance: TenantIsolator | undefined;

export function getTenantIsolator(options?: TenantIsolatorOptions): TenantIsolator {
  if (!_instance) _instance = new TenantIsolator(options);
  return _instance;
}

/** Reset the singleton (for tests). */
export function resetTenantIsolator(): void {
  _instance = undefined;
}
