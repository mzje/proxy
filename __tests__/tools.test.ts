import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  ToolRouter,
  BUILTIN_PACKS,
  DEFAULT_TOOL_ROUTER_CONFIG,
  type ToolAuthContext,
} from '../src/tool-router.js';

describe('ToolRouter', () => {
  let router: ToolRouter;

  beforeEach(() => {
    // Fresh instance per test — no file I/O (no packsDir)
    router = new ToolRouter({ enabled: true, packsDir: undefined });
  });

  afterEach(() => {
    ToolRouter.reset();
  });

  // ── Built-in packs ──────────────────────────────────────────────────────────

  describe('built-in packs', () => {
    it('loads 3 built-in packs on init', () => {
      const packs = router.listPacks();
      expect(packs).toHaveLength(3);
      const names = packs.map(p => p.name);
      expect(names).toContain('code');
      expect(names).toContain('search');
      expect(names).toContain('file-ops');
    });

    it('built-in packs have builtIn=true', () => {
      for (const pack of router.listPacks()) {
        expect(pack.builtIn).toBe(true);
      }
    });

    it('code pack has correct tools', () => {
      const pack = router.getPack('code');
      expect(pack).toBeDefined();
      const names = pack!.tools.map(t => t.name);
      expect(names).toContain('str_replace_editor');
      expect(names).toContain('bash');
      expect(names).toContain('read_file');
      expect(names).toContain('write_file');
      expect(names).toContain('list_directory');
    });

    it('search pack has rate limits', () => {
      const pack = router.getPack('search');
      const webSearch = pack!.tools.find(t => t.name === 'web_search');
      expect(webSearch?.rateLimit?.maxCallsPerSession).toBe(20);
      expect(webSearch?.rateLimit?.maxCallsPerMinute).toBe(5);
      const webFetch = pack!.tools.find(t => t.name === 'web_fetch');
      expect(webFetch?.rateLimit?.maxCallsPerSession).toBe(50);
      expect(webFetch?.rateLimit?.maxCallsPerMinute).toBe(10);
    });

    it('file-ops pack: delete_file is denied, others allowed', () => {
      const pack = router.getPack('file-ops');
      const deleteFile = pack!.tools.find(t => t.name === 'delete_file');
      expect(deleteFile?.policy).toBe('deny');
      const readFile = pack!.tools.find(t => t.name === 'read_file');
      expect(readFile?.policy).toBe('allow');
    });

    it('cannot delete a built-in pack', () => {
      expect(() => router.deletePack('code')).toThrow('built-in');
    });

    it('cannot update a built-in pack', () => {
      expect(() => router.updatePack('code', { description: 'hacked' })).toThrow('built-in');
    });
  });

  // ── Deny-by-default ─────────────────────────────────────────────────────────

  describe('deny-by-default', () => {
    it('denies all tools when no packs are active', () => {
      const ctx: ToolAuthContext = {
        sessionId: 'sess-1',
        activePacks: [],
        denyList: [],
        requestedTools: ['bash', 'web_search', 'some_tool'],
      };
      const result = router.checkTools(ctx);
      expect(result.allowed).toHaveLength(0);
      expect(result.denied).toEqual(['bash', 'web_search', 'some_tool']);
    });

    it('denies tools not in active pack', () => {
      const ctx: ToolAuthContext = {
        sessionId: 'sess-1',
        activePacks: ['code'],
        denyList: [],
        requestedTools: ['bash', 'web_search'],
      };
      const result = router.checkTools(ctx);
      expect(result.allowed).toContain('bash');
      expect(result.denied).toContain('web_search');
    });

    it('allows everything when router is disabled', () => {
      const r = new ToolRouter({ enabled: false, packsDir: undefined });
      const ctx: ToolAuthContext = {
        sessionId: 's',
        activePacks: [],
        denyList: [],
        requestedTools: ['any_tool', 'another'],
      };
      const result = r.checkTools(ctx);
      expect(result.allowed).toEqual(['any_tool', 'another']);
      expect(result.denied).toHaveLength(0);
    });
  });

  // ── Pack resolution from X-Task-Type ────────────────────────────────────────

  describe('resolveActivePacks', () => {
    it('resolves "code" task type to code pack', () => {
      const packs = router.resolveActivePacks('code');
      expect(packs).toEqual(['code']);
    });

    it('resolves "search" task type to search pack', () => {
      const packs = router.resolveActivePacks('search');
      expect(packs).toEqual(['search']);
    });

    it('resolves "file-ops" task type to file-ops pack', () => {
      const packs = router.resolveActivePacks('file-ops');
      expect(packs).toEqual(['file-ops']);
    });

    it('returns empty array for unknown task type', () => {
      const packs = router.resolveActivePacks('unknown-type');
      expect(packs).toEqual([]);
    });

    it('resolves "custom:{name}" to custom pack', () => {
      router.createPack({
        name: 'my-pack',
        description: 'test',
        tools: [{ name: 'my_tool', policy: 'allow' }],
        defaultPolicy: 'deny',
        version: '1.0.0',
      });
      const packs = router.resolveActivePacks('custom:my-pack');
      expect(packs).toEqual(['my-pack']);
    });

    it('returns empty for "custom:" prefix with unknown pack', () => {
      const packs = router.resolveActivePacks('custom:does-not-exist');
      expect(packs).toEqual([]);
    });

    it('returns empty for undefined task type', () => {
      const packs = router.resolveActivePacks(undefined);
      expect(packs).toEqual([]);
    });
  });

  // ── X-Task-Type activates correct pack ───────────────────────────────────────

  describe('X-Task-Type header activates pack', () => {
    it('code task allows str_replace_editor', () => {
      const activePacks = router.resolveActivePacks('code');
      const ctx: ToolAuthContext = {
        sessionId: 's',
        activePacks,
        denyList: [],
        requestedTools: ['str_replace_editor'],
      };
      expect(router.checkTools(ctx).allowed).toContain('str_replace_editor');
    });

    it('search task allows web_search', () => {
      const activePacks = router.resolveActivePacks('search');
      const ctx: ToolAuthContext = {
        sessionId: 's',
        activePacks,
        denyList: [],
        requestedTools: ['web_search'],
      };
      expect(router.checkTools(ctx).allowed).toContain('web_search');
    });

    it('search task denies bash', () => {
      const activePacks = router.resolveActivePacks('search');
      const ctx: ToolAuthContext = {
        sessionId: 's',
        activePacks,
        denyList: [],
        requestedTools: ['bash'],
      };
      expect(router.checkTools(ctx).denied).toContain('bash');
    });
  });

  // ── Agent overrides ──────────────────────────────────────────────────────────

  describe('agent overrides', () => {
    it('agent can override a tool from deny to allow', () => {
      const r = new ToolRouter({
        enabled: true,
        packsDir: undefined,
        agentConfigs: {
          'agent-1': {
            agentId: 'agent-1',
            allowPacks: [],
            denyPacks: [],
            toolOverrides: { bash: 'allow' },
          },
        },
      });
      const activePacks = r.resolveActivePacks('search', 'agent-1');
      const ctx: ToolAuthContext = {
        sessionId: 's',
        agentId: 'agent-1',
        activePacks,
        denyList: [],
        requestedTools: ['bash', 'web_search'],
      };
      const result = r.checkTools(ctx);
      expect(result.allowed).toContain('bash');
      expect(result.allowed).toContain('web_search');
    });

    it('agent can override an allowed tool to deny', () => {
      const r = new ToolRouter({
        enabled: true,
        packsDir: undefined,
        agentConfigs: {
          'agent-2': {
            agentId: 'agent-2',
            allowPacks: [],
            denyPacks: [],
            toolOverrides: { bash: 'deny' },
          },
        },
      });
      const activePacks = r.resolveActivePacks('code', 'agent-2');
      const ctx: ToolAuthContext = {
        sessionId: 's',
        agentId: 'agent-2',
        activePacks,
        denyList: [],
        requestedTools: ['bash'],
      };
      const result = r.checkTools(ctx);
      expect(result.denied).toContain('bash');
    });

    it('agent allowPacks adds a pack to active packs', () => {
      const r = new ToolRouter({
        enabled: true,
        packsDir: undefined,
        agentConfigs: {
          'agent-3': {
            agentId: 'agent-3',
            allowPacks: ['search'],
            denyPacks: [],
            toolOverrides: {},
          },
        },
      });
      // No X-Task-Type, but agent adds search pack
      const activePacks = r.resolveActivePacks(undefined, 'agent-3');
      expect(activePacks).toContain('search');
    });

    it('agent denyPacks removes a pack from active packs', () => {
      const r = new ToolRouter({
        enabled: true,
        packsDir: undefined,
        agentConfigs: {
          'agent-4': {
            agentId: 'agent-4',
            allowPacks: [],
            denyPacks: ['code'],
            toolOverrides: {},
          },
        },
      });
      const activePacks = r.resolveActivePacks('code', 'agent-4');
      expect(activePacks).not.toContain('code');
    });
  });

  // ── Explicit deny list ───────────────────────────────────────────────────────

  describe('explicit deny list', () => {
    it('deny list wins over active pack allow', () => {
      const ctx: ToolAuthContext = {
        sessionId: 's',
        activePacks: ['code'],
        denyList: ['bash'],
        requestedTools: ['bash', 'read_file'],
      };
      const result = router.checkTools(ctx);
      expect(result.denied).toContain('bash');
      expect(result.allowed).toContain('read_file');
    });

    it('deny list wins over agent allow override', () => {
      const r = new ToolRouter({
        enabled: true,
        packsDir: undefined,
        agentConfigs: {
          'agent-5': {
            agentId: 'agent-5',
            allowPacks: [],
            denyPacks: [],
            toolOverrides: { bash: 'allow' },
          },
        },
      });
      const ctx: ToolAuthContext = {
        sessionId: 's',
        agentId: 'agent-5',
        activePacks: ['code'],
        denyList: ['bash'],
        requestedTools: ['bash'],
      };
      const result = r.checkTools(ctx);
      expect(result.denied).toContain('bash');
    });
  });

  // ── X-Relay-Tools-Denied header ─────────────────────────────────────────────

  describe('X-Relay-Tools-Denied header', () => {
    it('populated when tools are denied', () => {
      const ctx: ToolAuthContext = {
        sessionId: 's',
        activePacks: ['code'],
        denyList: [],
        requestedTools: ['bash', 'web_search', 'web_fetch'],
      };
      const result = router.checkTools(ctx);
      expect(result.deniedHeader).toContain('web_search');
      expect(result.deniedHeader).toContain('web_fetch');
    });

    it('empty string when nothing is denied', () => {
      const ctx: ToolAuthContext = {
        sessionId: 's',
        activePacks: ['code'],
        denyList: [],
        requestedTools: ['bash'],
      };
      expect(router.checkTools(ctx).deniedHeader).toBe('');
    });
  });

  // ── Custom pack CRUD ─────────────────────────────────────────────────────────

  describe('custom pack creation', () => {
    it('creates a custom pack', () => {
      const pack = router.createPack({
        name: 'my-tools',
        description: 'Custom tools',
        tools: [{ name: 'custom_tool', policy: 'allow' }],
        defaultPolicy: 'deny',
        version: '1.0.0',
      });
      expect(pack.name).toBe('my-tools');
      expect(pack.builtIn).toBe(false);
      expect(router.getPack('my-tools')).toBeDefined();
    });

    it('throws when creating a pack with duplicate name', () => {
      router.createPack({
        name: 'dupe',
        description: 'd',
        tools: [],
        defaultPolicy: 'deny',
        version: '1.0.0',
      });
      expect(() => router.createPack({
        name: 'dupe',
        description: 'd2',
        tools: [],
        defaultPolicy: 'deny',
        version: '1.0.0',
      })).toThrow('already exists');
    });

    it('updates a custom pack', () => {
      router.createPack({
        name: 'updatable',
        description: 'old',
        tools: [],
        defaultPolicy: 'deny',
        version: '1.0.0',
      });
      const updated = router.updatePack('updatable', { description: 'new' });
      expect(updated.description).toBe('new');
    });

    it('deletes a custom pack', () => {
      router.createPack({
        name: 'deletable',
        description: 'd',
        tools: [],
        defaultPolicy: 'deny',
        version: '1.0.0',
      });
      router.deletePack('deletable');
      expect(router.getPack('deletable')).toBeUndefined();
    });

    it('throws when deleting a non-existent pack', () => {
      expect(() => router.deletePack('does-not-exist')).toThrow('not found');
    });

    it('custom pack tools are authorized correctly', () => {
      router.createPack({
        name: 'ci-tools',
        description: 'CI-specific',
        tools: [{ name: 'run_tests', policy: 'allow' }],
        defaultPolicy: 'deny',
        version: '1.0.0',
      });
      const ctx: ToolAuthContext = {
        sessionId: 's',
        activePacks: ['ci-tools'],
        denyList: [],
        requestedTools: ['run_tests', 'bash'],
      };
      const result = router.checkTools(ctx);
      expect(result.allowed).toContain('run_tests');
      expect(result.denied).toContain('bash');
    });
  });

  // ── Rate limits ──────────────────────────────────────────────────────────────

  describe('rate limits', () => {
    // Use custom packs so session and minute limits can be tested in isolation.
    // web_search has 5/min which would interfere if testing a 20-call session limit.

    it('allows calls within session limit', () => {
      router.createPack({
        name: 'rl-sess-pack',
        description: 'test',
        tools: [{ name: 'rl_tool', policy: 'allow', rateLimit: { maxCallsPerSession: 5, maxCallsPerMinute: 100 } }],
        defaultPolicy: 'deny', version: '1.0.0',
      });
      for (let i = 0; i < 5; i++) {
        expect(router.checkRateLimit('sess-rl', 'rl_tool', ['rl-sess-pack']).allowed).toBe(true);
      }
    });

    it('blocks when session limit exceeded', () => {
      router.createPack({
        name: 'sl-pack',
        description: 'test',
        tools: [{ name: 'sl_tool', policy: 'allow', rateLimit: { maxCallsPerSession: 3, maxCallsPerMinute: 100 } }],
        defaultPolicy: 'deny', version: '1.0.0',
      });
      for (let i = 0; i < 3; i++) {
        router.checkRateLimit('sess-sl', 'sl_tool', ['sl-pack']);
      }
      const result = router.checkRateLimit('sess-sl', 'sl_tool', ['sl-pack']);
      expect(result.allowed).toBe(false);
      expect(result.reason).toBe('session_limit');
    });

    it('blocks when per-minute limit exceeded', () => {
      // web_search: 5/min — exhaust the minute window
      const activePacks = router.resolveActivePacks('search');
      for (let i = 0; i < 5; i++) {
        router.checkRateLimit('sess-ml', 'web_search', activePacks);
      }
      const result = router.checkRateLimit('sess-ml', 'web_search', activePacks);
      expect(result.allowed).toBe(false);
      expect(result.reason).toBe('minute_limit');
    });

    it('allows unlimited calls for tools without rate limit', () => {
      const activePacks = router.resolveActivePacks('code');
      for (let i = 0; i < 100; i++) {
        expect(router.checkRateLimit('sess-ul', 'bash', activePacks).allowed).toBe(true);
      }
    });

    it('rate limits are per-session (different sessions independent)', () => {
      // Exhaust minute limit on session A, session B is unaffected
      const activePacks = router.resolveActivePacks('search');
      for (let i = 0; i < 5; i++) {
        router.checkRateLimit('sess-isol-a', 'web_search', activePacks);
      }
      expect(router.checkRateLimit('sess-isol-a', 'web_search', activePacks).allowed).toBe(false);
      expect(router.checkRateLimit('sess-isol-b', 'web_search', activePacks).allowed).toBe(true);
    });

    it('clearSessionRateLimits resets state', () => {
      router.createPack({
        name: 'clr-rl-pack',
        description: 'test',
        tools: [{ name: 'clr_tool', policy: 'allow', rateLimit: { maxCallsPerSession: 2, maxCallsPerMinute: 100 } }],
        defaultPolicy: 'deny', version: '1.0.0',
      });
      router.checkRateLimit('sess-clr2', 'clr_tool', ['clr-rl-pack']);
      router.checkRateLimit('sess-clr2', 'clr_tool', ['clr-rl-pack']);
      expect(router.checkRateLimit('sess-clr2', 'clr_tool', ['clr-rl-pack']).allowed).toBe(false);
      router.clearSessionRateLimits('sess-clr2');
      expect(router.checkRateLimit('sess-clr2', 'clr_tool', ['clr-rl-pack']).allowed).toBe(true);
    });
  });

  // ── Lazy schema loading ──────────────────────────────────────────────────────

  describe('lazy schema loading', () => {
    it('returns undefined for tools with no schemaRef', async () => {
      const schema = await router.getToolSchema('bash', ['code']);
      expect(schema).toBeUndefined();
    });

    it('returns undefined for unknown tool', async () => {
      const schema = await router.getToolSchema('unknown_tool', ['code']);
      expect(schema).toBeUndefined();
    });
  });

  // ── Denied tools log ──────────────────────────────────────────────────────────

  describe('denied tools log', () => {
    it('recordDenied stores denial', () => {
      router.recordDenied('sess-d', 'bad_tool', 'not_in_active_pack');
      const denied = router.getDeniedForSession('sess-d');
      expect(denied).toHaveLength(1);
      expect(denied[0]!.toolName).toBe('bad_tool');
      expect(denied[0]!.reason).toBe('not_in_active_pack');
    });

    it('getDeniedForSession returns empty array for unknown session', () => {
      expect(router.getDeniedForSession('no-session')).toEqual([]);
    });

    it('clearDeniedLog removes session entries', () => {
      router.recordDenied('sess-clr', 'tool_x', 'reason');
      router.clearDeniedLog('sess-clr');
      expect(router.getDeniedForSession('sess-clr')).toEqual([]);
    });
  });

  // ── DEFAULT_TOOL_ROUTER_CONFIG ────────────────────────────────────────────────

  describe('DEFAULT_TOOL_ROUTER_CONFIG', () => {
    it('is disabled by default (safe for local proxy — opt-in)', () => {
      expect(DEFAULT_TOOL_ROUTER_CONFIG.enabled).toBe(false);
    });

    it('packsDir points to ~/.relayplane/config/tool-packs', () => {
      expect(DEFAULT_TOOL_ROUTER_CONFIG.packsDir).toContain('.relayplane');
      expect(DEFAULT_TOOL_ROUTER_CONFIG.packsDir).toContain('tool-packs');
    });
  });

  // ── BUILTIN_PACKS export ──────────────────────────────────────────────────────

  describe('BUILTIN_PACKS', () => {
    it('exports 3 built-in packs', () => {
      expect(BUILTIN_PACKS).toHaveLength(3);
    });

    it('all packs have defaultPolicy=deny', () => {
      for (const pack of BUILTIN_PACKS) {
        expect(pack.defaultPolicy).toBe('deny');
      }
    });
  });

  // ── Performance ───────────────────────────────────────────────────────────────

  describe('performance', () => {
    it('checkTools completes in <5ms', () => {
      const ctx: ToolAuthContext = {
        sessionId: 'perf',
        activePacks: ['code'],
        denyList: [],
        requestedTools: ['bash', 'read_file', 'write_file'],
      };
      // Warm up
      router.checkTools(ctx);
      const start = performance.now();
      for (let i = 0; i < 1000; i++) {
        router.checkTools(ctx);
      }
      const elapsed = performance.now() - start;
      expect(elapsed / 1000).toBeLessThan(5);
    });
  });
});
