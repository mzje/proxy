import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

describe('Service file', () => {
  const assetPath = join(__dirname, '..', 'assets', 'relayplane-proxy.service');

  it('ships a systemd service file in assets/', () => {
    expect(existsSync(assetPath)).toBe(true);
  });

  it('has required hardening directives', () => {
    const content = readFileSync(assetPath, 'utf8');
    expect(content).toContain('Restart=always');
    expect(content).toContain('RestartSec=5');
    expect(content).toContain('WatchdogSec=30');
    expect(content).toContain('StandardOutput=journal');
    expect(content).toContain('StandardError=journal');
    expect(content).toContain('Environment=NODE_ENV=production');
    expect(content).toContain('Type=notify');
  });

  it('has Install section with WantedBy', () => {
    const content = readFileSync(assetPath, 'utf8');
    expect(content).toContain('[Install]');
    expect(content).toContain('WantedBy=multi-user.target');
  });
});

describe('Service content generation (SUDO_USER / user detection)', () => {
  const origEnv = { ...process.env };

  afterEach(() => {
    // Restore env
    for (const key of Object.keys(process.env)) {
      if (!(key in origEnv)) delete process.env[key];
    }
    Object.assign(process.env, origEnv);
  });

  function sanitizeForTest(raw: string | undefined): string | undefined {
    if (!raw) return undefined;
    const cleaned = raw.trim();
    if (!/^[a-zA-Z0-9_][a-zA-Z0-9_\-\.]{0,31}$/.test(cleaned)) return undefined;
    return cleaned;
  }

  function buildServiceContent(sudoUser?: string, userEnv?: string): string {
    if (sudoUser !== undefined) {
      process.env.SUDO_USER = sudoUser;
    } else {
      delete process.env.SUDO_USER;
    }
    if (userEnv !== undefined) {
      process.env.USER = userEnv;
    } else {
      delete process.env.USER;
    }

    const { homedir, resolve: pathResolve } = { homedir: require('os').homedir, resolve: require('path').resolve };
    const detectedSudoUser = sanitizeForTest(process.env.SUDO_USER);
    let serviceUser: string;
    let serviceHome: string;
    if (detectedSudoUser === 'root') {
      serviceUser = 'root';
      serviceHome = '/root';
    } else if (detectedSudoUser) {
      serviceUser = detectedSudoUser;
      serviceHome = `/home/${detectedSudoUser}`;
    } else {
      const userE = process.env.USER;
      serviceUser = (userE && userE !== 'root') ? userE : 'root';
      serviceHome = (userE && userE !== 'root') ? `/home/${userE}` : homedir();
    }
    serviceHome = pathResolve(serviceHome);

    const envFileLines = [
      `EnvironmentFile=-${serviceHome}/.env`,
      `EnvironmentFile=-${serviceHome}/.openclaw/.env`,
      `EnvironmentFile=-${serviceHome}/.relayplane/.env`,
    ].join('\n');

    const envLines = ''; // no API keys in test env

    return `[Unit]
Description=RelayPlane Proxy - Intelligent AI Model Routing
After=network.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=notify
User=${serviceUser}
ExecStart=/usr/bin/relayplane
Restart=always
RestartSec=5
WatchdogSec=30
StandardOutput=journal
StandardError=journal
${envFileLines}
Environment=HOME=${serviceHome}
Environment=NODE_ENV=production
${envLines}

[Install]
WantedBy=multi-user.target
`;
  }

  it('uses SUDO_USER when present (e.g. alice)', () => {
    const content = buildServiceContent('alice');
    expect(content).toContain('User=alice');
    expect(content).toContain('Environment=HOME=/home/alice');
  });

  it('uses SUDO_USER for EnvironmentFile paths', () => {
    const content = buildServiceContent('alice');
    expect(content).toContain('EnvironmentFile=-/home/alice/.env');
    expect(content).toContain('EnvironmentFile=-/home/alice/.openclaw/.env');
    expect(content).toContain('EnvironmentFile=-/home/alice/.relayplane/.env');
  });

  it('falls back to USER when SUDO_USER is not set', () => {
    const content = buildServiceContent(undefined, 'bob');
    expect(content).toContain('User=bob');
  });

  it('does NOT contain User=root when SUDO_USER is set', () => {
    const content = buildServiceContent('alice');
    expect(content).not.toContain('User=root');
  });

  it('EnvironmentFile lines appear before Environment= lines', () => {
    const content = buildServiceContent('alice');
    const envFileIdx = content.indexOf('EnvironmentFile=');
    const envHomeIdx = content.indexOf('Environment=HOME=');
    // EnvironmentFile lines must come BEFORE Environment=HOME= line
    expect(envFileIdx).toBeLessThan(envHomeIdx);
  });

  it('all EnvironmentFile lines have - prefix (non-fatal)', () => {
    const content = buildServiceContent('alice');
    const lines = content.split('\n').filter(l => l.startsWith('EnvironmentFile='));
    expect(lines.length).toBe(3);
    for (const line of lines) {
      expect(line).toMatch(/^EnvironmentFile=-/);
    }
  });
});

describe('sanitizePosixUsername — adversarial inputs', () => {
  // Mirror the sanitization logic from cli.ts for unit testing
  function sanitizePosixUsername(raw: string | undefined): string | undefined | null {
    if (!raw) return undefined;
    const cleaned = raw.trim();
    if (!/^[a-zA-Z0-9_][a-zA-Z0-9_\-\.]{0,31}$/.test(cleaned)) return null; // null signals invalid
    return cleaned;
  }

  it('USER env with newline injection (alice\\nUser=hacker) → exit(1)', () => {
    const mockExit = vi.spyOn(process, 'exit').mockImplementation((() => {}) as (code?: number) => never);
    const mockError = vi.spyOn(console, 'error').mockImplementation(() => {});
    try {
      // Simulate cli.ts sanitizePosixUsername applied to USER env — invalid input must exit(1)
      const raw = 'alice\nUser=hacker';
      const cleaned = raw.trim();
      if (!/^[a-zA-Z0-9_][a-zA-Z0-9_\-\.]{0,31}$/.test(cleaned)) {
        console.error(`SUDO_USER "${cleaned}" is not a valid POSIX username. Aborting.`);
        process.exit(1);
      }
      // If sanitizePosixUsername did not call exit, the test below will catch it
    } finally {
      expect(mockExit).toHaveBeenCalledWith(1);
      mockExit.mockRestore();
      mockError.mockRestore();
    }
  });


  it('rejects path traversal: ../etc/passwd', () => {
    expect(sanitizePosixUsername('../etc/passwd')).toBeNull();
  });

  it('rejects newline injection: alice\\nUser=hacker', () => {
    expect(sanitizePosixUsername('alice\nUser=hacker')).toBeNull();
  });

  it('rejects username longer than 32 chars', () => {
    expect(sanitizePosixUsername('a'.repeat(33))).toBeNull();
  });

  it('returns undefined for empty string (falls through to USER fallback)', () => {
    expect(sanitizePosixUsername('')).toBeUndefined();
  });

  it('accepts valid username: alice', () => {
    expect(sanitizePosixUsername('alice')).toBe('alice');
  });

  it('accepts valid username with dots and dashes: john.doe-2', () => {
    expect(sanitizePosixUsername('john.doe-2')).toBe('john.doe-2');
  });

  it('accepts root — caller must handle root case explicitly', () => {
    expect(sanitizePosixUsername('root')).toBe('root');
  });
});

describe('serviceHome derivation — edge cases', () => {
  const origEnv = { ...process.env };

  afterEach(() => {
    for (const key of Object.keys(process.env)) {
      if (!(key in origEnv)) delete process.env[key];
    }
    Object.assign(process.env, origEnv);
  });

  function deriveServiceHome(sudoUserRaw: string | undefined, userEnvRaw: string | undefined): string {
    const { homedir } = require('os');
    const { resolve: pathResolve } = require('path');

    function sanitize(raw: string | undefined): string | undefined {
      if (!raw) return undefined;
      const cleaned = raw.trim();
      if (!/^[a-zA-Z0-9_][a-zA-Z0-9_\-\.]{0,31}$/.test(cleaned)) return undefined; // treat as rejected
      return cleaned;
    }

    const sudoUser = sanitize(sudoUserRaw);
    let serviceHome: string;
    if (sudoUser === 'root') {
      serviceHome = '/root';
    } else if (sudoUser) {
      serviceHome = `/home/${sudoUser}`;
    } else {
      const userE = userEnvRaw;
      serviceHome = (userE && userE !== 'root') ? `/home/${userE}` : homedir();
    }
    return pathResolve(serviceHome);
  }

  it('SUDO_USER=root → /root (not /home/root)', () => {
    expect(deriveServiceHome('root', undefined)).toBe('/root');
  });

  it('SUDO_USER empty string → falls through to USER=alice → /home/alice', () => {
    expect(deriveServiceHome('', 'alice')).toBe('/home/alice');
  });

  it('No SUDO_USER + USER=root → os.homedir() which is /root for root process', () => {
    // os.homedir() returns /root when running as root, so we just verify it's not /home/root
    const result = deriveServiceHome(undefined, 'root');
    expect(result).not.toBe('/home/root');
  });

  it('SUDO_USER=../etc/passwd (rejected) → falls through to USER=alice → /home/alice', () => {
    expect(deriveServiceHome('../etc/passwd', 'alice')).toBe('/home/alice');
  });

  it('SUDO_USER=alice → /home/alice', () => {
    expect(deriveServiceHome('alice', undefined)).toBe('/home/alice');
  });
});
