#!/usr/bin/env node
/**
 * extract-knowledge.js
 *
 * Reads RelayPlane telemetry (last N hours), groups by task_type + model + outcome,
 * calls Claude Haiku to extract 3-5 bullet learnings, and appends to the relevant
 * knowledge file in ~/.openclaw/workspace/knowledge/.
 *
 * Usage:
 *   node extract-knowledge.js [--agent coder] [--hours 24]
 *
 * Requirements:
 *   ANTHROPIC_API_KEY env var (or will skip AI extraction and just summarize counts)
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const os = require('os');
const readline = require('readline');

// ── CLI args ────────────────────────────────────────────────────────────────
const args = process.argv.slice(2);
function getArg(flag, defaultVal) {
  const idx = args.indexOf(flag);
  if (idx !== -1 && args[idx + 1]) return args[idx + 1];
  return defaultVal;
}

const AGENT = getArg('--agent', 'coder');
const HOURS = parseInt(getArg('--hours', '24'), 10);

// ── Paths ────────────────────────────────────────────────────────────────────
const TELEMETRY_PATH = path.join(os.homedir(), '.relayplane', 'telemetry.jsonl');
const KNOWLEDGE_DIR = path.join(os.homedir(), '.openclaw', 'workspace', 'knowledge');

const AGENT_TO_FILE = {
  coder: 'coder-patterns.md',
  sentinel: 'security-patterns.md',
  hunter: 'gtm-patterns.md',
  writer: 'gtm-patterns.md',
  main: 'agent-patterns.md',
};

function getKnowledgeFile(agent) {
  return path.join(KNOWLEDGE_DIR, AGENT_TO_FILE[agent] || 'agent-patterns.md');
}

// ── Telemetry reading ────────────────────────────────────────────────────────
async function readRecentTelemetry(hours) {
  const cutoff = Date.now() - hours * 60 * 60 * 1000;
  const records = [];

  if (!fs.existsSync(TELEMETRY_PATH)) {
    console.log(`No telemetry file found at ${TELEMETRY_PATH}`);
    return records;
  }

  const rl = readline.createInterface({
    input: fs.createReadStream(TELEMETRY_PATH),
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    if (!line.trim()) continue;
    try {
      const record = JSON.parse(line);
      const ts = record.started_at
        ? new Date(record.started_at).getTime()
        : record.timestamp
          ? new Date(record.timestamp).getTime()
          : 0;
      if (ts >= cutoff) {
        records.push(record);
      }
    } catch {
      // skip malformed lines
    }
  }

  return records;
}

// ── Group by task_type + model + outcome ─────────────────────────────────────
function groupRecords(records) {
  const groups = {};
  for (const r of records) {
    const key = `${r.taskType || 'general'}|${r.model || 'unknown'}|${r.status || 'unknown'}`;
    if (!groups[key]) {
      groups[key] = { taskType: r.taskType || 'general', model: r.model || 'unknown', status: r.status || 'unknown', count: 0, totalCost: 0, totalLatency: 0, errors: [] };
    }
    groups[key].count++;
    groups[key].totalCost += r.costUsd || 0;
    groups[key].totalLatency += r.latencyMs || 0;
    if (r.status !== 'success' && r.error) {
      groups[key].errors.push(r.error);
    }
  }
  return Object.values(groups).sort((a, b) => b.count - a.count);
}

// ── Claude Haiku call ────────────────────────────────────────────────────────
function callHaiku(prompt) {
  return new Promise((resolve, reject) => {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      resolve(null);
      return;
    }

    const body = JSON.stringify({
      model: 'claude-haiku-4-5-20251001',
      max_tokens: 512,
      messages: [{ role: 'user', content: prompt }],
    });

    const options = {
      hostname: 'api.anthropic.com',
      path: '/v1/messages',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'Content-Length': Buffer.byteLength(body),
      },
    };

    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          const text = parsed.content?.[0]?.text || null;
          resolve(text);
        } catch {
          resolve(null);
        }
      });
    });

    req.on('error', (err) => {
      console.warn('Haiku call failed:', err.message);
      resolve(null);
    });

    req.setTimeout(15000, () => {
      req.destroy();
      resolve(null);
    });

    req.write(body);
    req.end();
  });
}

// ── Build extraction prompt ──────────────────────────────────────────────────
function buildPrompt(agent, groups, hours) {
  const summary = groups
    .map(
      (g) =>
        `- ${g.taskType} / ${g.model} / ${g.status}: ${g.count} requests` +
        (g.totalCost > 0 ? `, avg cost $${(g.totalCost / g.count).toFixed(5)}` : '') +
        (g.totalLatency > 0 ? `, avg latency ${Math.round(g.totalLatency / g.count)}ms` : '') +
        (g.errors.length > 0 ? `, errors: ${[...new Set(g.errors)].slice(0, 2).join('; ')}` : '')
    )
    .join('\n');

  return `You are analyzing AI agent telemetry data for the RelayPlane proxy. Agent: @${agent}.

Data from the last ${hours} hours:
${summary}

Extract 3-5 concise bullet point learnings that would be useful for the @${agent} agent in future sessions.
Focus on: cost patterns, model selection efficiency, error patterns, routing insights.
Be specific with numbers where helpful. Keep each bullet under 100 characters.
Format as a markdown bullet list only - no preamble, no headers, just the bullets.`;
}

// ── Fallback: generate summary without AI ───────────────────────────────────
function buildFallbackLearnings(groups, hours) {
  const lines = [];
  const total = groups.reduce((s, g) => s + g.count, 0);
  lines.push(`- ${total} requests in last ${hours}h across ${groups.length} task/model/status groups`);

  const topGroup = groups[0];
  if (topGroup) {
    lines.push(`- Most common: ${topGroup.taskType}/${topGroup.model} (${topGroup.count} requests, ${topGroup.status})`);
  }

  const errorGroups = groups.filter((g) => g.status !== 'success');
  if (errorGroups.length > 0) {
    const errorCount = errorGroups.reduce((s, g) => s + g.count, 0);
    lines.push(`- ${errorCount} error requests across ${errorGroups.length} groups`);
    const firstErr = errorGroups[0]?.errors[0];
    if (firstErr) lines.push(`- Most common error: ${firstErr.slice(0, 80)}`);
  }

  const totalCost = groups.reduce((s, g) => s + g.totalCost, 0);
  if (totalCost > 0) {
    lines.push(`- Total cost tracked: $${totalCost.toFixed(4)}`);
  }

  return lines.join('\n');
}

// ── Append to knowledge file ─────────────────────────────────────────────────
function appendToKnowledgeFile(knowledgeFile, agent, learnings) {
  fs.mkdirSync(path.dirname(knowledgeFile), { recursive: true });

  const timestamp = new Date().toISOString().slice(0, 10);
  const entry = `\n<!-- [${timestamp}] agent:${agent} -->\n${learnings}\n`;

  fs.appendFileSync(knowledgeFile, entry, 'utf8');
  console.log(`Appended to ${knowledgeFile}`);
}

// ── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  console.log(`Extracting knowledge for agent:${AGENT}, last ${HOURS}h`);

  const records = await readRecentTelemetry(HOURS);
  console.log(`Found ${records.length} telemetry records`);

  if (records.length === 0) {
    console.log('No telemetry data in the time window. Nothing to extract.');
    return;
  }

  const groups = groupRecords(records);
  console.log(`Grouped into ${groups.length} task/model/status combinations`);

  let learnings;
  const apiKey = process.env.ANTHROPIC_API_KEY;

  if (apiKey) {
    console.log('Calling Claude Haiku for extraction...');
    const prompt = buildPrompt(AGENT, groups, HOURS);
    learnings = await callHaiku(prompt);
    if (!learnings) {
      console.warn('Haiku extraction failed, falling back to summary');
      learnings = buildFallbackLearnings(groups, HOURS);
    }
  } else {
    console.log('ANTHROPIC_API_KEY not set - generating summary without AI');
    learnings = buildFallbackLearnings(groups, HOURS);
  }

  console.log('\nLearnings extracted:\n' + learnings);

  const knowledgeFile = getKnowledgeFile(AGENT);
  appendToKnowledgeFile(knowledgeFile, AGENT, learnings);

  console.log('\nDone.');
}

main().catch((err) => {
  console.error('Error:', err.message);
  process.exit(1);
});
