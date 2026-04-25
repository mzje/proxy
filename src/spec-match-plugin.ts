/**
 * Spec-Match Verification Plugin
 *
 * Before an agent marks a task complete, the orchestrator POSTs the task's
 * acceptance criteria + the agent's output (diff, text, screenshots) to
 * RelayPlane's /v1/spec-match endpoint. This plugin uses a cheap LLM
 * (Haiku by default) to evaluate whether the output satisfies each criterion.
 *
 * Returns a structured pass/fail result with per-criterion evidence and
 * confidence scores. Only pass:true results should proceed to production.
 *
 * Use in Matt's swarm: every coder task dispatch includes acceptance_criteria.
 * On completion, orchestrator calls spec-match. Failing results trigger retry
 * with escalation before the verifier agent sees the output.
 */

export interface AcceptanceCriterion {
  /** Short identifier, e.g. "hero-headline-changed" */
  id: string;
  /** Human-readable description of what must be true */
  description: string;
  /** How critical this criterion is */
  severity: 'blocker' | 'major' | 'minor';
}

export interface SpecMatchRequest {
  /** Task title or identifier for trace logging */
  task_id: string;
  /** Tenant this evaluation belongs to */
  tenant_id?: string;
  /** List of acceptance criteria to evaluate */
  acceptance_criteria: AcceptanceCriterion[];
  /**
   * The agent's output to evaluate against criteria.
   * At least one of diff, output_text, or screenshots must be provided.
   */
  diff?: string;
  output_text?: string;
  /** Base64-encoded PNG screenshots, or URLs */
  screenshots?: string[];
  /** Override the default evaluation model */
  model_override?: string;
  /** If true, include the full LLM prompt/response in the result */
  debug?: boolean;
}

export interface CriterionResult {
  criterion_id: string;
  criterion_description: string;
  met: boolean;
  evidence: string;
  confidence: 'high' | 'medium' | 'low';
  severity: 'blocker' | 'major' | 'minor';
}

export interface SpecMatchResult {
  /** True if all blocker criteria are met */
  pass: boolean;
  /** 0–100 score (percentage of weighted criteria met) */
  score: number;
  /** Per-criterion results */
  criteria_results: CriterionResult[];
  /** Criteria with severity=blocker that were not met */
  blockers: string[];
  /** Criteria with severity=major that were not met */
  warnings: string[];
  /** Model used for evaluation */
  model_used: string;
  /** Estimated cost of the spec-match call in USD */
  cost_usd: number;
  /** Trace ID for audit linking */
  trace_id: string;
  /** ISO timestamp */
  evaluated_at: string;
  /** Only present when request.debug=true */
  debug_prompt?: string;
  debug_response?: string;
}

export interface SpecMatchPluginOptions {
  /** Anthropic-compatible base URL (defaults to https://api.anthropic.com) */
  apiBaseUrl?: string;
  /** API key for the evaluation model */
  apiKey?: string;
  /** Model to use for evaluation (default: claude-haiku-4-5-20251001) */
  defaultModel?: string;
  /** Max tokens for the evaluation response (default: 2048) */
  maxTokens?: number;
}

/** Haiku price per 1M tokens (input/output) for cost estimation */
const HAIKU_INPUT_COST_PER_1M = 0.80;
const HAIKU_OUTPUT_COST_PER_1M = 4.00;

function buildEvaluationPrompt(request: SpecMatchRequest): string {
  const criteriaList = request.acceptance_criteria
    .map((c, i) => `${i + 1}. [${c.id}] (${c.severity}) ${c.description}`)
    .join('\n');

  const outputSections: string[] = [];
  if (request.diff) outputSections.push(`<diff>\n${request.diff}\n</diff>`);
  if (request.output_text) outputSections.push(`<output>\n${request.output_text}\n</output>`);
  if (request.screenshots?.length) {
    outputSections.push(`<screenshots>${request.screenshots.length} screenshot(s) provided.</screenshots>`);
  }

  return `You are a strict QA evaluator. Evaluate whether the agent's output satisfies each acceptance criterion below.

<acceptance_criteria>
${criteriaList}
</acceptance_criteria>

<agent_output>
${outputSections.join('\n\n')}
</agent_output>

For each criterion, respond with a JSON array. Each element must be:
{
  "criterion_id": "<id from above>",
  "met": true|false,
  "evidence": "<one sentence citing specific evidence from the output>",
  "confidence": "high"|"medium"|"low"
}

Rules:
- met:true only when you see clear, direct evidence in the output
- confidence:high = definitive evidence; medium = likely but not certain; low = cannot tell
- For missing/unclear evidence, met:false with confidence:low
- Do not infer; evaluate only what is present

Respond with ONLY the JSON array. No preamble.`;
}

function estimateCost(inputTokens: number, outputTokens: number): number {
  return (inputTokens / 1_000_000) * HAIKU_INPUT_COST_PER_1M +
    (outputTokens / 1_000_000) * HAIKU_OUTPUT_COST_PER_1M;
}

function buildTraceId(): string {
  return `sm_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

/**
 * SpecMatchPlugin evaluates agent output against acceptance criteria using
 * an LLM judge. Use this at task completion before marking work as done.
 */
export class SpecMatchPlugin {
  private apiBaseUrl: string;
  private apiKey: string;
  private defaultModel: string;
  private maxTokens: number;

  constructor(options: SpecMatchPluginOptions = {}) {
    this.apiBaseUrl = options.apiBaseUrl ?? (process.env['ANTHROPIC_BASE_URL'] ?? 'https://api.anthropic.com');
    this.apiKey = options.apiKey ?? (process.env['ANTHROPIC_API_KEY'] ?? '');
    this.defaultModel = options.defaultModel ?? 'claude-haiku-4-5-20251001';
    this.maxTokens = options.maxTokens ?? 2048;
  }

  async evaluate(request: SpecMatchRequest): Promise<SpecMatchResult> {
    const traceId = buildTraceId();
    const model = request.model_override ?? this.defaultModel;
    const prompt = buildEvaluationPrompt(request);

    let rawResponse = '';
    let inputTokens = 0;
    let outputTokens = 0;

    try {
      const response = await fetch(`${this.apiBaseUrl}/v1/messages`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'x-api-key': this.apiKey,
          'anthropic-version': '2023-06-01',
        },
        body: JSON.stringify({
          model,
          max_tokens: this.maxTokens,
          messages: [{ role: 'user', content: prompt }],
        }),
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`Spec-match LLM call failed: ${response.status} ${errText}`);
      }

      const body = await response.json() as {
        content: Array<{ type: string; text: string }>;
        usage?: { input_tokens: number; output_tokens: number };
      };

      rawResponse = body.content.find(c => c.type === 'text')?.text ?? '[]';
      inputTokens = body.usage?.input_tokens ?? Math.ceil(prompt.length / 4);
      outputTokens = body.usage?.output_tokens ?? Math.ceil(rawResponse.length / 4);
    } catch (err) {
      // Return a graceful failure result rather than throwing
      return {
        pass: false,
        score: 0,
        criteria_results: request.acceptance_criteria.map(c => ({
          criterion_id: c.id,
          criterion_description: c.description,
          met: false,
          evidence: `Spec-match evaluation failed: ${err instanceof Error ? err.message : String(err)}`,
          confidence: 'low',
          severity: c.severity,
        })),
        blockers: request.acceptance_criteria.filter(c => c.severity === 'blocker').map(c => c.id),
        warnings: request.acceptance_criteria.filter(c => c.severity === 'major').map(c => c.id),
        model_used: model,
        cost_usd: 0,
        trace_id: traceId,
        evaluated_at: new Date().toISOString(),
      };
    }

    // Parse LLM response
    let llmResults: Array<{
      criterion_id: string;
      met: boolean;
      evidence: string;
      confidence: 'high' | 'medium' | 'low';
    }> = [];

    try {
      // Strip markdown fences if present
      const cleaned = rawResponse.replace(/^```json\n?/i, '').replace(/\n?```$/i, '').trim();
      llmResults = JSON.parse(cleaned);
    } catch {
      // Fallback: all criteria unmet if we can't parse
      llmResults = request.acceptance_criteria.map(c => ({
        criterion_id: c.id,
        met: false,
        evidence: 'Could not parse evaluator response.',
        confidence: 'low' as const,
      }));
    }

    // Build per-criterion results
    const criteriaMap = new Map(request.acceptance_criteria.map(c => [c.id, c]));
    const criteriaResults: CriterionResult[] = llmResults.map(r => {
      const criterion = criteriaMap.get(r.criterion_id);
      return {
        criterion_id: r.criterion_id,
        criterion_description: criterion?.description ?? r.criterion_id,
        met: r.met,
        evidence: r.evidence,
        confidence: r.confidence,
        severity: criterion?.severity ?? 'minor',
      };
    });

    // Score: weighted by severity (blocker=3, major=2, minor=1)
    const weights = { blocker: 3, major: 2, minor: 1 };
    const totalWeight = criteriaResults.reduce((sum, c) => sum + weights[c.severity], 0);
    const metWeight = criteriaResults
      .filter(c => c.met)
      .reduce((sum, c) => sum + weights[c.severity], 0);
    const score = totalWeight > 0 ? Math.round((metWeight / totalWeight) * 100) : 100;

    const blockers = criteriaResults
      .filter(c => c.severity === 'blocker' && !c.met)
      .map(c => c.criterion_id);
    const warnings = criteriaResults
      .filter(c => c.severity === 'major' && !c.met)
      .map(c => c.criterion_id);

    const result: SpecMatchResult = {
      pass: blockers.length === 0,
      score,
      criteria_results: criteriaResults,
      blockers,
      warnings,
      model_used: model,
      cost_usd: estimateCost(inputTokens, outputTokens),
      trace_id: traceId,
      evaluated_at: new Date().toISOString(),
    };

    if (request.debug) {
      result.debug_prompt = prompt;
      result.debug_response = rawResponse;
    }

    return result;
  }
}

/** Singleton instance. */
let _instance: SpecMatchPlugin | undefined;

export function getSpecMatchPlugin(options?: SpecMatchPluginOptions): SpecMatchPlugin {
  if (!_instance) _instance = new SpecMatchPlugin(options);
  return _instance;
}

export function resetSpecMatchPlugin(): void {
  _instance = undefined;
}
