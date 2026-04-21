import { describe, it, expect, beforeEach } from 'vitest';
import { SemanticToolSearch, type SearchableTool } from '../src/tool-search-semantic.js';

// 25 realistic MCP-style tools covering diverse capabilities
const TOOL_LIBRARY: SearchableTool[] = [
  { name: 'read_file', description: 'Read the contents of a file from the filesystem' },
  { name: 'write_file', description: 'Write or overwrite a file on the filesystem' },
  { name: 'list_directory', description: 'List files and directories in a given path' },
  { name: 'delete_file', description: 'Delete a file or empty directory' },
  { name: 'move_file', description: 'Move or rename a file or directory' },
  { name: 'web_search', description: 'Search the internet for up-to-date information' },
  { name: 'fetch_url', description: 'Fetch the HTML or JSON content of a URL' },
  { name: 'browser_screenshot', description: 'Take a screenshot of a rendered web page' },
  { name: 'browser_click', description: 'Click a DOM element in a headless browser session' },
  { name: 'browser_type', description: 'Type text into a form field in the browser' },
  { name: 'run_sql', description: 'Execute a SQL query against a database connection' },
  { name: 'list_tables', description: 'List all tables in the connected database' },
  { name: 'describe_schema', description: 'Return the column schema for a database table' },
  { name: 'run_code', description: 'Execute Python or JavaScript code in a sandbox' },
  { name: 'install_package', description: 'Install a package from npm or pip into the sandbox' },
  { name: 'send_email', description: 'Send an email via SMTP to one or more recipients' },
  { name: 'send_slack', description: 'Post a message to a Slack channel or direct message' },
  { name: 'create_calendar_event', description: 'Create a calendar event and invite attendees' },
  { name: 'git_commit', description: 'Stage files and create a git commit with a message' },
  { name: 'git_push', description: 'Push local commits to a remote git repository' },
  { name: 'create_github_pr', description: 'Open a pull request on GitHub for a feature branch' },
  { name: 'create_github_issue', description: 'File a new issue in a GitHub repository' },
  { name: 'get_weather', description: 'Retrieve current weather and forecast for a location' },
  { name: 'translate_text', description: 'Translate text between human languages' },
  { name: 'summarize_document', description: 'Generate a concise summary of a long document' },
];

/** Rough token estimate: 1 token ≈ 4 characters (OpenAI approximation) */
function estimateTokens(tools: SearchableTool[]): number {
  return tools.reduce((sum, t) => sum + Math.ceil((t.name.length + t.description.length) / 4), 0);
}

describe('SemanticToolSearch', () => {
  let search: SemanticToolSearch;

  beforeEach(() => {
    search = new SemanticToolSearch();
    search.indexTools(TOOL_LIBRARY);
  });

  // ── Indexing ────────────────────────────────────────────────────────────────

  describe('indexTools', () => {
    it('indexes all provided tools', () => {
      expect(search.size).toBe(TOOL_LIBRARY.length);
    });

    it('re-indexing replaces the previous index', () => {
      const smaller = TOOL_LIBRARY.slice(0, 5);
      search.indexTools(smaller);
      expect(search.size).toBe(5);
    });
  });

  // ── Semantic search ─────────────────────────────────────────────────────────

  describe('search', () => {
    it('returns exactly topK results when topK < total tools', () => {
      const results = search.search('read a file from disk', 3);
      expect(results).toHaveLength(3);
    });

    it('returns all tools when topK >= total tools', () => {
      const results = search.search('anything', TOOL_LIBRARY.length + 10);
      expect(results).toHaveLength(TOOL_LIBRARY.length);
    });

    it('ranks file-reading tools highest for a filesystem query', () => {
      const results = search.search('read a file from disk', 3);
      const topName = results[0].name;
      expect(['read_file', 'list_directory', 'write_file']).toContain(topName);
    });

    it('ranks web tools highest for a browser/search query', () => {
      const results = search.search('search the web and take a screenshot', 3);
      const topNames = results.map(r => r.name);
      const webTools = ['web_search', 'fetch_url', 'browser_screenshot', 'browser_click'];
      expect(topNames.some(n => webTools.includes(n))).toBe(true);
    });

    it('ranks database tools highest for a SQL query', () => {
      const results = search.search('run a SQL query against postgres', 3);
      const topNames = results.map(r => r.name);
      expect(topNames.some(n => ['run_sql', 'list_tables', 'describe_schema'].includes(n))).toBe(true);
    });

    it('returns result objects with name and description', () => {
      const results = search.search('commit code changes', 2);
      for (const r of results) {
        expect(r).toHaveProperty('name');
        expect(r).toHaveProperty('description');
        expect(typeof r.name).toBe('string');
        expect(typeof r.description).toBe('string');
      }
    });

    it('is deterministic — same query returns same order', () => {
      const a = search.search('send a message', 5).map(r => r.name);
      const b = search.search('send a message', 5).map(r => r.name);
      expect(a).toEqual(b);
    });
  });

  // ── Token reduction (acceptance criterion) ──────────────────────────────────

  describe('token reduction', () => {
    it('reduces tool-list tokens by ≥60% for a >20-tool library', () => {
      // Full tool list has 25 entries — well above the 20-tool threshold
      expect(TOOL_LIBRARY.length).toBeGreaterThan(20);

      const fullTokens = estimateTokens(TOOL_LIBRARY);

      // Retrieve top-8 semantically relevant tools for a specific query
      const topK = 8;
      const filtered = search.search('read and write files on disk', topK);
      const filteredTokens = estimateTokens(filtered);

      const reductionPct = (fullTokens - filteredTokens) / fullTokens;
      expect(reductionPct).toBeGreaterThanOrEqual(0.6);
    });

    it('returns topK=8 results that cover the query intent without needing all 25 tools', () => {
      const results = search.search('execute python code', 8);
      expect(results.length).toBeLessThan(TOOL_LIBRARY.length);
      // The code-execution tool should be present in a focused top-8
      expect(results.map(r => r.name)).toContain('run_code');
    });
  });

  // ── Edge cases ───────────────────────────────────────────────────────────────

  describe('edge cases', () => {
    it('throws or returns empty array when no tools are indexed', () => {
      const empty = new SemanticToolSearch();
      // Either an error is thrown or an empty array is returned — both are valid
      let result: SearchableTool[] | undefined;
      try {
        result = empty.search('anything', 3);
      } catch {
        result = undefined;
      }
      if (result !== undefined) {
        expect(result).toHaveLength(0);
      }
    });

    it('handles a single-word query without throwing', () => {
      expect(() => search.search('database', 3)).not.toThrow();
    });
  });
});
