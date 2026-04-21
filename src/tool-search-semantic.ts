export interface SearchableTool {
  name: string;
  description: string;
}

function tokenize(text: string): string[] {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(t => t.length > 1);
}

function buildVector(tokens: string[]): Map<string, number> {
  const freq = new Map<string, number>();
  for (const t of tokens) {
    freq.set(t, (freq.get(t) ?? 0) + 1);
  }
  return freq;
}

function cosineSimilarity(a: Map<string, number>, b: Map<string, number>): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (const [term, countA] of a) {
    dot += countA * (b.get(term) ?? 0);
    normA += countA * countA;
  }
  for (const [, countB] of b) {
    normB += countB * countB;
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export class SemanticToolSearch {
  private tools: SearchableTool[] = [];
  private vectors: Map<string, number>[] = [];

  get size(): number {
    return this.tools.length;
  }

  indexTools(tools: SearchableTool[]): void {
    this.tools = [...tools];
    this.vectors = tools.map(t =>
      buildVector(tokenize(`${t.name} ${t.description}`)),
    );
  }

  search(query: string, topK: number): SearchableTool[] {
    if (this.tools.length === 0) return [];
    const queryVec = buildVector(tokenize(query));
    const scored = this.tools.map((tool, i) => ({
      tool,
      score: cosineSimilarity(queryVec, this.vectors[i]),
    }));
    scored.sort((a, b) => b.score - a.score || a.tool.name.localeCompare(b.tool.name));
    return scored.slice(0, topK).map(s => s.tool);
  }
}
