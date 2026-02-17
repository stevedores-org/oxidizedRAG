/**
 * Example TypeScript module for testing RAG indexing of TypeScript code.
 */

/**
 * Represents a data point in the analysis.
 */
export interface DataPoint {
  name: string;
  value: number;
  metadata?: Record<string, unknown>;
}

/**
 * Configuration for statistical analysis.
 */
export interface AnalysisConfig {
  windowSize: number;
  threshold?: number;
  useCache?: boolean;
}

/**
 * Abstract base class for data processing.
 */
export abstract class DataProcessor<T = DataPoint> {
  abstract process(data: T[]): Promise<T[]>;

  validate(data: T[]): boolean {
    return Array.isArray(data) && data.length > 0;
  }
}

/**
 * Analyzes statistical properties of data with caching.
 */
export class StatisticalAnalyzer extends DataProcessor<DataPoint> {
  private cache: Map<string, number>;
  private config: AnalysisConfig;

  constructor(config: AnalysisConfig) {
    super();
    this.config = config;
    this.cache = new Map();
  }

  async process(data: DataPoint[]): Promise<DataPoint[]> {
    if (!this.validate(data)) {
      return [];
    }

    const results: DataPoint[] = [];
    const { windowSize } = this.config;

    for (let i = 0; i <= data.length - windowSize; i++) {
      const window = data.slice(i, i + windowSize);
      const mean = this.calculateMean(
        window.map(d => d.value)
      );

      results.push({
        name: `mean_${i}`,
        value: mean,
        metadata: { window: i }
      });
    }

    return results;
  }

  private calculateMean(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((a, b) => a + b, 0) / values.length;
  }

  calculateVariance(values: number[]): number {
    if (values.length === 0) return 0;
    const mean = this.calculateMean(values);
    const squared = values.map(v => Math.pow(v - mean, 2));
    return squared.reduce((a, b) => a + b, 0) / values.length;
  }

  calculateStdDev(values: number[]): number {
    return Math.sqrt(this.calculateVariance(values));
  }

  getPercentile(data: number[], p: number): number {
    if (data.length === 0) return 0;
    const sorted = [...data].sort((a, b) => a - b);
    const index = Math.floor((sorted.length * p) / 100);
    return sorted[Math.min(index, sorted.length - 1)];
  }

  clearCache(): void {
    this.cache.clear();
  }
}

/**
 * Type-safe aggregation function.
 */
export type AggregationType = 'mean' | 'max' | 'min' | 'median';

export function aggregateResults(
  results: DataPoint[],
  aggregation: AggregationType = 'mean'
): number | null {
  if (results.length === 0) return null;

  const values = results.map(r => r.value);

  switch (aggregation) {
    case 'mean':
      return values.reduce((a, b) => a + b, 0) / values.length;
    case 'max':
      return Math.max(...values);
    case 'min':
      return Math.min(...values);
    case 'median':
      const sorted = [...values].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 !== 0
        ? sorted[mid]
        : (sorted[mid - 1] + sorted[mid]) / 2;
    default:
      const _exhaustive: never = aggregation;
      throw new Error(`Unknown aggregation: ${_exhaustive}`);
  }
}

/**
 * Generic pipeline executor with error handling.
 */
export class PipelineExecutor<T> {
  private history: T[][] = [];
  private errorHandlers: Map<string, (error: Error) => Promise<void>>;

  constructor(
    private processors: DataProcessor<T>[],
    private options?: { maxHistory?: number }
  ) {
    this.errorHandlers = new Map();
  }

  async execute(data: T[]): Promise<T[]> {
    let current = data;

    for (const processor of this.processors) {
      try {
        current = await processor.process(current);
        this.recordHistory(current);
      } catch (error) {
        const err = error instanceof Error ? error : new Error(String(error));
        const handler = this.errorHandlers.get(processor.constructor.name);
        if (handler) {
          await handler(err);
        } else {
          throw err;
        }
      }
    }

    return current;
  }

  private recordHistory(data: T[]): void {
    const { maxHistory = 10 } = this.options || {};
    this.history.push(data);
    if (this.history.length > maxHistory) {
      this.history.shift();
    }
  }

  rollback(steps: number = 1): T[] {
    if (this.history.length <= steps) {
      throw new Error('Cannot rollback beyond available history');
    }
    this.history.splice(-steps);
    return this.history[this.history.length - 1];
  }

  getHistory(): ReadonlyArray<readonly T[]> {
    return [...this.history];
  }

  onError(
    processorName: string,
    handler: (error: Error) => Promise<void>
  ): void {
    this.errorHandlers.set(processorName, handler);
  }
}
