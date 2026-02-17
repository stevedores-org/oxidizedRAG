/**
 * Example JavaScript module for testing RAG indexing of JavaScript code.
 */

/**
 * Represents a data point in the analysis.
 * @typedef {Object} DataPoint
 * @property {string} name - Name of the data point
 * @property {number} value - Numeric value
 * @property {Object} [metadata] - Optional metadata
 */

/**
 * DataProcessor - Base class for processing data.
 */
class DataProcessor {
  /**
   * Process data points.
   * @param {DataPoint[]} data - Input data
   * @returns {Promise<DataPoint[]>} Processed data
   */
  async process(data) {
    throw new Error('process() must be implemented');
  }

  /**
   * Validate data integrity.
   * @param {DataPoint[]} data - Data to validate
   * @returns {boolean} True if valid
   */
  validate(data) {
    return Array.isArray(data) && data.length > 0;
  }
}

/**
 * StatisticalAnalyzer - Analyzes statistical properties.
 */
class StatisticalAnalyzer extends DataProcessor {
  constructor(windowSize = 5) {
    super();
    this.windowSize = windowSize;
    this.cache = new Map();
  }

  async process(data) {
    if (!this.validate(data)) {
      return [];
    }

    const results = [];
    for (let i = 0; i <= data.length - this.windowSize; i++) {
      const window = data.slice(i, i + this.windowSize);
      const mean = this.calculateMean(window.map(d => d.value));
      results.push({
        name: `mean_${i}`,
        value: mean,
        metadata: { window: i }
      });
    }
    return results;
  }

  calculateMean(values) {
    if (values.length === 0) return 0;
    return values.reduce((a, b) => a + b, 0) / values.length;
  }

  calculateVariance(values) {
    if (values.length === 0) return 0;
    const mean = this.calculateMean(values);
    const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
    return squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
  }

  getPercentile(data, p) {
    if (data.length === 0) return 0;
    const sorted = [...data].sort((a, b) => a - b);
    const index = Math.floor((sorted.length * p) / 100);
    return sorted[Math.min(index, sorted.length - 1)];
  }
}

/**
 * Aggregate results into a single value.
 * @param {DataPoint[]} results - Results to aggregate
 * @param {string} aggregation - Type of aggregation (mean, max, min)
 * @returns {number|null} Aggregated value
 */
function aggregateResults(results, aggregation = 'mean') {
  if (results.length === 0) return null;

  const values = results.map(r => r.value);

  switch (aggregation) {
    case 'mean':
      return values.reduce((a, b) => a + b, 0) / values.length;
    case 'max':
      return Math.max(...values);
    case 'min':
      return Math.min(...values);
    default:
      throw new Error(`Unknown aggregation: ${aggregation}`);
  }
}

/**
 * PipelineExecutor - Executes processors in sequence.
 */
class PipelineExecutor {
  constructor(processors) {
    this.processors = processors;
    this.history = [];
  }

  async execute(data) {
    let current = data;
    for (const processor of this.processors) {
      current = await processor.process(current);
      this.history.push(current);
    }
    return current;
  }

  rollback() {
    if (this.history.length > 0) {
      this.history.pop();
    }
  }
}

module.exports = {
  DataProcessor,
  StatisticalAnalyzer,
  aggregateResults,
  PipelineExecutor
};
