"""Example Python module for testing RAG indexing of Python code."""

from typing import List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class DataPoint:
    """Represents a single data point in analysis."""
    name: str
    value: float
    metadata: Optional[dict] = None


class DataProcessor(ABC):
    """Abstract base class for data processing."""

    @abstractmethod
    def process(self, data: List[DataPoint]) -> List[DataPoint]:
        """Process data points."""
        pass

    def validate(self, data: List[DataPoint]) -> bool:
        """Validate data integrity."""
        return len(data) > 0


class StatisticalAnalyzer(DataProcessor):
    """Analyzes statistical properties of data."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._cache = {}

    def process(self, data: List[DataPoint]) -> List[DataPoint]:
        """Apply statistical analysis to data points."""
        if not self.validate(data):
            return []

        results = []
        for i in range(len(data) - self.window_size + 1):
            window = data[i : i + self.window_size]
            mean = sum(d.value for d in window) / len(window)
            results.append(
                DataPoint(
                    name=f"mean_{i}",
                    value=mean,
                    metadata={"window": i}
                )
            )
        return results

    def calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance

    def get_percentile(self, data: List[float], p: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * p / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


def aggregate_results(
    results: List[DataPoint],
    aggregation: str = "mean"
) -> Optional[float]:
    """Aggregate multiple data points into single value."""
    if not results:
        return None

    values = [r.value for r in results]

    if aggregation == "mean":
        return sum(values) / len(values)
    elif aggregation == "max":
        return max(values)
    elif aggregation == "min":
        return min(values)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


class PipelineExecutor:
    """Executes a pipeline of data processors."""

    def __init__(self, processors: List[DataProcessor]):
        self.processors = processors
        self.history = []

    def execute(self, data: List[DataPoint]) -> List[DataPoint]:
        """Execute all processors in sequence."""
        current = data
        for processor in self.processors:
            current = processor.process(current)
            self.history.append(current)
        return current

    def rollback(self) -> None:
        """Rollback to previous state."""
        if self.history:
            self.history.pop()
