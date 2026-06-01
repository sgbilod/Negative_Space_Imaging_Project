# Analytics API Reference

Complete reference documentation for the Analytics system's public APIs.

## Table of Contents

1. [Events Module](#events-module)
2. [Streaming Module](#streaming-module)
3. [Configuration Module](#configuration-module)
4. [Storage Module](#storage-module)
5. [Statistical Analysis](#statistical-analysis)
6. [Metrics Computation](#metrics-computation)
7. [Anomaly Detection](#anomaly-detection)

---

## Events Module

### Class: `Event`

Represents an individual analytics event with validation.

```python
from analytics.events import Event

# Create an event
event = Event(
    timestamp=datetime.now(),
    event_type="user_action",
    data={"user_id": 123, "action": "click"},
    metadata={"source": "web", "version": "1.0"}
)
```

#### Constructor

```python
Event(timestamp, event_type, data, metadata=None)
```

**Parameters**:
- `timestamp` (datetime): Event occurrence time
- `event_type` (str): Type of event
- `data` (dict): Event payload
- `metadata` (dict, optional): Additional context

**Raises**:
- `ValueError`: Invalid timestamp or empty event_type

#### Methods

##### `validate(schema=None) -> bool`

Validates event against optional schema.

```python
if event.validate(schema={"user_id": int, "action": str}):
    print("Event is valid")
```

**Returns**: `bool` - True if valid

**Raises**: `ValueError` if validation fails

##### `get_field(field_path) -> Any`

Retrieve nested field using dot notation.

```python
user_id = event.get_field("data.user_id")
source = event.get_field("metadata.source")
```

**Parameters**:
- `field_path` (str): Dot-separated path (e.g., "data.user_id")

**Returns**: Field value or None if not found

##### `to_dict() -> dict`

Convert event to dictionary.

```python
event_dict = event.to_dict()
```

**Returns**: `dict` representation of event

---

### Class: `EventEnricher`

Adds contextual metadata to events.

```python
from analytics.events import EventEnricher

enricher = EventEnricher(
    geo_lookup=True,
    add_timestamp=True,
    add_source_info=True
)
```

#### Constructor

```python
EventEnricher(geo_lookup=False, add_timestamp=False, add_source_info=False)
```

**Parameters**:
- `geo_lookup` (bool): Add geographic data
- `add_timestamp` (bool): Add processing timestamp
- `add_source_info` (bool): Add source information

#### Methods

##### `add_context(event, context=None) -> Event`

Enrich event with contextual information.

```python
enriched = enricher.add_context(event, context={"user_segment": "premium"})
```

**Parameters**:
- `event` (Event): Event to enrich
- `context` (dict, optional): Additional context

**Returns**: `Event` - Enriched event

**Example**:
```python
event = Event(datetime.now(), "purchase", {"amount": 100})
enriched = enricher.add_context(
    event,
    context={"currency": "USD", "country": "US"}
)
```

##### `add_field(event, field_name, value) -> Event`

Add single field to event.

```python
event = enricher.add_field(event, "priority", "high")
```

**Parameters**:
- `event` (Event): Event to modify
- `field_name` (str): Name of field to add
- `value` (Any): Field value

**Returns**: `Event` - Modified event

---

### Class: `EventStream`

Manages sequences of events with filtering and transformation.

```python
from analytics.events import EventStream

stream = EventStream(
    events=[event1, event2, event3],
    buffer_size=1000
)
```

#### Constructor

```python
EventStream(events=None, buffer_size=10000)
```

**Parameters**:
- `events` (list, optional): Initial events
- `buffer_size` (int): Maximum events in memory

#### Methods

##### `filter(predicate) -> EventStream`

Filter events using predicate.

```python
# Filter high-value transactions
high_value = stream.filter(
    lambda e: e.get_field("data.amount") > 1000
)

# Filter by event type
purchases = stream.filter(
    lambda e: e.event_type == "purchase"
)
```

**Parameters**:
- `predicate` (callable): Function returning bool

**Returns**: `EventStream` - Filtered stream

##### `map(transform) -> EventStream`

Transform each event.

```python
# Extract amounts
amounts = stream.map(
    lambda e: {"amount": e.get_field("data.amount")}
)

# Normalize data
normalized = stream.map(
    lambda e: Event(
        e.timestamp,
        e.event_type,
        {k: str(v).lower() for k, v in e.data.items()}
    )
)
```

**Parameters**:
- `transform` (callable): Transformation function

**Returns**: `EventStream` - Transformed stream

##### `to_list() -> list`

Convert stream to list of events.

```python
events = stream.to_list()
```

**Returns**: `list` of Event objects

---

## Streaming Module

### Class: `MetricsStreamAggregator`

Groups events by time windows and computes metrics.

```python
from analytics.streaming import MetricsStreamAggregator

aggregator = MetricsStreamAggregator(
    window_size_seconds=60,
    grace_period_seconds=10
)
```

#### Constructor

```python
MetricsStreamAggregator(window_size_seconds=60, grace_period_seconds=10)
```

**Parameters**:
- `window_size_seconds` (int): Window duration
- `grace_period_seconds` (int): Late arrival grace period

#### Methods

##### `aggregate(events) -> dict`

Aggregate events by time window.

```python
events = [event1, event2, event3, ...]
windows = aggregator.aggregate(events)

for window_id, events_in_window in windows.items():
    print(f"Window {window_id}: {len(events_in_window)} events")
```

**Parameters**:
- `events` (list): Events to aggregate

**Returns**: `dict` mapping window_id to list of events

##### `process_stream(event_stream, callback) -> None`

Process continuous event stream.

```python
def process_window(window_id, events):
    print(f"Processing {len(events)} events in window {window_id}")
    metrics = compute_metrics(events)
    store_metrics(metrics)

aggregator.process_stream(event_stream, process_window)
```

**Parameters**:
- `event_stream` (EventStream): Stream to process
- `callback` (callable): Function(window_id, events)

**Returns**: None

---

### Class: `StreamProcessor`

Processes high-volume event streams with configurable windows.

```python
from analytics.streaming import StreamProcessor

processor = StreamProcessor(
    window_type="tumbling",
    window_size_seconds=60
)
```

#### Constructor

```python
StreamProcessor(window_type="tumbling", window_size_seconds=60, overlap_seconds=0)
```

**Parameters**:
- `window_type` (str): "tumbling" or "sliding"
- `window_size_seconds` (int): Window duration
- `overlap_seconds` (int): For sliding windows

#### Methods

##### `process(events, metrics_func) -> list`

Process events and compute metrics per window.

```python
results = processor.process(
    events,
    metrics_func=lambda window_events: {
        "count": len(window_events),
        "mean": compute_mean([e.data["value"] for e in window_events])
    }
)
```

**Parameters**:
- `events` (list): Events to process
- `metrics_func` (callable): Function(events) -> metrics

**Returns**: `list` of metric dictionaries

---

## Configuration Module

### Class: `AnalyticsConfig`

Core configuration container with validation.

```python
from analytics.config import AnalyticsConfig

config = AnalyticsConfig(
    window_size_seconds=60,
    anomaly_threshold=2.5,
    storage_backend="json"
)
```

#### Constructor

```python
AnalyticsConfig(**kwargs)
```

**Common Parameters**:
- `window_size_seconds` (int): Default window size
- `grace_period_seconds` (int): Late arrival grace period
- `anomaly_threshold` (float): Z-score threshold
- `storage_backend` (str): Storage type (json, memory, db)
- `log_level` (str): Logging level (DEBUG, INFO, WARNING, ERROR)

#### Methods

##### `from_dict(config_dict) -> AnalyticsConfig`

Create config from dictionary.

```python
config_dict = {
    "window_size_seconds": 60,
    "anomaly_threshold": 2.5
}
config = AnalyticsConfig.from_dict(config_dict)
```

**Parameters**:
- `config_dict` (dict): Configuration dictionary

**Returns**: `AnalyticsConfig` instance

**Raises**: `ValueError` if validation fails

##### `from_file(file_path) -> AnalyticsConfig`

Load configuration from file.

```python
config = AnalyticsConfig.from_file("config/analytics.json")
```

**Parameters**:
- `file_path` (str): Path to config file (JSON or YAML)

**Returns**: `AnalyticsConfig` instance

**Raises**: `FileNotFoundError`, `ValueError`

##### `to_file(file_path) -> None`

Save configuration to file.

```python
config.to_file("config/analytics_backup.json")
```

**Parameters**:
- `file_path` (str): Destination file path

**Returns**: None

##### `validate() -> bool`

Validate configuration against schema.

```python
if config.validate():
    print("Configuration is valid")
```

**Returns**: `bool` - True if valid

**Raises**: `ValueError` if invalid

##### `get(key, default=None) -> Any`

Get configuration value.

```python
window_size = config.get("window_size_seconds", 60)
threshold = config.get("anomaly_threshold")
```

**Parameters**:
- `key` (str): Configuration key
- `default` (Any, optional): Default value

**Returns**: Configuration value or default

---

### Class: `DynamicConfig`

Runtime-mutable configuration.

```python
from analytics.config import DynamicConfig

dyn_config = DynamicConfig(config)
```

#### Methods

##### `update_setting(key, value) -> None`

Update configuration at runtime.

```python
dyn_config.update_setting("anomaly_threshold", 3.0)
dyn_config.update_setting("window_size_seconds", 120)
```

**Parameters**:
- `key` (str): Configuration key
- `value` (Any): New value

**Returns**: None

**Raises**: `ValueError` if validation fails

---

## Storage Module

### Class: `AnalyticsStorage`

Abstract base class for storage backends.

#### Methods

##### `save(key, data) -> None`

Persist data to storage.

```python
storage.save("metrics_2025_11_09", metrics_data)
```

**Parameters**:
- `key` (str): Data identifier
- `data` (Any): Data to store

##### `load(key) -> Any`

Retrieve data from storage.

```python
metrics = storage.load("metrics_2025_11_09")
```

**Parameters**:
- `key` (str): Data identifier

**Returns**: Stored data

**Raises**: `KeyError` if not found

##### `query(predicate) -> list`

Find data matching criteria.

```python
results = storage.query(
    lambda d: d.get("timestamp") > threshold
)
```

**Parameters**:
- `predicate` (callable): Filter function

**Returns**: `list` matching items

---

### Class: `JsonStorage`

File-based JSON storage backend.

```python
from analytics.storage import JsonStorage

storage = JsonStorage(base_path="./analytics_data")
```

#### Constructor

```python
JsonStorage(base_path="./data", create_if_missing=True)
```

**Parameters**:
- `base_path` (str): Directory for storage
- `create_if_missing` (bool): Create directory if needed

#### Methods

##### `save(key, data) -> None`

Save data to JSON file.

```python
storage.save("metrics_hourly", metrics_dict)
# Creates: ./analytics_data/metrics_hourly.json
```

##### `load(key) -> dict`

Load data from JSON file.

```python
metrics = storage.load("metrics_hourly")
```

---

### Class: `InMemoryStorage`

Fast in-memory storage for testing and caching.

```python
from analytics.storage import InMemoryStorage

storage = InMemoryStorage(max_items=10000)
```

#### Methods

Same as AnalyticsStorage (save, load, query)

---

### Class: `StorageFactory`

Factory for creating storage backends.

```python
from analytics.storage import StorageFactory

storage = StorageFactory.create("json", base_path="./data")
storage = StorageFactory.create("memory", max_items=5000)
```

#### Methods

##### `create(backend_type, **kwargs) -> AnalyticsStorage`

Create storage backend.

**Parameters**:
- `backend_type` (str): "json", "memory", "database"
- `**kwargs`: Backend-specific options

**Returns**: Storage instance

---

## Statistical Analysis

### Class: `StatisticalAnalyzer`

Core statistical computation engine.

```python
from analytics.statistical import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()
```

#### Methods

##### `compute_mean(data) -> float`

Calculate arithmetic mean.

```python
values = [10, 20, 30, 40, 50]
mean = analyzer.compute_mean(values)  # 30.0
```

##### `compute_variance(data) -> float`

Calculate variance.

```python
variance = analyzer.compute_variance(values)
```

##### `compute_std_dev(data) -> float`

Calculate standard deviation.

```python
std_dev = analyzer.compute_std_dev(values)
```

##### `compute_percentile(data, percentile) -> float`

Compute percentile value.

```python
p95 = analyzer.compute_percentile(values, 95)  # 95th percentile
p99 = analyzer.compute_percentile(values, 99)  # 99th percentile
```

---

### Class: `AnomalyDetector`

Ensemble anomaly detection with multiple methods.

```python
from analytics.anomaly_detection import AnomalyDetector

detector = AnomalyDetector(
    methods=["zscore", "iqr", "isolation_forest"],
    threshold=0.6  # 60% methods must agree
)
```

#### Constructor

```python
AnomalyDetector(methods=None, threshold=0.5)
```

**Parameters**:
- `methods` (list): Detection methods to use
- `threshold` (float): Voting threshold (0-1)

#### Methods

##### `detect(data) -> dict`

Detect anomalies in data.

```python
result = detector.detect([1, 2, 3, 100, 5, 6])

# Returns:
# {
#     "is_anomaly": True,
#     "confidence": 0.75,
#     "methods": {
#         "zscore": True,
#         "iqr": True,
#         "isolation_forest": False
#     }
# }
```

**Parameters**:
- `data` (list): Values to analyze

**Returns**: `dict` with anomaly detection results

---

## Metrics Computation

### Class: `MetricsComputer`

Aggregation and metric calculation.

```python
from analytics.metrics import MetricsComputer

computer = MetricsComputer()
```

#### Methods

##### `compute(events) -> dict`

Compute all metrics from events.

```python
events = [event1, event2, event3]
metrics = computer.compute(events)

# Returns:
# {
#     "count": 3,
#     "mean": 25.5,
#     "min": 10,
#     "max": 50,
#     "std_dev": 15.2,
#     "p50": 25,
#     "p95": 48,
#     "p99": 49
# }
```

**Parameters**:
- `events` (list): Events to process

**Returns**: `dict` of computed metrics

##### `compute_descriptive(values) -> dict`

Compute descriptive statistics.

```python
stats = computer.compute_descriptive([10, 20, 30])
```

---

## Complete Example

```python
from analytics.events import Event, EventEnricher, EventStream
from analytics.streaming import MetricsStreamAggregator
from analytics.config import AnalyticsConfig
from analytics.storage import JsonStorage
from analytics.anomaly_detection import AnomalyDetector
from datetime import datetime, timedelta

# 1. Configure
config = AnalyticsConfig.from_file("config.json")

# 2. Create events
events = [
    Event(datetime.now() - timedelta(seconds=i), "metric", {"value": 100 + i})
    for i in range(100)
]

# 3. Enrich events
enricher = EventEnricher(add_timestamp=True, add_source_info=True)
enriched_events = [enricher.add_context(e) for e in events]

# 4. Stream processing
stream = EventStream(enriched_events)
aggregator = MetricsStreamAggregator(window_size_seconds=60)
windows = aggregator.aggregate(stream.to_list())

# 5. Anomaly detection
detector = AnomalyDetector(methods=["zscore", "iqr"])
values = [e.data["value"] for e in events]
result = detector.detect(values)

# 6. Storage
storage = JsonStorage("./output")
storage.save("anomaly_results", result)

print(f"Anomaly detected: {result['is_anomaly']}")
print(f"Confidence: {result['confidence']}")
```

---

**API Version**: 1.0
**Last Updated**: November 9, 2025
**Status**: Stable
