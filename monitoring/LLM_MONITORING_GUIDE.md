# LLM Monitoring and Observability Guide

This guide explains how to use and extend the LLM monitoring system implemented in the LLM Ops Pipeline.

## Overview

The monitoring system tracks LLM-specific metrics using Prometheus and visualizes them in Grafana dashboards. It provides visibility into:

- Request rates and latencies
- Token usage and estimated costs
- Error rates and types
- System resource usage (GPU memory)
- Model parameters tracking

## Architecture

The monitoring system consists of:

1. **Metrics Collection**:
   - Prometheus client instrumentation in the API and inference code
   - Custom metrics for LLM-specific metrics (tokens, costs, etc.)
   - Automatic API endpoint instrumentation

2. **Storage**:
   - Prometheus time-series database
   - Retention according to Prometheus defaults

3. **Visualization**:
   - Grafana dashboards
   - Pre-configured visualizations for key LLM metrics

## Available Metrics

The following metrics are collected:

### Request Metrics
- `llm_request_total` - Total number of LLM requests (labels: endpoint, model, status)
- `llm_request_latency_seconds` - Latency histogram (labels: endpoint, model)

### Token Usage Metrics
- `llm_prompt_tokens_total` - Total prompt tokens (labels: model)
- `llm_completion_tokens_total` - Total completion tokens (labels: model)
- `llm_tokens_total` - Total tokens (prompt + completion) (labels: model)

### Cost Metrics
- `llm_estimated_cost_usd_total` - Estimated cost in USD (labels: model)

### Error Metrics
- `llm_error_total` - Error count by type (labels: endpoint, model, error_type)

### System Metrics
- `llm_gpu_memory_bytes` - GPU memory usage (labels: device)

### Model Parameter Metrics
- `llm_temperature_distribution` - Distribution of temperature values (labels: model)

### Cache Metrics
- `llm_cache_hit_total` - Cache hit count (labels: model)
- `llm_cache_miss_total` - Cache miss count (labels: model)

### Quality Metrics
- `llm_quality_score` - Quality score if available (labels: model, metric_type)

## Dashboards

The main dashboard is **LLM Observability Dashboard**, which provides:

- Request overview (rates, success/failure)
- Latency tracking (p50, p95)
- Token usage monitoring
- Cost estimation
- Error analysis
- System resource monitoring

## How to Use

### Accessing Metrics

1. **Prometheus Metrics Endpoint**:
   - The API exposes a `/metrics` endpoint that returns all metrics in Prometheus format
   - This endpoint is automatically scraped by Prometheus every 15 seconds

2. **API Response Metrics**:
   - Each `/generate` API response includes a `metrics` field with request-specific metrics
   - These metrics include token counts, processing time, and estimated cost

### Viewing Dashboards

1. Access Grafana at `http://<host>:3000`
2. Navigate to Dashboards → LLM Monitoring → LLM Observability Dashboard
3. Use the time range selector to zoom in on specific periods

## Extending the Monitoring System

### Adding New Metrics

1. Define new metrics in `llm_ops_pipeline/utils/monitoring.py`
2. Instrument the code where these metrics should be collected
3. Add the metrics to the Grafana dashboard

Example for adding a new metric:

```python
# In monitoring.py
NEW_METRIC = Counter(
    "llm_new_metric_total",
    "Description of the new metric",
    ["label1", "label2"],
    registry=REGISTRY
)

# In your code
from llm_ops_pipeline.utils.monitoring import NEW_METRIC

NEW_METRIC.labels(label1="value1", label2="value2").inc()
```

### Creating Custom Dashboards

1. Create a new dashboard in Grafana
2. Use PromQL queries to visualize the metrics
3. Save the dashboard JSON to `/monitoring/grafana/dashboards/`

### Alerting

1. Configure alerting rules in Prometheus
2. Set up notification channels in Grafana
3. Add alert conditions to dashboard panels

Example Prometheus alert rule:

```yaml
groups:
- name: LLM Alerts
  rules:
  - alert: HighErrorRate
    expr: sum(rate(llm_error_total[5m])) / sum(rate(llm_request_total[5m])) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is above 10% for 5 minutes"
```

## Best Practices

1. **Monitoring Hygiene**:
   - Only collect metrics that provide actionable insights
   - Use appropriate metric types (counter, gauge, histogram)
   - Label metrics correctly for effective querying

2. **Dashboard Design**:
   - Group related metrics together
   - Use appropriate visualization types
   - Include both high-level overviews and detailed breakdowns

3. **Operational Usage**:
   - Regularly review dashboards to identify trends
   - Set up alerts for critical thresholds
   - Use metrics to inform scaling decisions and cost optimization

## Troubleshooting

### Missing Metrics

1. Check if the API is running and accessible
2. Verify that Prometheus is scraping the `/metrics` endpoint
3. Inspect the Prometheus target status in the Prometheus UI

### Dashboard Issues

1. Verify that Grafana can connect to Prometheus
2. Check that the queries in the dashboard panels are correctly formatted
3. Ensure the time range selected includes data

## Future Enhancements

1. **Advanced LLM Metrics**:
   - Hallucination detection measurements
   - Semantic similarity scoring
   - Relevance metrics for responses

2. **Integration with External Services**:
   - OpenAI API usage monitoring
   - Azure OpenAI monitoring
   - Anthropic Claude monitoring

3. **Advanced Visualizations**:
   - Heat maps for token usage patterns
   - Anomaly detection visualizations
   - Cost projection tools