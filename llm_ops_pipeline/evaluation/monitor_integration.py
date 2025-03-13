"""
Integration module connecting the LLM Evaluation Framework with the monitoring system.
Allows evaluation results to be exported to Prometheus metrics and visualized in Grafana.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Union
import json

from prometheus_client import Counter, Gauge, Histogram, Summary, push_to_gateway
from prometheus_client import CollectorRegistry, REGISTRY

from llm_ops_pipeline.evaluation.llm_evaluation_framework import EvaluationResult
from llm_ops_pipeline.utils.logging import setup_logger

# Set up logger
logger = setup_logger(name="monitor_integration")

class EvaluationMonitor:
    """
    Connects the evaluation framework with the monitoring system.
    Exports evaluation metrics to Prometheus for runtime monitoring.
    """
    
    def __init__(
        self,
        namespace: str = "llm_evaluation",
        subsystem: str = "metrics",
        registry: Optional[CollectorRegistry] = None,
        push_gateway_url: Optional[str] = None,
        export_interval: int = 15  # seconds
    ):
        """
        Initialize the evaluation monitor.
        
        Args:
            namespace: Prometheus metric namespace
            subsystem: Prometheus metric subsystem
            registry: Custom Prometheus registry (if None, uses default registry)
            push_gateway_url: Prometheus push gateway URL (if None, uses direct exposure)
            export_interval: Interval for pushing metrics to gateway (if used)
        """
        self.namespace = namespace
        self.subsystem = subsystem
        self.registry = registry or REGISTRY
        self.push_gateway_url = push_gateway_url
        self.export_interval = export_interval
        self.job_name = "llm_evaluation_monitor"
        
        # Initialize metric collections
        self._init_metrics()
        
        # Last export timestamp
        self.last_export_time = 0
    
    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        # Accuracy metrics
        self.accuracy_gauge = Gauge(
            f"{self.namespace}_{self.subsystem}_accuracy",
            "Accuracy of LLM evaluations",
            ["prompt_id", "model_id", "task_id"],
            registry=self.registry
        )
        
        self.f1_score_gauge = Gauge(
            f"{self.namespace}_{self.subsystem}_f1_score",
            "F1 score of LLM evaluations",
            ["prompt_id", "model_id", "task_id", "type"],  # type: macro, micro, or class name
            registry=self.registry
        )
        
        # Latency metrics
        self.latency_histogram = Histogram(
            f"{self.namespace}_{self.subsystem}_latency_ms",
            "Latency of LLM evaluations in milliseconds",
            ["prompt_id", "model_id", "task_id"],
            buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
            registry=self.registry
        )
        
        # Token usage metrics
        self.token_counter = Counter(
            f"{self.namespace}_{self.subsystem}_tokens_total",
            "Total number of tokens used in LLM evaluations",
            ["prompt_id", "model_id", "task_id", "token_type"],  # token_type: prompt, completion, total
            registry=self.registry
        )
        
        # Cost metrics
        self.cost_counter = Counter(
            f"{self.namespace}_{self.subsystem}_cost_usd",
            "Estimated cost of LLM evaluations in USD",
            ["prompt_id", "model_id", "task_id"],
            registry=self.registry
        )
        
        # Examples count
        self.examples_gauge = Gauge(
            f"{self.namespace}_{self.subsystem}_examples_count",
            "Number of examples evaluated",
            ["prompt_id", "model_id", "task_id", "status"],  # status: valid, invalid, total
            registry=self.registry
        )
        
        # Evaluation run counter
        self.eval_run_counter = Counter(
            f"{self.namespace}_{self.subsystem}_runs_total",
            "Total number of evaluation runs",
            ["task_id"],
            registry=self.registry
        )
    
    def export_results(self, results: Union[EvaluationResult, List[EvaluationResult], Dict[str, EvaluationResult]]):
        """
        Export evaluation results to Prometheus metrics.
        
        Args:
            results: Evaluation result(s) to export
        """
        # Convert results to list for uniform handling
        results_list = []
        
        if isinstance(results, EvaluationResult):
            results_list = [results]
        elif isinstance(results, dict):
            results_list = list(results.values())
        elif isinstance(results, list):
            results_list = results
        else:
            logger.error(f"Unsupported results type: {type(results)}")
            return
        
        # Process each result
        for result in results_list:
            self._export_result(result)
        
        # Push to gateway if configured
        if self.push_gateway_url and time.time() - self.last_export_time >= self.export_interval:
            try:
                if self.registry != REGISTRY:
                    push_to_gateway(
                        self.push_gateway_url,
                        job=self.job_name,
                        registry=self.registry
                    )
                else:
                    push_to_gateway(
                        self.push_gateway_url,
                        job=self.job_name
                    )
                self.last_export_time = time.time()
            except Exception as e:
                logger.error(f"Failed to push metrics to gateway: {e}")
    
    def _export_result(self, result: EvaluationResult):
        """Export a single evaluation result to Prometheus metrics."""
        # Get labels
        prompt_id = result.prompt_id or "unknown"
        model_id = result.model_id or "unknown"
        task_id = result.task_id or "unknown"
        
        # Increment evaluation run counter
        self.eval_run_counter.labels(task_id=task_id).inc()
        
        # Export accuracy metrics
        if "accuracy" in result.metrics:
            self.accuracy_gauge.labels(
                prompt_id=prompt_id,
                model_id=model_id,
                task_id=task_id
            ).set(result.metrics["accuracy"])
        
        # Export F1 score metrics
        for metric_name, metric_value in result.metrics.items():
            if metric_name.startswith("f1"):
                metric_type = metric_name.replace("f1_", "")
                self.f1_score_gauge.labels(
                    prompt_id=prompt_id,
                    model_id=model_id,
                    task_id=task_id,
                    type=metric_type
                ).set(metric_value)
            # Handle per-class F1 scores
            elif isinstance(metric_value, dict) and metric_name == "per_class":
                for class_name, class_metrics in metric_value.items():
                    if "f1" in class_metrics:
                        self.f1_score_gauge.labels(
                            prompt_id=prompt_id,
                            model_id=model_id,
                            task_id=task_id,
                            type=class_name
                        ).set(class_metrics["f1"])
        
        # Export latency metrics
        if result.latency_ms:
            self.latency_histogram.labels(
                prompt_id=prompt_id,
                model_id=model_id,
                task_id=task_id
            ).observe(result.latency_ms)
        
        # Export token usage metrics
        for token_type, token_count in result.token_usage.items():
            self.token_counter.labels(
                prompt_id=prompt_id,
                model_id=model_id,
                task_id=task_id,
                token_type=token_type
            ).inc(token_count)
        
        # Export cost metrics if available
        if "cost_estimate_usd" in result.metrics:
            self.cost_counter.labels(
                prompt_id=prompt_id,
                model_id=model_id,
                task_id=task_id
            ).inc(result.metrics["cost_estimate_usd"])
        
        # Export examples count
        valid_examples = len([ex for ex in result.examples if ex.get("prediction") is not None])
        invalid_examples = len(result.examples) - valid_examples
        
        self.examples_gauge.labels(
            prompt_id=prompt_id,
            model_id=model_id,
            task_id=task_id,
            status="valid"
        ).set(valid_examples)
        
        self.examples_gauge.labels(
            prompt_id=prompt_id,
            model_id=model_id,
            task_id=task_id,
            status="invalid"
        ).set(invalid_examples)
        
        self.examples_gauge.labels(
            prompt_id=prompt_id,
            model_id=model_id,
            task_id=task_id,
            status="total"
        ).set(len(result.examples))
    
    def create_grafana_dashboard(self, dashboard_path: str, dashboard_title: str = "LLM Evaluation Metrics"):
        """
        Create a Grafana dashboard configuration for LLM evaluation metrics.
        
        Args:
            dashboard_path: Path to save the dashboard JSON
            dashboard_title: Title for the dashboard
            
        Returns:
            Dashboard configuration as dictionary
        """
        dashboard = {
            "annotations": {
                "list": [
                    {
                        "builtIn": 1,
                        "datasource": "-- Grafana --",
                        "enable": True,
                        "hide": True,
                        "iconColor": "rgba(0, 211, 255, 1)",
                        "name": "Annotations & Alerts",
                        "type": "dashboard"
                    }
                ]
            },
            "editable": True,
            "gnetId": None,
            "graphTooltip": 0,
            "id": None,
            "links": [],
            "panels": [],
            "refresh": "10s",
            "schemaVersion": 16,
            "style": "dark",
            "tags": ["llm", "evaluation", "metrics"],
            "templating": {
                "list": [
                    {
                        "allValue": None,
                        "current": {
                            "tags": [],
                            "text": "All",
                            "value": ["$__all"]
                        },
                        "datasource": "Prometheus",
                        "definition": f'label_values({self.namespace}_{self.subsystem}_accuracy, prompt_id)',
                        "hide": 0,
                        "includeAll": True,
                        "label": "Prompt",
                        "multi": True,
                        "name": "prompt_id",
                        "options": [],
                        "query": {
                            "query": f'label_values({self.namespace}_{self.subsystem}_accuracy, prompt_id)',
                            "refId": "StandardVariableQuery"
                        },
                        "refresh": 1,
                        "regex": "",
                        "skipUrlSync": False,
                        "sort": 0,
                        "tagValuesQuery": "",
                        "tags": [],
                        "tagsQuery": "",
                        "type": "query",
                        "useTags": False
                    },
                    {
                        "allValue": None,
                        "current": {
                            "tags": [],
                            "text": "All",
                            "value": ["$__all"]
                        },
                        "datasource": "Prometheus",
                        "definition": f'label_values({self.namespace}_{self.subsystem}_accuracy, model_id)',
                        "hide": 0,
                        "includeAll": True,
                        "label": "Model",
                        "multi": True,
                        "name": "model_id",
                        "options": [],
                        "query": {
                            "query": f'label_values({self.namespace}_{self.subsystem}_accuracy, model_id)',
                            "refId": "StandardVariableQuery"
                        },
                        "refresh": 1,
                        "regex": "",
                        "skipUrlSync": False,
                        "sort": 0,
                        "tagValuesQuery": "",
                        "tags": [],
                        "tagsQuery": "",
                        "type": "query",
                        "useTags": False
                    },
                    {
                        "allValue": None,
                        "current": {
                            "tags": [],
                            "text": "All",
                            "value": ["$__all"]
                        },
                        "datasource": "Prometheus",
                        "definition": f'label_values({self.namespace}_{self.subsystem}_accuracy, task_id)',
                        "hide": 0,
                        "includeAll": True,
                        "label": "Task",
                        "multi": True,
                        "name": "task_id",
                        "options": [],
                        "query": {
                            "query": f'label_values({self.namespace}_{self.subsystem}_accuracy, task_id)',
                            "refId": "StandardVariableQuery"
                        },
                        "refresh": 1,
                        "regex": "",
                        "skipUrlSync": False,
                        "sort": 0,
                        "tagValuesQuery": "",
                        "tags": [],
                        "tagsQuery": "",
                        "type": "query",
                        "useTags": False
                    }
                ]
            },
            "time": {
                "from": "now-6h",
                "to": "now"
            },
            "timepicker": {
                "refresh_intervals": [
                    "5s",
                    "10s",
                    "30s",
                    "1m",
                    "5m",
                    "15m",
                    "30m",
                    "1h",
                    "2h",
                    "1d"
                ],
                "time_options": [
                    "5m",
                    "15m",
                    "1h",
                    "6h",
                    "12h",
                    "24h",
                    "2d",
                    "7d",
                    "30d"
                ]
            },
            "timezone": "",
            "title": dashboard_title,
            "uid": None,
            "version": 0
        }
        
        # Add panels
        panels = []
        
        # Accuracy metrics panel
        panels.append({
            "datasource": "Prometheus",
            "fieldConfig": {
                "defaults": {
                    "color": {
                        "mode": "palette-classic"
                    },
                    "custom": {
                        "axisLabel": "",
                        "axisPlacement": "auto",
                        "barAlignment": 0,
                        "drawStyle": "line",
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "hideFrom": {
                            "legend": False,
                            "tooltip": False,
                            "viz": False
                        },
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "pointSize": 5,
                        "scaleDistribution": {
                            "type": "linear"
                        },
                        "showPoints": "never",
                        "spanNulls": True,
                        "stacking": {
                            "group": "A",
                            "mode": "none"
                        },
                        "thresholdsStyle": {
                            "mode": "off"
                        }
                    },
                    "mappings": [],
                    "max": 1,
                    "min": 0,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {
                                "color": "green",
                                "value": None
                            }
                        ]
                    },
                    "unit": "percentunit"
                },
                "overrides": []
            },
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 0,
                "y": 0
            },
            "id": 1,
            "options": {
                "legend": {
                    "calcs": [],
                    "displayMode": "list",
                    "placement": "bottom"
                },
                "tooltip": {
                    "mode": "single"
                }
            },
            "targets": [
                {
                    "expr": f"{self.namespace}_{self.subsystem}_accuracy{{prompt_id=~\"$prompt_id\", model_id=~\"$model_id\", task_id=~\"$task_id\"}}",
                    "interval": "",
                    "legendFormat": "{{prompt_id}} - {{model_id}} - {{task_id}}",
                    "refId": "A"
                }
            ],
            "title": "Accuracy",
            "type": "timeseries"
        })
        
        # F1 Score panel
        panels.append({
            "datasource": "Prometheus",
            "fieldConfig": {
                "defaults": {
                    "color": {
                        "mode": "palette-classic"
                    },
                    "custom": {
                        "axisLabel": "",
                        "axisPlacement": "auto",
                        "barAlignment": 0,
                        "drawStyle": "line",
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "hideFrom": {
                            "legend": False,
                            "tooltip": False,
                            "viz": False
                        },
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "pointSize": 5,
                        "scaleDistribution": {
                            "type": "linear"
                        },
                        "showPoints": "never",
                        "spanNulls": True,
                        "stacking": {
                            "group": "A",
                            "mode": "none"
                        },
                        "thresholdsStyle": {
                            "mode": "off"
                        }
                    },
                    "mappings": [],
                    "max": 1,
                    "min": 0,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {
                                "color": "green",
                                "value": None
                            }
                        ]
                    },
                    "unit": "percentunit"
                },
                "overrides": []
            },
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 12,
                "y": 0
            },
            "id": 2,
            "options": {
                "legend": {
                    "calcs": [],
                    "displayMode": "list",
                    "placement": "bottom"
                },
                "tooltip": {
                    "mode": "single"
                }
            },
            "targets": [
                {
                    "expr": f"{self.namespace}_{self.subsystem}_f1_score{{prompt_id=~\"$prompt_id\", model_id=~\"$model_id\", task_id=~\"$task_id\", type=\"macro\"}}",
                    "interval": "",
                    "legendFormat": "{{prompt_id}} - {{model_id}} - {{task_id}}",
                    "refId": "A"
                }
            ],
            "title": "F1 Score (Macro)",
            "type": "timeseries"
        })
        
        # Latency panel
        panels.append({
            "datasource": "Prometheus",
            "fieldConfig": {
                "defaults": {
                    "color": {
                        "mode": "palette-classic"
                    },
                    "custom": {
                        "axisLabel": "",
                        "axisPlacement": "auto",
                        "barAlignment": 0,
                        "drawStyle": "line",
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "hideFrom": {
                            "legend": False,
                            "tooltip": False,
                            "viz": False
                        },
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "pointSize": 5,
                        "scaleDistribution": {
                            "type": "linear"
                        },
                        "showPoints": "never",
                        "spanNulls": True,
                        "stacking": {
                            "group": "A",
                            "mode": "none"
                        },
                        "thresholdsStyle": {
                            "mode": "off"
                        }
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {
                                "color": "green",
                                "value": None
                            }
                        ]
                    },
                    "unit": "ms"
                },
                "overrides": []
            },
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 0,
                "y": 8
            },
            "id": 3,
            "options": {
                "legend": {
                    "calcs": [],
                    "displayMode": "list",
                    "placement": "bottom"
                },
                "tooltip": {
                    "mode": "single"
                }
            },
            "targets": [
                {
                    "expr": f"histogram_quantile(0.95, sum(rate({self.namespace}_{self.subsystem}_latency_ms_bucket{{prompt_id=~\"$prompt_id\", model_id=~\"$model_id\", task_id=~\"$task_id\"}}[5m])) by (le, prompt_id, model_id, task_id))",
                    "interval": "",
                    "legendFormat": "p95 - {{prompt_id}} - {{model_id}} - {{task_id}}",
                    "refId": "A"
                },
                {
                    "expr": f"histogram_quantile(0.50, sum(rate({self.namespace}_{self.subsystem}_latency_ms_bucket{{prompt_id=~\"$prompt_id\", model_id=~\"$model_id\", task_id=~\"$task_id\"}}[5m])) by (le, prompt_id, model_id, task_id))",
                    "interval": "",
                    "legendFormat": "p50 - {{prompt_id}} - {{model_id}} - {{task_id}}",
                    "refId": "B"
                }
            ],
            "title": "Latency",
            "type": "timeseries"
        })
        
        # Token usage panel
        panels.append({
            "datasource": "Prometheus",
            "fieldConfig": {
                "defaults": {
                    "color": {
                        "mode": "palette-classic"
                    },
                    "custom": {
                        "axisLabel": "",
                        "axisPlacement": "auto",
                        "barAlignment": 0,
                        "drawStyle": "line",
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "hideFrom": {
                            "legend": False,
                            "tooltip": False,
                            "viz": False
                        },
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "pointSize": 5,
                        "scaleDistribution": {
                            "type": "linear"
                        },
                        "showPoints": "never",
                        "spanNulls": True,
                        "stacking": {
                            "group": "A",
                            "mode": "none"
                        },
                        "thresholdsStyle": {
                            "mode": "off"
                        }
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {
                                "color": "green",
                                "value": None
                            }
                        ]
                    },
                    "unit": "none"
                },
                "overrides": []
            },
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 12,
                "y": 8
            },
            "id": 4,
            "options": {
                "legend": {
                    "calcs": [],
                    "displayMode": "list",
                    "placement": "bottom"
                },
                "tooltip": {
                    "mode": "single"
                }
            },
            "targets": [
                {
                    "expr": f"sum(rate({self.namespace}_{self.subsystem}_tokens_total{{prompt_id=~\"$prompt_id\", model_id=~\"$model_id\", task_id=~\"$task_id\", token_type=\"total\"}}[5m])) by (prompt_id, model_id, task_id)",
                    "interval": "",
                    "legendFormat": "{{prompt_id}} - {{model_id}} - {{task_id}}",
                    "refId": "A"
                }
            ],
            "title": "Token Usage (Total)",
            "type": "timeseries"
        })
        
        # Cost panel
        panels.append({
            "datasource": "Prometheus",
            "fieldConfig": {
                "defaults": {
                    "color": {
                        "mode": "palette-classic"
                    },
                    "custom": {
                        "axisLabel": "",
                        "axisPlacement": "auto",
                        "barAlignment": 0,
                        "drawStyle": "line",
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "hideFrom": {
                            "legend": False,
                            "tooltip": False,
                            "viz": False
                        },
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "pointSize": 5,
                        "scaleDistribution": {
                            "type": "linear"
                        },
                        "showPoints": "never",
                        "spanNulls": True,
                        "stacking": {
                            "group": "A",
                            "mode": "none"
                        },
                        "thresholdsStyle": {
                            "mode": "off"
                        }
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {
                                "color": "green",
                                "value": None
                            }
                        ]
                    },
                    "unit": "currencyUSD"
                },
                "overrides": []
            },
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 0,
                "y": 16
            },
            "id": 5,
            "options": {
                "legend": {
                    "calcs": [],
                    "displayMode": "list",
                    "placement": "bottom"
                },
                "tooltip": {
                    "mode": "single"
                }
            },
            "targets": [
                {
                    "expr": f"sum(rate({self.namespace}_{self.subsystem}_cost_usd{{prompt_id=~\"$prompt_id\", model_id=~\"$model_id\", task_id=~\"$task_id\"}}[5m])) by (prompt_id, model_id, task_id)",
                    "interval": "",
                    "legendFormat": "{{prompt_id}} - {{model_id}} - {{task_id}}",
                    "refId": "A"
                }
            ],
            "title": "Cost (USD)",
            "type": "timeseries"
        })
        
        # Example count panel
        panels.append({
            "datasource": "Prometheus",
            "fieldConfig": {
                "defaults": {
                    "color": {
                        "mode": "palette-classic"
                    },
                    "custom": {
                        "axisLabel": "",
                        "axisPlacement": "auto",
                        "barAlignment": 0,
                        "drawStyle": "line",
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "hideFrom": {
                            "legend": False,
                            "tooltip": False,
                            "viz": False
                        },
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "pointSize": 5,
                        "scaleDistribution": {
                            "type": "linear"
                        },
                        "showPoints": "never",
                        "spanNulls": True,
                        "stacking": {
                            "group": "A",
                            "mode": "none"
                        },
                        "thresholdsStyle": {
                            "mode": "off"
                        }
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {
                                "color": "green",
                                "value": None
                            }
                        ]
                    },
                    "unit": "none"
                },
                "overrides": []
            },
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 12,
                "y": 16
            },
            "id": 6,
            "options": {
                "legend": {
                    "calcs": [],
                    "displayMode": "list",
                    "placement": "bottom"
                },
                "tooltip": {
                    "mode": "single"
                }
            },
            "targets": [
                {
                    "expr": f"{self.namespace}_{self.subsystem}_examples_count{{prompt_id=~\"$prompt_id\", model_id=~\"$model_id\", task_id=~\"$task_id\", status=\"valid\"}}",
                    "interval": "",
                    "legendFormat": "Valid - {{prompt_id}} - {{model_id}} - {{task_id}}",
                    "refId": "A"
                },
                {
                    "expr": f"{self.namespace}_{self.subsystem}_examples_count{{prompt_id=~\"$prompt_id\", model_id=~\"$model_id\", task_id=~\"$task_id\", status=\"invalid\"}}",
                    "interval": "",
                    "legendFormat": "Invalid - {{prompt_id}} - {{model_id}} - {{task_id}}",
                    "refId": "B"
                }
            ],
            "title": "Examples Count",
            "type": "timeseries"
        })
        
        # Add panels to dashboard
        dashboard["panels"] = panels
        
        # Save dashboard JSON
        if dashboard_path:
            os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)
            with open(dashboard_path, "w") as f:
                json.dump(dashboard, f, indent=2)
        
        return dashboard