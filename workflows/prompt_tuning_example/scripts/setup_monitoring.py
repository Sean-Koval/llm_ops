#!/usr/bin/env python
"""
Set up monitoring for the LLM evaluation framework.
Creates Grafana dashboards and Prometheus configuration for tracking metrics.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from llm_ops_pipeline.utils.logging import setup_logger
from llm_ops_pipeline.evaluation.monitor_integration import EvaluationMonitor

# Set up logger
logger = setup_logger(name="setup_monitoring")

def setup_grafana_dashboard(args, config: Dict[str, Any]):
    """Set up Grafana dashboard for evaluation metrics."""
    # Initialize evaluation monitor
    monitor = EvaluationMonitor(
        namespace="llm_evaluation",
        subsystem="metrics",
        push_gateway_url=config.get("monitoring", {}).get("push_gateway_url")
    )
    
    # Determine dashboard path
    if args.dashboard_path:
        dashboard_path = args.dashboard_path
    else:
        dashboard_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "monitoring",
            "grafana",
            "dashboards",
            "llm_evaluation_dashboard.json"
        )
    
    # Create dashboard title
    dashboard_title = args.dashboard_title or f"LLM Evaluation Metrics - {args.prompt_id} / {args.model_id}"
    
    # Create dashboard
    dashboard = monitor.create_grafana_dashboard(
        dashboard_path=dashboard_path,
        dashboard_title=dashboard_title
    )
    
    logger.info(f"Created Grafana dashboard at {dashboard_path}")
    
    # Create dashboard provisioning config if requested
    if args.create_provisioning:
        provisioning_dir = os.path.dirname(dashboard_path)
        provisioning_path = os.path.join(provisioning_dir, "dashboard.yaml")
        
        provisioning_config = {
            "apiVersion": 1,
            "providers": [
                {
                    "name": "LLM Evaluation Dashboards",
                    "orgId": 1,
                    "folder": "LLM Evaluation",
                    "type": "file",
                    "disableDeletion": False,
                    "updateIntervalSeconds": 10,
                    "allowUiUpdates": True,
                    "options": {
                        "path": os.path.dirname(dashboard_path),
                        "foldersFromFilesStructure": True
                    }
                }
            ]
        }
        
        with open(provisioning_path, "w") as f:
            yaml.dump(provisioning_config, f, default_flow_style=False)
        
        logger.info(f"Created Grafana dashboard provisioning config at {provisioning_path}")

def setup_prometheus_config(args, config: Dict[str, Any]):
    """Set up Prometheus configuration for scraping metrics."""
    if not args.prometheus_config_path:
        logger.warning("Prometheus config path not provided, skipping configuration")
        return
    
    prometheus_config_path = args.prometheus_config_path
    
    # Create basic Prometheus config if it doesn't exist
    if not os.path.exists(prometheus_config_path):
        prometheus_config = {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "scrape_configs": [
                {
                    "job_name": "prometheus",
                    "static_configs": [
                        {
                            "targets": ["localhost:9090"]
                        }
                    ]
                },
                {
                    "job_name": "llm_evaluation",
                    "scrape_interval": "10s",
                    "static_configs": [
                        {
                            "targets": ["localhost:8000"]
                        }
                    ]
                }
            ]
        }
        
        # Add push gateway if configured
        push_gateway_url = config.get("monitoring", {}).get("push_gateway_url")
        if push_gateway_url:
            push_gateway_host = push_gateway_url.split("://")[-1]
            prometheus_config["scrape_configs"].append({
                "job_name": "pushgateway",
                "scrape_interval": "5s",
                "honor_labels": True,
                "static_configs": [
                    {
                        "targets": [push_gateway_host]
                    }
                ]
            })
        
        with open(prometheus_config_path, "w") as f:
            yaml.dump(prometheus_config, f, default_flow_style=False)
        
        logger.info(f"Created new Prometheus config at {prometheus_config_path}")
    else:
        # Update existing config
        with open(prometheus_config_path, "r") as f:
            prometheus_config = yaml.safe_load(f)
        
        # Check if LLM evaluation job already exists
        llm_job_exists = False
        for job in prometheus_config.get("scrape_configs", []):
            if job.get("job_name") == "llm_evaluation":
                llm_job_exists = True
                break
        
        # Add LLM evaluation job if not exists
        if not llm_job_exists:
            prometheus_config.setdefault("scrape_configs", []).append({
                "job_name": "llm_evaluation",
                "scrape_interval": "10s",
                "static_configs": [
                    {
                        "targets": ["localhost:8000"]
                    }
                ]
            })
            
            with open(prometheus_config_path, "w") as f:
                yaml.dump(prometheus_config, f, default_flow_style=False)
            
            logger.info(f"Updated Prometheus config at {prometheus_config_path}")
        else:
            logger.info(f"Prometheus config already contains LLM evaluation job")

def main():
    parser = argparse.ArgumentParser(description="Set up monitoring for LLM evaluation")
    parser.add_argument("--config", type=str, default="../configs/evaluation_config.yaml",
                        help="Path to evaluation configuration file")
    parser.add_argument("--prompt-id", type=str, required=True,
                        help="ID of the prompt to monitor")
    parser.add_argument("--model-id", type=str, required=True,
                        help="ID of the model to monitor")
    parser.add_argument("--dashboard-path", type=str,
                        help="Path to save Grafana dashboard JSON")
    parser.add_argument("--dashboard-title", type=str,
                        help="Title for Grafana dashboard")
    parser.add_argument("--prometheus-config-path", type=str,
                        help="Path to Prometheus configuration file")
    parser.add_argument("--create-provisioning", action="store_true",
                        help="Create Grafana dashboard provisioning configuration")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            if args.config.endswith(".yaml") or args.config.endswith(".yml"):
                config = yaml.safe_load(f)
            elif args.config.endswith(".json"):
                config = json.load(f)
            else:
                logger.error(f"Unsupported configuration file format: {args.config}")
                sys.exit(1)
    else:
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Set up Grafana dashboard
    setup_grafana_dashboard(args, config)
    
    # Set up Prometheus configuration
    setup_prometheus_config(args, config)
    
    logger.info("Monitoring setup completed successfully")

if __name__ == "__main__":
    main()